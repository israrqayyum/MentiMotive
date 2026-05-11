from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import List
from models.schemas import IngestResponse, RAGCollectionStatsResponse
from services.rag_service import rag_service
from config import settings
import logging
import tempfile
import os

router = APIRouter(prefix="/chat", tags=["RAG"])
logger = logging.getLogger(__name__)


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Upload PDF/TXT to the vector store",
    description="Multipart: field name **`files`**, one or more `pdf` or `txt` files (in Swagger, add multiple `files` rows). "
    "Non‑PDF/TXT uploads are logged and **skipped** (not counted in `files_processed`). "
    "The Chroma collection name is returned in the **response** field `collection` (config-driven; not a request field).",
    responses={500: {"description": "Unexpected ingestion or filesystem error; partial chunks may be missing for some files."}},
)
async def ingest_documents(
    files: List[UploadFile] = File(
        ...,
        description="One or more `pdf` or `txt` documents. In Swagger UI, use **Add string item** under `files` to add multiple file rows.",
    )
) -> IngestResponse:
    try:
        total_chunks = 0
        files_processed = 0

        for file in files:
            file_ext = file.filename.split(".")[-1].lower()

            if file_ext not in ("pdf", "txt"):
                logger.warning(f"Skipping unsupported file: {file.filename}")
                continue

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{file_ext}"
            ) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_path = temp_file.name

            try:
                chunks = rag_service.ingest_document(temp_path, file_ext)
                total_chunks += chunks
                files_processed += 1
                logger.info(f"Ingested {file.filename}: {chunks} chunks")
            except Exception as e:
                logger.error(f"Error ingesting {file.filename}: {e}")
            finally:
                os.unlink(temp_path)

        return IngestResponse(
            status="success",
            files_processed=files_processed,
            chunks_created=total_chunks,
            collection=settings.COLLECTION_NAME,
        )

    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e


@router.get(
    "/collection/stats",
    response_model=RAGCollectionStatsResponse,
    summary="RAG index size and name",
    description="Returns the Chroma collection name and a row count, or in rare cases an `error` field from the RAG service without raising HTTP 500 (check the body).",
    responses={500: {"description": "Router-level failure when stats cannot be obtained at all."}},
)
async def get_collection_stats() -> RAGCollectionStatsResponse:
    try:
        stats = rag_service.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Collection stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e
