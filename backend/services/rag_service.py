from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from services.onnx_embeddings_service import get_onnx_embeddings
from config import settings
import logging
import os
import time

logger = logging.getLogger(__name__)

class RAGService:
    """Manages RAG operations with ChromaDB."""
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def initialize(self):
        """Initialize embeddings and vector store."""
        if self.vectorstore is not None:
            return
        
        logger.info("Initializing RAG service...")
        
        # Initialize ONNX embeddings (preloaded by get_onnx_embeddings)
        self.embeddings = get_onnx_embeddings()
        logger.info(f"Using ONNX embeddings from {settings.ONNX_EMBEDDING_MODEL_PATH}")
        
        # Initialize or load ChromaDB
        self.vectorstore = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )
        
        logger.info("RAG service initialized successfully")
    
    def ingest_document(self, file_path: str, file_type: str) -> int:
        """Ingest a document into ChromaDB."""
        self.initialize()
        
        logger.info(f"Ingesting document: {file_path}")
        
        # Load document based on type
        if file_type == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_type in ['txt', 'text']:
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Load and split
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        
        # Add to vectorstore
        self.vectorstore.add_documents(chunks)
        
        logger.info(f"Ingested {len(chunks)} chunks from {file_path}")
        return len(chunks)
    
    def retrieve_context(self, query: str, k: int = None) -> tuple[str, list]:
        """Retrieve relevant context from ChromaDB."""
        self.initialize()
        
        k = k or settings.RAG_TOP_K
        
        t_start = time.time()
        
        try:
            # Embeddings + similarity search
            docs = self.vectorstore.similarity_search(query, k=k)
            
            t_end = time.time()
            logger.info(f"   🔍 RAG Retrieval: {round((t_end - t_start) * 1000, 2)} ms - Retrieved {len(docs)} documents")
            
            if not docs:
                return "No relevant background information found.", []
            
            # Build context string
            context_parts = []
            sources = []
            
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'unknown')
                sources.append(source)
                context_parts.append(f"[Source {i}: {os.path.basename(source)}]\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            return context, sources
            
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            return "Error retrieving background information.", []
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        self.initialize()
        
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            return {
                "collection_name": settings.COLLECTION_NAME,
                "document_count": count
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

# Global instance
rag_service = RAGService()
