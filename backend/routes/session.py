from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import logging

from db.database import get_db
from models.schemas import SessionHistoryResponse, ClearSessionResponse
from services.session_manager import session_manager

router = APIRouter(prefix="/sessions", tags=["Sessions"])
logger = logging.getLogger(__name__)

DEFAULT_USER_ID = "11111111-1111-1111-1111-111111111111"

@router.get(
    "/{session_id}",
    response_model=SessionHistoryResponse,
    summary="Get conversation history for a session",
    description="""
    Retrieves the entire chronological message history mapped to `session_id`.
    Returns multi-modal emotion data tracked across the history of interactions.
    """,
    responses={
        404: {"description": "Session could not be located in the database for this user."}
    }
)
async def get_history(
    session_id: str, 
    user_id: Optional[str] = Query(None, description="The user's UUID. Uses default if missing."),
    db: AsyncSession = Depends(get_db)
) -> SessionHistoryResponse:
    effective_user_id = user_id or DEFAULT_USER_ID
    try:
        session = await session_manager.get_session(db, session_id, effective_user_id)
        
        # Manually construct response since SQLALchemy requires us to map messages
        # Note: messages are eagerly/lazy loaded. session.messages might fail async unless mapped or queried beforehand.
        messages = await session_manager.get_messages(db, session_id, effective_user_id)
        
        return SessionHistoryResponse(
            id=str(session.id),
            user_id=str(session.user_id),
            title=session.title,
            messages=messages,
            message_count=len(messages),
            created_at=session.created_at,
            updated_at=session.updated_at,
        )
    except Exception as e:
        logger.error(f"Get history error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        ) from e


@router.delete(
    "/{session_id}",
    response_model=ClearSessionResponse,
    summary="Delete a session forever",
    description="""
    Permanently erases the specified `session_id` and securely wipes all corresponding interaction history 
    messages mapped to the user within the Database via cascading row deletion.
    """,
    responses={
        404: {"description": "Session could not be located in the database for this user."}
    }
)
async def clear_history(
    session_id: str, 
    user_id: Optional[str] = Query(None, description="The user's UUID. Uses default if missing."),
    db: AsyncSession = Depends(get_db)
) -> ClearSessionResponse:
    effective_user_id = user_id or DEFAULT_USER_ID
    try:
        await session_manager.clear_session(db, session_id, effective_user_id)
        return ClearSessionResponse(
            status="success",
            message=f"Session {session_id} cleared",
        )
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        ) from e


@router.patch(
    "/{session_id}/title",
    summary="Update session title",
    description="""
    Useful when you want the Chatbot to auto-generate a topic or title based on the recent conversational turn 
    and push the new label down into the database (used heavily on front-ends to populate user memory sidebars).
    """,
    responses={
        200: {"description": "Successfully updated the title."},
        404: {"description": "Session could not be located in the database for this user."}
    }
)
async def update_session_title(
    session_id: str,
    title: str = Query(..., description="The new title"),
    user_id: Optional[str] = Query(None, description="The user's UUID. Uses default if missing."),
    db: AsyncSession = Depends(get_db)
):
    effective_user_id = user_id or DEFAULT_USER_ID
    try:
        session = await session_manager.get_session(db, session_id, effective_user_id)
        session.title = title
        db.add(session)
        await db.commit()
        return {"status": "success", "title": title}
    except Exception as e:
        logger.error(f"Update title error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        ) from e
