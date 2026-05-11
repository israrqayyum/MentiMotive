from fastapi import APIRouter, status
from models.schemas import HealthResponse, RootResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/",
    response_model=RootResponse,
    summary="Service banner",
    description="Returns a short welcome string and the API `version` string.",
)
async def root():
    return {"message": "Welcome to MentiMotive API 👋", "version": "2.0"}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness",
    status_code=status.HTTP_200_OK,
    description="Use for load balancers and uptime checks. Does not validate ML subsystems; only the HTTP process must be up.",
)
async def health_check():
    return {"status": "ok", "message": "API is healthy and running!"}
