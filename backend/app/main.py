from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from app.routes import auth as auth_router
from app.routes import submissions as submissions_router
from app.utils.logging import configure_logging
from app.config import settings
from app.database import engine, Base

configure_logging()

app = FastAPI(
    title="Python Editor API",
    description="API for ML-powered Python code review and recommendations",
    version="0.1.0",
)

Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {"message": "hello"}

app.include_router(auth_router.router, prefix="/api/auth", tags=["auth"])
app.include_router(submissions_router.router, prefix="/api/submissions", tags=["submissions"])

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
