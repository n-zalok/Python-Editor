from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth.dependencies import get_current_user
from app.schemas.submission import SubmissionCreate, SubmissionCreateResponse, SubmissionResult
from app.services.submission_service import create_submission, get_submission_by_id
from app.ml.pipeline import pipeline
from app.models.user import User


router = APIRouter()


@router.get("/config")
def get_config():
    try:
        from app.ml.pipeline import char_limit

        return {"char_limit": round(int(char_limit), -3)}
    except Exception:
        # Return default if calculation fails
        return {"char_limit": 10000}
    

@router.post("/", response_model=SubmissionCreateResponse)
def submit_code(
    submission_in: SubmissionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    features, score, message, version = pipeline(submission_in.raw_code)
    version_value = str(version.version)
    saved = create_submission(
        db,
        current_user.id,
        submission_in,
        features,
        prediction_output=score,
        model_version=version_value,
    )
    return {
        "id": saved.id,
        "raw_code": saved.raw_code,
        "recommendation_text": message,
    }


@router.get("/{submission_id}", response_model=SubmissionResult)
def get_submission(submission_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    submission = get_submission_by_id(db, submission_id, current_user.id)
    if not submission:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Submission not found")

    _, _, message, _ = pipeline(submission.raw_code)
    return {
        "raw_code": submission.raw_code,
        "recommendation_text": message,
    }
