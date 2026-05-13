from sqlalchemy.orm import Session

from app.models.submission import CodeSubmission
from app.schemas.submission import SubmissionCreate


def create_submission(
    db: Session,
    user_id: int,
    submission_in: SubmissionCreate,
    extracted_features: dict,
    prediction_output: float,
    model_version: str,
) -> CodeSubmission:
    submission = CodeSubmission(
        user_id=user_id,
        raw_code=submission_in.raw_code,
        extracted_features=extracted_features,
        prediction_output=prediction_output,
        model_version=model_version,
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)
    return submission


def get_submission_by_id(db: Session, submission_id: int, user_id: int) -> CodeSubmission | None:
    return (
        db.query(CodeSubmission)
        .filter(CodeSubmission.id == submission_id, CodeSubmission.user_id == user_id)
        .first()
    )
