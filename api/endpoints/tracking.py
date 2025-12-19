
import uuid
import os
from fastapi import APIRouter, File, UploadFile, HTTPException, Response
from fastapi.responses import FileResponse
from worker.tasks import process_video_task
from worker.celery_app import celery_app
from celery.result import AsyncResult
from fastapi import Depends
from sqlalchemy.orm import Session
from db.session import get_db
from db.models import VideoJob, JobStatus

router = APIRouter()
UPLOAD_DIRETORY = "data/uploads"
RESULTS_DIRECTORY = "data/results"

@router.post("/upload-video")
async def upload_video(video: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Handles the video upload and starts the background processing task.
    """
    file_extension = video.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIRETORY, unique_filename)

    try:
        with open(file_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    
    new_job = VideoJob(
        original_filename = video.filename,
        input_filepath = file_path,
        status = JobStatus.PENDING
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job) # Get the newly created ID and defaults

     # .delay() sends a message to Redis with the task name and its arguments.
    task = process_video_task.delay(str(new_job.id))
    
    # task.id is the "ticket number" that user can use to check the status later
    return {
        "message":  f"Successfully uploaded {video.filename}",
        "task_id" : new_job.id
    }

@router.get("/status/{job_id}")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """
    Checks the status of a job from the database.
    """
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format.")

    job = db.query(VideoJob).filter(VideoJob.id == job_uuid).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    response = {
        "job_id": job.id,
        "status": job.status.value,
        "created_at": job.created_at,
    }
    
    if job.status == JobStatus.COMPLETED:
        response["result_filename"] = os.path.basename(str(job.output_filepath))
    elif job.status == JobStatus.FAILED:
        response["error"] = job.error_message

    return response

@router.get("/download/{filename}")
async def download_video(filename: str):
    """
    Provides the processed video file for download.
    """
    file_path = os.path.join(RESULTS_DIRECTORY, filename)

    if os.path.exists(file_path):

        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found")