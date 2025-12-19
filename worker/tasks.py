from .celery_app import celery_app
from core.processing import ParkVisionProcessor
import os
from db.session import SessionLocal
from db.models import VideoJob, JobStatus
import uuid

MODELS_DIR = "models"
DATA_DIR = "data"
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok = True)

class ProcessorSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("Initializing ParkVisionProcessor for the first time...")
            cls._instance = ParkVisionProcessor(
                occupied_model_path=os.path.join(MODELS_DIR, "occupied_model.pt"),
                spot_model_path=os.path.join(MODELS_DIR, "parking_spot_model.pt"),
                car_model_path=os.path.join(MODELS_DIR, "yolov8n.pt")
            )
        return cls._instance


@celery_app.task
def process_video_task(job_id: str):
    """
    The background task that now operates on a database job ID.
    """
    db = SessionLocal()
    
    job = None 
    try:

        job = db.query(VideoJob).filter(VideoJob.id == uuid.UUID(job_id)).first()

        if not job:
            print(f"Error: Job with ID {job_id} not found in the database.")
            return 

        job.status = JobStatus.PROCESSING
        db.commit()

        processor = ProcessorSingleton.get_instance()
        
        output_filename = f"{job.id}_processed.mp4"
        output_path = os.path.join(RESULTS_DIR, output_filename)

        result_path = processor.process_video(job.input_filepath, output_path)

        job.status = JobStatus.COMPLETED
        job.output_filepath = result_path
        db.commit()
        print(f"Job {job_id} completed successfully.")

    except Exception as e:
        print(f"Job {job_id} failed. Error: {e}")
        if job:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            db.commit()
        raise e
    finally:
        db.close()