# worker.py
from celery import Celery
from config import get_settings
from pipeline import VidPalAIPipeline
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

celery_app = Celery(
    "vidpal_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

@celery_app.task(name='worker.run_pipeline', bind=True)
def run_pipeline(self, episode_id, audio_url, video_urls, title):
    logger.info(f"Worker received task for episode {episode_id}")
    
    try:
        pipeline = VidPalAIPipeline()
        result = pipeline.process_episode(
            episode_id=episode_id,
            audio_gcs_uri=audio_url,
            video_gcs_uris=video_urls,
            title=title
        )
        return result
    except Exception as e:
        logger.error(f"Task failed: {e}")
        # Re-raise to let Celery mark as failed
        raise e
