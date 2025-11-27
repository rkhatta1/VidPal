# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import logging
from celery import Celery
from config import get_settings
from db.models import EpisodeRepository
from db.connection import db

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# Flask Setup
app = Flask(__name__)
CORS(app)

# Celery Setup
celery_app = Celery(
    "vidpal_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "vidpal-api"}), 200

@app.route('/process', methods=['POST'])
def start_processing():
    """
    Expects JSON:
    {
        "episode_id": "optional-uuid",
        "title": "My Podcast",
        "audio_url": "gs://bucket/audio.mp3",  // OPTIONAL now
        "video_urls": {
            "cam_wide": "gs://bucket/cam1.mp4",
            "cam_host": "gs://bucket/cam2.mp4"
        },
        "sync_options": {
            "enabled": true,
            "master_file_id": "cam_wide"  // Audio extracted from here if audio_url is null
        }
    }
    """
    data = request.json
    
    # video_urls is always required
    if not data or 'video_urls' not in data:
        return jsonify({"error": "Missing video_urls"}), 400
    
    if not data['video_urls']:
        return jsonify({"error": "video_urls cannot be empty"}), 400

    # audio_url is now optional
    audio_url = data.get('audio_url')  # Can be None
    
    episode_id = data.get('episode_id') or str(uuid.uuid4())
    
    try:
        EpisodeRepository.create_episode(
            episode_id=episode_id,
            title=data.get('title', 'Untitled'),
            audio_path=audio_url  # Can be None initially
        )
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return jsonify({"error": "Database initialization failed"}), 500

    task = celery_app.send_task(
        'worker.run_pipeline',
        args=[
            episode_id,
            audio_url,  # Can be None
            data['video_urls'],
            data.get('title'),
            data.get('sync_options'),
        ]
    )
    
    return jsonify({
        "message": "Processing started",
        "episode_id": episode_id,
        "task_id": task.id
    }), 202

@app.route('/status/<episode_id>', methods=['GET'])
def get_status(episode_id):
    # Check DB for status
    episode = EpisodeRepository.get_episode(episode_id)
    if not episode:
        return jsonify({"error": "Episode not found"}), 404
     
    response = {
        "episode_id": episode['episode_id'],
        "title": episode['title'],
        "status": episode['processing_status'],
        "updated_at": episode['updated_at']
    }

    if episode['metadata']:
        response["output"] = episode['metadata']

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
