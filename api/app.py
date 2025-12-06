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


@app.route('/prepare-mapping', methods=['POST'])
def prepare_mapping():
    """
    Manual mapping Phase A.

    Payload is identical to /process, but instead of running the full
    pipeline, it stops after diarization + snippet generation and
    returns when the episode is ready for human mapping.

    {
        "episode_id": "optional-uuid",
        "title": "My Podcast",
        "audio_url": "gs://bucket/audio.mp3",  // OPTIONAL
        "video_urls": { "cam_wide": "...", "cam_host": "..." },
        "sync_options": { "enabled": true, "master_file_id": "cam_wide" }
    }
    """
    data = request.json or {}

    if 'video_urls' not in data:
        return jsonify({"error": "Missing video_urls"}), 400
    if not data['video_urls']:
        return jsonify({"error": "video_urls cannot be empty"}), 400

    audio_url = data.get('audio_url')
    episode_id = data.get('episode_id') or str(uuid.uuid4())

    try:
        EpisodeRepository.create_episode(
            episode_id=episode_id,
            title=data.get('title', 'Untitled'),
            audio_path=audio_url,
            metadata={
                "mapping_mode": "manual",
                "mapping_status": "preparing",
            }
        )
    except Exception as e:
        logger.error(f"DB Error in prepare-mapping: {e}")
        return jsonify({"error": "Database initialization failed"}), 500

    task = celery_app.send_task(
        'worker.prepare_for_mapping',
        args=[
            episode_id,
            audio_url,
            data['video_urls'],
            data.get('title'),
            data.get('sync_options'),
        ]
    )

    return jsonify({
        "message": "Manual mapping preparation started",
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

@app.route('/mapping/<episode_id>', methods=['POST'])
def submit_mapping(episode_id):
    """
    Submit human-provided speaker-role → camera mapping and start Phase B.

    Expected payload:
    {
      "role_camera_map": {
        "speaker_00": "cam_host",
        "speaker_01": "cam_guest"
      },
      "default_wide": "cam_wide"   // optional; will be added to role_camera_map
    }
    """
    episode = EpisodeRepository.get_episode(episode_id)
    if not episode:
        return jsonify({"error": "Episode not found"}), 404

    data = request.json or {}
    role_camera_map = data.get("role_camera_map") or {}
    if not isinstance(role_camera_map, dict) or not role_camera_map:
        return jsonify({"error": "role_camera_map must be a non-empty object"}), 400

    default_wide = data.get("default_wide")
    if default_wide:
        role_camera_map.setdefault("default_wide", default_wide)

    # Persist mapping + update metadata/status
    EpisodeRepository.update_episode_metadata(
        episode_id,
        {
            "manual_role_camera_map": role_camera_map,
            "mapping_status": "mapping_received",
        }
    )

    task = celery_app.send_task(
        'worker.finish_with_mapping',
        args=[episode_id, role_camera_map]
    )

    return jsonify({
        "message": "Manual mapping received, final processing started",
        "episode_id": episode_id,
        "task_id": task.id
    }), 202

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
