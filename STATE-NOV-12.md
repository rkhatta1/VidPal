### Project Summary

The project is **VidPal**, a multi-cam video editing copilot. Its primary goal is to take multiple video files (e.g., `cam_host`, `cam_guest`, `cam_wide`) and a master audio file, and automatically generate a "first draft" edit (as an `.fcpxml` file) based on which speaker is talking.

### Core Architecture & Current State

The pipeline is a series of automated steps defined in `pipeline.py`:

1.  **Phase 1: Audio Processing (`speaker_identification.py`)**
    * Uses Google Cloud Speech-to-Text for speaker diarization (identifying *who* spoke when, e.g., `SPEAKER_00`, `SPEAKER_01`).
    * It *also* extracts the full, word-level transcript directly from this same Google API response.

2.  **Phase 1.5: Speaker-Camera Mapping (`speaker_camera_mapping.py`)**
    * This is a new, advanced VLM step. It takes the abstract speaker IDs (e.g., `SPEAKER_00`) and finds their corresponding camera.
    * It extracts 3-second video snippets from all cameras during a speaker's segment, uploads them to GCS, and asks the Gemini VLM which camera is focused on the person who is *visibly speaking*.
    * The output is a simple, crucial map: `{'host': 'cam_a', 'guest': 'cam_b'}`.

3.  **Phase 2: RAG Ingestion (`rag/pgvector_store.py`)**
    * The full transcript is chunked and ingested into a PGVector database to create a semantic search index.

4.  **Phase 3: Rule-Based EDL (`edl/rules.py`)**
    * Generates the "first pass" edit. Using the `role_camera_map` from Phase 1.5, it creates a basic Edit Decision List (EDL) by cutting to the correct camera based on the diarization data.

5.  **Phase 4: VLM Context (`vlm_processor.py`)**
    * To help the AI *refine* the edit, this step analyzes the rule-based cuts.
    * It uses FastVLM to capture frames from all cameras at the *exact moment of each transition* (e.g., -2s, -1s, 0s, +1s, +2s). This VLM context is used to judge if the cut was good.

6.  **Phase 5: LLM Refinement (`llm_refiner.py`)**
    * This is the "AI Editor." It reviews the rule-based EDL in small chunks (5 cuts at a time).
    * It uses the VLM context from Phase 4 and the transcript to decide if the rule-based cuts are good. It's very effective at merging rapid, unnecessary cuts into longer, more natural shots (e.g., reducing 76 cuts to 35).

7.  **Phase 6: FCPXML Generation (`fcpxml/generator.py`)**
    * The final, refined EDL is formatted and saved as an `.fcpxml` file, which can be opened in video editing software.

---

### Problems Faced & Changes Implemented

We've made several significant changes to improve the pipeline's robustness and accuracy:

1.  **Removed `whisperx`/`pyannote`:**
    * **Problem:** The original pipeline had `whisperx` and `pyannote.audio` for transcription, which caused CUDA/PyTorch dependency crashes.
    * **Solution:** We removed the `processing/audio_transcription.py` file entirely.
    * **Change:** We modified `processing/speaker_identification.py` to extract the full transcript from the Google Speech-to-Text response payload, as it was already providing it alongside the diarization data.

2.  **Fixed "Garbage-In, Garbage-Out" Diarization:**
    * **Problem:** The rule-based EDL was only as good as the diarization. The system had no way of knowing if `SPEAKER_00` was the "host" on `cam_a` or the "guest" on `cam_b`.
    * **Solution:** We built the new `SpeakerCameraMapper` service (Phase 1.5).
    * **Change:** This service uses Gemini VLM on video snippets to create a reliable, automated map between speaker roles and camera IDs.
    * **Change:** We also pinned the speaker count to `3` in `config.py` to help the diarization API produce more accurate segments.

3.  **Improved LLM Refiner Effectiveness:**
    * **Problem:** We briefly tested a change where the VLM (Phase 4) only sampled *within* a cut. This proved ineffective, as the LLM Refiner lost the ability to judge the *transition itself* and stopped merging cuts.
    * **Solution:** We reverted the VLM logic in `processing/vlm_processor.py` back to its original, more effective state.
    * **Change:** The VLM processor now analyzes frames *around* each transition, giving the LLM refiner the context it needs to make smart decisions.

4.  **Sped Up Expensive Operations:**
    * **Problem:** Phase 1.5 (Speaker Mapping) and Phase 4 (VLM Context) are computationally expensive and slow.
    * **Solution:** We implemented file-based caching.
    * **Change:** The pipeline now uses a single, content-based hash key (from the audio file) to cache the results of Phase 1, Phase 1.5, and Phase 4. This makes re-running the pipeline (even with new episode IDs) almost instantaneous.

5.  **API/Git Housekeeping:**
    * We fixed numerous `AttributeError`s and `TypeError`s by aligning the `google-genai` calls with the correct API for **Vertex AI** (e.g., uploading snippets to GCS via `Part.from_uri` instead of using the Developer-only `client.files.upload` method).
    * We resolved a `pydantic` `ValidationError` by cleaning old variables out of the `.env` file.
    * We correctly managed the Git state, moving from a detached HEAD to a feature branch and rebasing onto `main`.
