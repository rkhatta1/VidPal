## **Project Technical Specification: VidPal AI Editing Copilot**

### 1. Project Objective

VidPal is an AI-driven multi-camera video editing copilot. Its primary function is to ingest multiple video files (e.g., `cam_a`, `cam_b`, `cam_wide`) and a master audio file, process them, and automatically generate a "first draft" edit decision list (EDL) as an FCPXML file. This FCPXML file is based on speaker diarization, speaker-to-camera mapping, and a set of dynamic editing rules.

### 2. Core Technologies

* **Orchestration:** Python
* **Configuration:** Pydantic (via `config.py`)
* **Database:** PostgreSQL with `pgvector` extension for structured data and vector storage
* **Audio/Video Processing:** `ffmpeg`
* **Speech & Diarization:** Google Cloud Speech-to-Text (Long Running Recognize)
* **VLM & LLM:** Google Gemini via Vertex AI (for VLM mapping and optional refinement)
* **File Storage:** Google Cloud Storage (for temporary VLM snippets)
* **RAG:** `pgvector` for vector search
* **File Caching:** A simple file-based JSON cache is used for expensive operations (diarization, VLM mapping) to speed up re-runs.

---

### 3. System Architecture & Data Flow

The project is orchestrated by `pipeline.py` and follows a sequential, multi-phase process.

#### **Phase 1: Audio Processing & Diarization**
* **Component:** `processing.speaker_identification.SpeakerIdentifier`
* **Action:**
    1.  The master audio file is converted to a 16kHz mono WAV file using `ffmpeg`.
    2.  This WAV file is uploaded to Google Cloud Storage.
    3.  A `long_running_recognize` request is sent to the Google Cloud Speech-to-Text API, configured with speaker diarization (e.g., `min_speaker_count`, `max_speaker_count`).
    4.  The operation result is polled with a 1-hour timeout (3600s).
    5.  Two data streams are extracted from the successful response:
        * **Full Word-level Transcript:** A list of `{"word", "start", "end", "speaker"}`.
        * **Speaker Segments:** A list of continuous speech blocks, grouped by the raw speaker tag (e.g., `SPEAKER_00`, `SPEAKER_01`).
    6.  The `_create_role_mapping` function processes the speaker segments, sorts speakers by total talk time, and assigns abstract numerical roles (e.g., `speaker_00`, `speaker_01`, etc.).
* **Output:** `speaker_segments`, `role_mapping` (e.g., `{'SPEAKER_01': 'speaker_00'}`), `transcript`.

#### **Phase 1.5: Speaker-to-Camera Mapping (VLM)**
* **Component:** `processing.speaker_camera_mapping.SpeakerCameraMapper`
* **Action:**
    1.  For each abstract role (`speaker_00`, `speaker_01`), the system finds their longest speech segment from Phase 1.
    2.  Using `ffmpeg`, it extracts 3-second video snippets from that segment's midpoint *from all available camera feeds* (`cam_a`, `cam_b`, `cam_wide`).
    3.  These snippets are uploaded to GCS.
    4.  A prompt is sent to the Gemini VLM, providing all video snippets and asking: "The audio track confirms that speaker 'SPEAKER_01' is talking... Which camera... is clearly focused on the person who is speaking?".
    5.  The VLM's text response (e.g., "cam_a") is parsed.
* **Output:** `role_camera_map` (e.g., `{'speaker_00': 'cam_b', 'speaker_01': 'cam_a', 'default_wide': 'cam_wide'}`).

#### **Phase 2: RAG Ingestion (Optional)**
* **Component:** `rag.pgvector_store.PGVectorRAGStore`
* **Action:**
    1.  The full word-level transcript is chunked into N-second blocks (e.g., 30s).
    2.  Gemini embedding models are used to generate embeddings for each text chunk.
    3.  The text chunk, timestamps, and embedding are stored in the `transcript_chunks` table for later semantic search.
* **Purpose:** Provides contextual data for the LLM Refiner (Phase 5).

#### **Phase 3: EDL Generation (Rule-Based)**
* **Component:** `edl.rules.RuleBasedEDLGenerator`
* **Action:**
    1.  This is the primary editing logic. It iterates through the `speaker_segments` from Phase 1.
    2.  **Rule 1:** Start the timeline with an N-second wide shot (e.g., `3.0s`).
    3.  **Rule 2:** For each speaker segment, *always* cut to the `default_wide` camera for a set duration (e.g., `5.0s`).
    4.  **Rule 3:** If the speaker's segment is *longer* than this intro duration, and they have a dedicated close-up (from Phase 1.5), cut to their close-up camera for the remainder of their segment.
    5.  **Rule 4:** If the segment is short, or the speaker's mapped camera *is* the wide cam, stay on the wide cam.
    6.  **Cleanup:**
        * `_merge_consecutive`: Merges identical, back-to-back camera cuts.
        * `_eliminate_gaps`: Scans the timeline and programmatically inserts new `cam_wide` clips to fill any silent gaps between cuts.
* **Output:** A list of "base EDL" `cuts` (a list of dicts).

#### **Phase 3.5: Base FCPXML Generation (Conditional)**
* **Component:** `pipeline.py` / `fcpxml.generator.FCPXMLGenerator`
* **Action:** If `REFINE_WITH_LLM` is `True`, the pipeline saves the *current* state of the EDL as `_base_rules.fcpxml` before proceeding.

#### **Phase 4: VLM Context Generation (Optional)**
* **Component:** `processing.vlm_processor.VLMProcessor`
* **Action:**
    1.  Analyzes the `cuts` from Phase 3 to find all timestamps where the `camera_id` *changes*.
    2.  For each transition, it extracts frames from *all* video files at offsets (e.g., -2s, -1s, 0s, +1s, +2s).
    3.  It uses a local VLM (FastVLM) to describe each frame ("Describe this video frame... Focus on: people visible, their actions, facial expressions...").
    4.  These descriptions are stored in the `vlm_descriptions` table.
* **Purpose:** Provides rich visual context for the LLM Refiner (Phase 5).

#### **Phase 5: LLM Refinement (Conditional)**
* **Component:** `processing.llm_refiner.LLMRefiner`
* **Action:**
    1.  If `REFINE_WITH_LLM` is `True`, this phase runs.
    2.  It processes the "base EDL" in chunks (e.g., 10 cuts at a time).
    3.  For each chunk, it builds a comprehensive, generic prompt for Gemini.
    4.  **Prompt Includes:**
        * The chunk of transcript (using abstract `speaker_00`, `speaker_01` labels).
        * The VLM descriptions from Phase 4 ("Visual Context").
        * The RAG context from Phase 2 ("Similar Moments").
        * The current "base EDL" cuts.
        * A set of generic rules (e.g., "Use Wide Shots... during rapid back-and-forth," "Use Close-up Shots... when a speaker is talking for 8+ seconds").
    5.  It requests a JSON response, which replaces the "base EDL" chunk with the new, refined cuts.
* **Output:** A *new* list of `cuts` (the "refined EDL").

#### **Phase 6: Final FCPXML Generation**
* **Component:** `fcpxml.generator.FCPXMLGenerator`
* **Action:**
    1.  Takes the final list of `cuts` (which is either the base EDL or the refined EDL).
    2.  Generates an `fcpxml` XML structure.
    3.  It creates `<asset>` tags for all video/audio sources.
    4.  It creates a `<spine>` and populates it with `<audio>` (for the master track) and `<clip>` elements corresponding to each cut in the final EDL.
    5.  The final file is saved to the `output/` directory.

---

### 4. Database Schema

The PostgreSQL database (DB: `vidpalai`) serves as the "single source of truth" for a project's state.

* **`episodes`**: Main table, tracks the episode ID, title, and processing status (`pending`, `processing`, `completed`, `failed`).
* **`video_files`**: Stores metadata for each video source, linking `camera_id` to a file path.
* **`speakers`**: Stores the abstract roles (e.g., `speaker_00`) and their `speaker_id` (e.g., `SPEAKER_01`), along with stats.
* **`speaker_segments`**: The raw output of diarization. Stores start/end times for each speech block.
* **`edl_cuts`**: Stores the final EDL (base or refined) for the episode.
* **`transcript_chunks`**: RAG table. Stores text chunks and their 768-dim `vector` embeddings.
* **`vlm_descriptions`**: RAG table. Stores VLM-generated text descriptions and their `vector` embeddings.

---

### 5. Future Architecture: Emotion Detection (MediaPipe Service)

This is a proposed new module to enhance editing logic by detecting non-verbal reactions.

* **Objective:** Identify key emotional moments (laughter, surprise, etc.) and use them as editing triggers, moving beyond simple speaker-following.

* **Proposed Data Flow:**
    1.  **Phase 1 (Crawl - New Component): `EmotionDetector`**
        * **Tool:** MediaPipe Face Landmarker.
        * **Action:** This service will run *locally* on all video feeds (`cam_a`, `cam_b`, `cam_wide`) *in parallel* with Phase 1 Audio Processing.
        * It will scan for facial blendshape combinations that signify key emotions (e.g., `mouthSmile` + `jawOpen` = Laughter).
        * **Output:** A list of `ReactionEvents` (e.g., `{"timestamp": 301.5, "event": "collective_laughter", "camera": "cam_wide"}`).

    2.  **Phase 2 (Walk - Modify `edl/rules.py`): Programmatic Reaction**
        * The `RuleBasedEDLGenerator` will receive this new list of `ReactionEvents`.
        * A new rule will be added: "If a `collective_laughter` event occurs, override all other speaker logic and insert a 4-second cut to `cam_wide`."
        * This provides an immediate, reliable (though simple) improvement.

    3.  **Phase 3 (Run - Modify `processing/vlm_processor.py`): VLM Context**
        * The `VLMProcessor` will be modified to *also* process timestamps from the `ReactionEvents` list.
        * This will provide VLM descriptions (the "why") for each reaction (e.g., "speaker_01 is laughing in reaction to speaker_00's story").

    4.  **Phase 4 (Fly - Enable `processing/llm_refiner.py`): LLM Refinement**
        * Re-enable the LLM Refiner.
        * The prompt (from `_build_refinement_prompt`) will be updated to include a new section:
            > **"Reaction Events:**
            > * `t=301.5s`: VLM reports 'speaker\_01 and speaker\_02 are laughing hard.'
            > * `t=405.2s`: VLM reports 'speaker\_01 has a look of surprise.'
            >
            > Refine the cuts to incorporate these reactions naturally."
        * The LLM can then make human-like decisions (e.g., "Cut to wide" vs. "Hold on speaker" vs. "Insert PiP").

    5.  **Phase 5 (Future - Modify `fcpxml/generator.py`): Advanced Edits**
        * The FCPXML generator will be enhanced to accept advanced commands from the LLM.
        * This includes adding XML tags for multi-track editing (Picture-in-Picture) or `transform` keyframes (slow zooms/pans) to highlight these reactions dynamically.
