
-- init-db/01_init.sql

-- Create extension for vector embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- ========== Main Tables ==========

-- Episodes/Projects table
CREATE TABLE episodes (
    id SERIAL PRIMARY KEY,
    episode_id VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(500),
    description TEXT,
    duration_seconds FLOAT,
    video_count INTEGER,
    audio_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_status VARCHAR(50) DEFAULT 'pending',
    metadata JSONB
);

-- Video files table
CREATE TABLE video_files (
    id SERIAL PRIMARY KEY,
    episode_id VARCHAR(255) REFERENCES episodes(episode_id) ON DELETE CASCADE,
    camera_id VARCHAR(100) NOT NULL,
    file_path TEXT NOT NULL,
    file_hash VARCHAR(64),
    duration_seconds FLOAT,
    width INTEGER,
    height INTEGER,
    fps FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    UNIQUE(episode_id, camera_id)
);

-- Speaker/Diarization results
CREATE TABLE speakers (
    id SERIAL PRIMARY KEY,
    episode_id VARCHAR(255) REFERENCES episodes(episode_id) ON DELETE CASCADE,
    speaker_id VARCHAR(100) NOT NULL,
    role VARCHAR(50),  -- host, guest, speaker_3, etc.
    total_duration FLOAT,
    segment_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    UNIQUE(episode_id, speaker_id)
);

-- Speaker segments/turns
CREATE TABLE speaker_segments (
    id SERIAL PRIMARY KEY,
    episode_id VARCHAR(255) REFERENCES episodes(episode_id) ON DELETE CASCADE,
    speaker_id VARCHAR(100) NOT NULL,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    text TEXT,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX idx_speaker_segments_episode ON speaker_segments(episode_id);
CREATE INDEX idx_speaker_segments_time ON speaker_segments(episode_id, start_time, end_time);

-- EDL Cuts
CREATE TABLE edl_cuts (
    id SERIAL PRIMARY KEY,
    episode_id VARCHAR(255) REFERENCES episodes(episode_id) ON DELETE CASCADE,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    camera_id VARCHAR(100) NOT NULL,
    reason TEXT,  -- speaker, reaction, rapid_exchange, opening, etc.
    sequence_order INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX idx_edl_cuts_episode ON edl_cuts(episode_id);
CREATE INDEX idx_edl_cuts_time ON edl_cuts(episode_id, start_time, end_time);

-- Processing cache for expensive operations
CREATE TABLE processing_cache (
    id SERIAL PRIMARY KEY,
    cache_key VARCHAR(64) UNIQUE NOT NULL,
    stage VARCHAR(100) NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    hit_count INTEGER DEFAULT 0
);

CREATE INDEX idx_cache_key_stage ON processing_cache(cache_key, stage);


-- ========== RAG/Vector Store Tables ==========

-- Transcript chunks with embeddings
CREATE TABLE transcript_chunks (
    id SERIAL PRIMARY KEY,
    episode_id VARCHAR(255) REFERENCES episodes(episode_id) ON DELETE CASCADE,
    chunk_id INTEGER NOT NULL,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    speaker VARCHAR(100),
    text TEXT NOT NULL,
    embedding vector(768),  -- Default: 768 dimensions (configurable to 3072)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    UNIQUE(episode_id, chunk_id)
);

CREATE INDEX idx_transcript_chunks_episode ON transcript_chunks(episode_id);
CREATE INDEX idx_transcript_chunks_embedding ON transcript_chunks 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ========== VLM Descriptions Table ==========
CREATE TABLE vlm_descriptions (
    id SERIAL PRIMARY KEY,
    episode_id VARCHAR(255) REFERENCES episodes(episode_id) ON DELETE CASCADE,
    camera_id VARCHAR(100) NOT NULL,
    time_seconds FLOAT NOT NULL,
    transition_time FLOAT,
    offset_seconds FLOAT,
    description TEXT NOT NULL,
    embedding vector(768),  -- Gemini embedding of description
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX idx_vlm_descriptions_episode ON vlm_descriptions(episode_id);
CREATE INDEX idx_vlm_descriptions_time ON vlm_descriptions(episode_id, time_seconds);
CREATE INDEX idx_vlm_descriptions_embedding ON vlm_descriptions 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ========== Utility Functions ==========

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_episodes_updated_at BEFORE UPDATE ON episodes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


-- ========== Indexes for Performance ==========
CREATE INDEX idx_episodes_status ON episodes(processing_status);
CREATE INDEX idx_episodes_created ON episodes(created_at DESC);
CREATE INDEX idx_video_files_episode ON video_files(episode_id);
CREATE INDEX idx_speakers_episode ON speakers(episode_id);
