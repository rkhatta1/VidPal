# rag/pgvector_store.py (COMPLETE FIXED VERSION)
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from google import genai
from google.genai.types import EmbedContentConfig
from db.connection import db
from config import get_settings

logger = logging.getLogger(__name__)


class PGVectorRAGStore:
    """RAG store using PostgreSQL with pgvector and Gemini embeddings."""
    
    def __init__(self):
        settings = get_settings()
        
        # Initialize Gemini client
        if settings.GOOGLE_GENAI_USE_VERTEXAI:
            logger.info("Initializing Gemini client with Vertex AI (ADC)")
            self.client = genai.Client(
                vertexai=True,
                project=settings.GOOGLE_CLOUD_PROJECT,
                location=settings.GOOGLE_CLOUD_LOCATION,
            )
        else:
            logger.info("Initializing Gemini client with API key")
            self.client = genai.Client(
                api_key=settings.GOOGLE_API_KEY,
            )
        
        # Use gemini-embedding-001 (768 dimensions by default)
        self.embedding_model = settings.EMBEDDING_MODEL
        self.embedding_dim = settings.EMBEDDING_DIM
        
        logger.info(
            f"✅ RAG store initialized "
            f"(model: {self.embedding_model}, dim: {self.embedding_dim})"
        )
    
    def _generate_embeddings(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> List[List[float]]:
        """
        Generate embeddings using Gemini.
        
        Args:
            texts: List of texts to embed
            task_type: RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY
        
        Returns:
            List of embedding vectors as Python lists of plain floats
        """
        # Gemini embedding API has a limit of 250 texts per request
        batch_size = 250
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=batch_texts,
                    config=EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.embedding_dim,
                    ),
                )
                
                # Extract embeddings and convert to plain Python lists of floats
                for emb in response.embeddings:
                    # Get the values - they might be various types
                    values = emb.values
                    
                    # Convert to plain Python list of plain Python floats
                    if isinstance(values, np.ndarray):
                        # NumPy array - convert to list
                        embedding_list = [float(v) for v in values.flatten()]
                    elif isinstance(values, (list, tuple)):
                        # Already a list/tuple - ensure all elements are plain floats
                        embedding_list = [float(v) for v in values]
                    elif hasattr(values, 'tolist'):
                        # Has tolist method (like numpy)
                        temp_list = values.tolist()
                        embedding_list = [float(v) for v in temp_list]
                    else:
                        # Last resort: iterate and convert
                        embedding_list = [float(v) for v in values]
                    
                    # Verify dimension
                    if len(embedding_list) != self.embedding_dim:
                        logger.warning(
                            f"Embedding dimension mismatch: "
                            f"expected {self.embedding_dim}, got {len(embedding_list)}"
                        )
                    
                    all_embeddings.append(embedding_list)
                
            except Exception as e:
                logger.error(f"Embedding generation failed for batch {i}: {e}")
                # Return zero vectors as fallback
                all_embeddings.extend(
                    [[0.0] * self.embedding_dim] * len(batch_texts)
                )
        
        return all_embeddings
    
    # rag/pgvector_store.py (corrected ingest_transcript_chunks method)

    def ingest_transcript_chunks(
        self,
        episode_id: str,
        transcript: List[Dict[str, Any]],
        chunk_duration: float = 30.0,
    ) -> None:
        """
        Ingest transcript chunks with embeddings into the database.
        
        Args:
            episode_id: Unique identifier for this episode
            transcript: List of word-level transcript data
            chunk_duration: Duration of each chunk in seconds
        """
        if not transcript:
            logger.warning("Empty transcript provided")
            return
        
        logger.info(f"Ingesting transcript chunks for episode {episode_id}")
        
        # Build chunks
        chunks = self._build_chunks(transcript, chunk_duration)
        
        if not chunks:
            logger.warning("No chunks generated")
            return
        
        # Generate embeddings for all chunks using Gemini
        texts = [chunk['text'] for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self._generate_embeddings(texts, task_type="RETRIEVAL_DOCUMENT")
        
        # Verify embeddings
        if len(embeddings) != len(chunks):
            logger.error(f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)}")
            return
        
        # Insert into database
        with db.get_cursor() as cursor:
            # Delete existing chunks for this episode
            cursor.execute(
                "DELETE FROM transcript_chunks WHERE episode_id = %s",
                (episode_id,)
            )
            
            # Insert new chunks
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Ensure embedding is a list of plain Python floats
                if not isinstance(embedding, list):
                    embedding = list(embedding)
                
                # Convert all values to plain Python floats (strip numpy types)
                embedding = [float(v) for v in embedding]
                
                # Convert to pgvector format: [val1,val2,val3,...]
                # IMPORTANT: Use square brackets, not curly braces!
                embedding_str = '[' + ','.join(str(v) for v in embedding) + ']'
                
                try:
                    cursor.execute(
                        """
                        INSERT INTO transcript_chunks 
                        (episode_id, chunk_id, start_time, end_time, speaker, text, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s)
                        """,
                        (
                            episode_id,
                            i,
                            chunk['start_time'],
                            chunk['end_time'],
                            chunk.get('speaker', 'unknown'),
                            chunk['text'],
                            embedding_str,  # Use square bracket format
                            None,
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to insert chunk {i}: {e}")
                    logger.error(f"Embedding sample: {embedding[:5]}")
                    raise
        
        logger.info(f"✅ Ingested {len(chunks)} chunks with Gemini embeddings")

    
    def _build_chunks(
        self,
        transcript: List[Dict[str, Any]],
        chunk_duration: float,
    ) -> List[Dict[str, Any]]:
        """Build text chunks from transcript."""
        if not transcript:
            return []
        
        chunks = []
        current_chunk = {
            'text': [],
            'start_time': transcript[0]['start'],
            'speaker': transcript[0].get('speaker', 'unknown'),
        }
        
        for word_data in transcript:
            # Check if we should start a new chunk
            if word_data['start'] - current_chunk['start_time'] >= chunk_duration:
                # Finalize current chunk
                current_chunk['text'] = ' '.join(current_chunk['text'])
                current_chunk['end_time'] = word_data['start']
                chunks.append(current_chunk)
                
                # Start new chunk
                current_chunk = {
                    'text': [],
                    'start_time': word_data['start'],
                    'speaker': word_data.get('speaker', 'unknown'),
                }
            
            current_chunk['text'].append(word_data['word'])
        
        # Finalize last chunk
        if current_chunk['text']:
            current_chunk['text'] = ' '.join(current_chunk['text'])
            current_chunk['end_time'] = transcript[-1]['end']
            chunks.append(current_chunk)
        
        return chunks
    
    def ingest_vlm_descriptions(
        self,
        episode_id: str,
        vlm_descriptions: List[Dict[str, Any]],
    ) -> None:
        """
        Ingest VLM descriptions with embeddings into the database.
        
        Args:
            episode_id: Unique identifier for this episode
            vlm_descriptions: List of VLM description dictionaries
        """
        if not vlm_descriptions:
            logger.warning("Empty VLM descriptions provided")
            return
        
        logger.info(f"Ingesting {len(vlm_descriptions)} VLM descriptions for episode {episode_id}")
        
        # Generate embeddings for all descriptions
        texts = [desc['description'] for desc in vlm_descriptions]
        logger.info(f"Generating embeddings for {len(texts)} VLM descriptions...")
        embeddings = self._generate_embeddings(texts, task_type="RETRIEVAL_DOCUMENT")
        
        # Insert into database
        with db.get_cursor() as cursor:
            # Delete existing VLM descriptions for this episode
            cursor.execute(
                "DELETE FROM vlm_descriptions WHERE episode_id = %s",
                (episode_id,)
            )
            
            # Insert new descriptions
            for desc, embedding in zip(vlm_descriptions, embeddings):
                # Ensure embedding is a list of plain Python floats
                embedding = [float(v) for v in embedding]
                
                # Convert to pgvector format
                embedding_str = '[' + ','.join(str(v) for v in embedding) + ']'
                
                try:
                    cursor.execute(
                        """
                        INSERT INTO vlm_descriptions 
                        (episode_id, camera_id, time_seconds, transition_time, 
                         offset_seconds, description, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                        """,
                        (
                            episode_id,
                            desc['camera_id'],
                            desc['time'],
                            desc['transition_time'],
                            desc['offset'],
                            desc['description'],
                            embedding_str,
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to insert VLM description: {e}")
                    raise
        
        logger.info(f"✅ Ingested {len(vlm_descriptions)} VLM descriptions")
    
    def retrieve_vlm_context(
        self,
        episode_id: str,
        start_time: float,
        end_time: float,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve VLM descriptions within a time range.
        
        Args:
            episode_id: Episode identifier
            start_time: Start time in seconds
            end_time: End time in seconds
            top_k: Maximum number of results
        
        Returns:
            List of VLM descriptions with metadata
        """
        sql = """
            SELECT 
                camera_id,
                time_seconds,
                transition_time,
                offset_seconds,
                description
            FROM vlm_descriptions
            WHERE episode_id = %s
              AND time_seconds >= %s
              AND time_seconds <= %s
            ORDER BY time_seconds
            LIMIT %s
        """
        
        with db.get_cursor() as cursor:
            cursor.execute(sql, (episode_id, start_time, end_time, top_k))
            results = cursor.fetchall()
        
        return [dict(row) for row in results]
    
    def retrieve(
    self,
    query: str,
    top_k: int = 3,
    episode_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context chunks using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            episode_id: Optional filter by episode
        
        Returns:
            List of relevant chunks with metadata
        """
        # Generate query embedding using Gemini
        logger.debug(f"Generating query embedding for: {query[:50]}...")
        query_embeddings = self._generate_embeddings([query], task_type="RETRIEVAL_QUERY")
        
        if not query_embeddings:
            logger.error("Failed to generate query embedding")
            return []
        
        query_embedding = query_embeddings[0]
        
        # Ensure plain floats
        query_embedding = [float(v) for v in query_embedding]
        
        # Convert to pgvector format: [val1,val2,val3,...]
        embedding_str = '[' + ','.join(str(v) for v in query_embedding) + ']'
        
        # Build SQL query with cosine similarity
        sql = """
            SELECT 
                chunk_id,
                start_time,
                end_time,
                speaker,
                text,
                1 - (embedding <=> %s::vector) AS similarity
            FROM transcript_chunks
        """
        
        params = [embedding_str]
        
        if episode_id:
            sql += " WHERE episode_id = %s"
            params.append(episode_id)
        
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([embedding_str, top_k])
        
        # Execute query
        with db.get_cursor() as cursor:
            cursor.execute(sql, params)
            results = cursor.fetchall()
        
        # Format results
        chunks = []
        for row in results:
            chunks.append({
                'chunk_id': row['chunk_id'],
                'start_time': row['start_time'],
                'end_time': row['end_time'],
                'speaker': row['speaker'],
                'text': row['text'],
                'similarity': float(row['similarity']),
            })
        
        return chunks

    
    def clear_episode(self, episode_id: str) -> None:
        """Clear all chunks for a specific episode."""
        with db.get_cursor() as cursor:
            cursor.execute(
                "DELETE FROM transcript_chunks WHERE episode_id = %s",
                (episode_id,)
            )
            rows_deleted = cursor.rowcount
        
        logger.info(f"Cleared {rows_deleted} chunks for episode {episode_id}")
