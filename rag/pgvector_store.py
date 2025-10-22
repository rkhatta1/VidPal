# rag/pgvector_store.py
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
        
        # Use gemini-embedding-001 (3072 dimensions by default)
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
            List of embedding vectors
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
                
                # Extract embeddings
                batch_embeddings = [
                    emb.values for emb in response.embeddings
                ]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Embedding generation failed for batch {i}: {e}")
                # Return zero vectors as fallback
                all_embeddings.extend(
                    [[0.0] * self.embedding_dim] * len(batch_texts)
                )
        
        return all_embeddings
    
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
        
        # Insert into database
        with db.get_cursor() as cursor:
            # Delete existing chunks for this episode
            cursor.execute(
                "DELETE FROM transcript_chunks WHERE episode_id = %s",
                (episode_id,)
            )
            
            # Insert new chunks
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                cursor.execute(
                    """
                    INSERT INTO transcript_chunks 
                    (episode_id, chunk_id, start_time, end_time, speaker, text, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        episode_id,
                        i,
                        chunk['start_time'],
                        chunk['end_time'],
                        chunk.get('speaker', 'unknown'),
                        chunk['text'],
                        embedding,
                        None,
                    )
                )
        
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
        
        params = [query_embedding]
        
        if episode_id:
            sql += " WHERE episode_id = %s"
            params.append(episode_id)
        
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding, top_k])
        
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
