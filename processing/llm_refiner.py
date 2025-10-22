
# processing/llm_refiner.py
import logging
from typing import List, Dict, Any, Optional
import json
from google import genai
from google.genai import types
from config import get_settings
from rag.pgvector_store import PGVectorRAGStore

logger = logging.getLogger(__name__)


class LLMRefiner:
    """Optional LLM refinement for EDL using Google Gemini via Vertex AI."""
    
    def __init__(self, rag_store: Optional[PGVectorRAGStore] = None):
        self.settings = get_settings()
        self.rag_store = rag_store
        
        # Initialize Gemini client
        if self.settings.GOOGLE_GENAI_USE_VERTEXAI:
            logger.info("Initializing Gemini with Vertex AI (ADC)")
            self.client = genai.Client(
                vertexai=True,
                project=self.settings.GOOGLE_CLOUD_PROJECT,
                location=self.settings.GOOGLE_CLOUD_LOCATION,
            )
        else:
            logger.info("Initializing Gemini with API key")
            self.client = genai.Client(
                api_key=self.settings.GOOGLE_API_KEY,
            )
        
        logger.info(f"✅ LLM refiner initialized (model: {self.settings.GEMINI_MODEL})")
    
    def refine_edl(
        self,
        episode_id: str,
        base_edl: List[Dict[str, Any]],
        transcript: List[Dict[str, Any]],
        role_mapping: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Optionally refine EDL using LLM for nuanced decisions.
        
        Args:
            episode_id: Episode identifier
            base_edl: Rule-based EDL cuts
            transcript: Full transcript with speaker labels
            role_mapping: Speaker ID to role mapping
        
        Returns:
            Refined EDL cuts
        """
        if not self.settings.REFINE_WITH_LLM:
            logger.info("LLM refinement disabled, using rule-based EDL")
            return base_edl
        
        logger.info("Refining EDL with LLM...")
        
        # Process in segments for manageable context
        segment_duration = 120.0  # 2-minute segments
        refined_edl = []
        
        for i in range(0, len(base_edl), 10):  # Process ~10 cuts at a time
            segment_cuts = base_edl[i:i+10]
            if not segment_cuts:
                break
            
            start_time = segment_cuts[0]['start_time']
            end_time = segment_cuts[-1]['end_time']
            
            # Get relevant context
            context = self._build_context(
                episode_id,
                start_time,
                end_time,
                transcript,
                role_mapping,
            )
            
            # Build prompt
            prompt = self._build_refinement_prompt(segment_cuts, context, role_mapping)
            
            # Call LLM
            try:
                response = self.client.models.generate_content(
                    model=self.settings.GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.settings.LLM_TEMPERATURE,
                        response_mime_type="application/json",
                    ),
                )
                
                # Parse response
                refined_segment = json.loads(response.text)
                refined_edl.extend(refined_segment.get("cuts", segment_cuts))
                
            except Exception as e:
                logger.warning(f"LLM refinement failed for segment: {e}")
                refined_edl.extend(segment_cuts)
        
        logger.info(f"✅ Refined EDL: {len(base_edl)} → {len(refined_edl)} cuts")
        return refined_edl
    
    def _build_context(
        self,
        episode_id: str,
        start_time: float,
        end_time: float,
        transcript: List[Dict[str, Any]],
        role_mapping: Dict[str, str],
    ) -> Dict[str, Any]:
        """Build context for LLM including RAG retrieval."""
        # Extract transcript segment
        segment_transcript = [
            w for w in transcript
            if start_time <= w['start'] <= end_time
        ]
        
        # Format transcript with speaker labels
        transcript_lines = []
        current_speaker = None
        current_line = []
        
        for word_data in segment_transcript:
            speaker = word_data.get('speaker', 'unknown')
            role = role_mapping.get(speaker, speaker)
            
            if speaker != current_speaker:
                if current_line:
                    transcript_lines.append(f"{role}: {' '.join(current_line)}")
                current_speaker = speaker
                current_line = []
            
            current_line.append(word_data['word'])
        
        if current_line:
            transcript_lines.append(f"{role}: {' '.join(current_line)}")
        
        context = {
            "transcript": "\n".join(transcript_lines),
            "start_time": start_time,
            "end_time": end_time,
        }
        
        # Add RAG context if available
        if self.rag_store and self.settings.USE_RAG:
            query = " ".join(transcript_lines[:5])  # Use beginning as query
            rag_chunks = self.rag_store.retrieve(
                query=query,
                top_k=self.settings.RAG_TOP_K,
                episode_id=episode_id,
            )
            
            if rag_chunks:
                context["similar_moments"] = [
                    {
                        "time": f"{c['start_time']:.1f}s - {c['end_time']:.1f}s",
                        "text": c['text'][:200],
                    }
                    for c in rag_chunks
                ]
        
        return context
    
    def _build_refinement_prompt(
        self,
        cuts: List[Dict[str, Any]],
        context: Dict[str, Any],
        role_mapping: Dict[str, str],
    ) -> str:
        """Build prompt for LLM refinement."""
        reverse_mapping = {v: k for k, v in role_mapping.items()}
        
        prompt = f"""You are an expert video editor reviewing camera cuts for a multi-camera conversation.

**Context:**
Time: {context['start_time']:.1f}s - {context['end_time']:.1f}s

**Transcript:**
{context['transcript']}

**Current Cuts (rule-based):**
{json.dumps(cuts, indent=2)}

**Your Task:**
Review the cuts and make minor adjustments if needed for:
1. Better pacing and flow
2. Capturing important reactions
3. Avoiding jarring cuts during key moments
4. Maintaining visual interest

**Rules:**
- Keep most cuts unchanged if they're good
- Minimum shot duration: 2.0 seconds
- Available cameras: cam_host, cam_guest, cam_wide
- Preserve timing accuracy (round to 0.033s for 30fps)

Respond with JSON only:
{{
  "cuts": [
    {{"start_time": 0.0, "end_time": 2.5, "camera_id": "cam_host", "reason": "speaker"}},
    ...
  ]
}}"""
        
        return prompt
