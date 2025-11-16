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
        vlm_descriptions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Optionally refine EDL using LLM for nuanced decisions.
        
        Args:
            episode_id: Episode identifier
            base_edl: Rule-based EDL cuts
            transcript: Full transcript with speaker labels
            role_mapping: Speaker ID to role mapping (e.g., 'SPEAKER_00' -> 'speaker_00')
        
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
                vlm_descriptions,
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
        vlm_descriptions: Optional[List[Dict[str, Any]]] = None,
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
            # MODIFIED: Use role_mapping to get 'speaker_00', 'speaker_01', etc.
            speaker_label = role_mapping.get(speaker, speaker)
            
            if speaker != current_speaker:
                if current_line:
                    transcript_lines.append(f"{speaker_label}: {' '.join(current_line)}")
                current_speaker = speaker
                current_line = []
            
            current_line.append(word_data['word'])
        
        if current_line:
            # MODIFIED: Use speaker_label for the last line
            speaker_label = role_mapping.get(current_speaker, current_speaker)
            transcript_lines.append(f"{speaker_label}: {' '.join(current_line)}")
        
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
                
        if vlm_descriptions and self.rag_store:
            vlm_context = self.rag_store.retrieve_vlm_context(
                episode_id=episode_id,
                start_time=start_time,
                end_time=end_time,
                top_k=5,
            )
            if vlm_context:
                context["visual_descriptions"] = [
                    {
                        "time": f"{v['time_seconds']:.1f}s",
                        "camera": v['camera_id'],
                        "description": v['description'],
                    }
                    for v in vlm_context
                ]
        
        return context
    
    def _build_refinement_prompt(
    self,
    cuts: List[Dict[str, Any]],
    context: Dict[str, Any],
    role_mapping: Dict[str, str],
    ) -> str:
        """
        MODIFIED: Build a generic prompt for LLM refinement.
        """
        
        vlm_section = ""
        if context.get("visual_descriptions"):
            vlm_section = "\n**Visual Context:**\n"
            for vd in context["visual_descriptions"]:
                vlm_section += f"- {vd['time']} [{vd['camera']}]: {vd['description']}\n"
        
        # Get all camera IDs mentioned in the cuts
        all_cameras = sorted(list(set(c['camera_id'] for c in cuts)))
        # Get all speaker IDs mentioned in the transcript
        all_speakers = sorted(list(set(role_mapping.values())))

        prompt = f"""You are an expert video editor. Your task is to refine a sequence of video cuts based on a transcript and visual context.

        **Context:**
        Time: {context['start_time']:.1f}s - {context['end_time']:.1f}s
        Available Cameras: {', '.join(all_cameras)}
        Visible Speakers: {', '.join(all_speakers)}

        **Transcript:**
        {context['transcript']}
        
        **VLM description (if available):**
        {vlm_section}
        
        **Current Cuts (rule-based):**
        {json.dumps(cuts, indent=2)}

        **Your Task:**
        Review the cuts and make adjustments to improve flow and viewer engagement.

        **Critical Guidelines:**

        1. **Prioritize Conversational Flow:**
           - The goal is a natural, engaging conversation.
           - Cut when the *focus* of the conversation changes (e.EXAMPLE., a new speaker, a reaction, an interjection).
           - Longer, continuous shots (5-15 seconds) are generally good, but *not* if they miss important reactions.

        2. **Use Wide vs. Close-up Shots Effectively:**
           - **Wide Shots** (cameras showing multiple people): Use these during rapid back-and-forth exchanges (e.g., speaker turns < 8 seconds), or to capture group reactions and laughter.
           - **Close-up Shots** (cameras focused on one person): Use these when a speaker is talking for an extended period (8+ seconds) to build connection.
        
        3. **Shot Duration Guidelines:**
           - Preferred: 4-10 seconds.
           - Acceptable longer: 15-20+ seconds for engaging monologues.
           - *Avoid* excessively long, static shots (30+ seconds).

        4. **When to Cut:**
           - ✅ Natural speaker changes (if the new speaker talks for 2+ seconds).
           - ✅ Clear topic transitions.
           - ✅ **Significant reactions** (laughter, surprise) -> Use a wide shot or cut to the reactor.
           - ❌ Avoid cutting mid-sentence *unless* it's to catch an important interjection.

        **Output Requirements:**
        - Do not be afraid to keep more cuts if the rule-based ones follow the conversation well.
        - **Merge cuts only if they are on the same camera and are unnecessarily short (< 2 seconds).**
        - Ensure NO GAPS between cuts (each cut must start exactly where the previous ended).
        - Round all times to 0.033s (30fps frame boundaries).

        Respond with JSON only:
        {{
        "cuts": [
            {{"start_time": 0.0, "end_time": 5.0, "camera_id": "cam_a", "reason": "speaker_00 opening"}},
            ...
        ]
        }}"""
        
        return prompt
