
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
                
                # Parse response and sanitize
                refined_output = json.loads(response.text)
                llm_cuts = refined_output.get("cuts", segment_cuts)
                
                sanitized_cuts = []
                for cut in llm_cuts:
                    try:
                        cut['start_time'] = float(cut['start_time'])
                        cut['end_time'] = float(cut['end_time'])
                        sanitized_cuts.append(cut)
                    except (ValueError, TypeError, KeyError) as e:
                        logger.warning(f"Skipping malformed cut from LLM due to {e}: {cut}")
                
                refined_edl.extend(sanitized_cuts)
                
            except Exception as e:
                logger.warning(f"LLM refinement failed for segment: {e}")
                refined_edl.extend(segment_cuts)
        
        logger.info(f"✅ Refined EDL: {len(base_edl)} → {len(refined_edl)} cuts")
        return refined_edl
    
# processing/llm_refiner.py (FIX _build_context method)

    def _build_context(
        self,
        episode_id: str,
        start_time: float,
        end_time: float,
        transcript: List[Dict[str, Any]],
        role_mapping: Dict[str, str],
        vlm_descriptions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Build context for LLM including RAG and VLM.
        No artificial limits - Gemini 2.5 Pro can handle large context.
        """
        
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
        
        # Build context dictionary with ALL required fields
        context = {
            "transcript": "\n".join(transcript_lines),
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,  # ✅ ADD THIS
            "word_count": len(segment_transcript),
        }
        
        # Add RAG context (semantic search on transcript)
        if self.rag_store and self.settings.USE_RAG:
            query = " ".join(transcript_lines[:3])
            
            rag_chunks = self.rag_store.retrieve(
                query=query,
                top_k=self.settings.RAG_TOP_K,
                episode_id=episode_id,
            )
            
            if rag_chunks:
                context["similar_moments"] = [
                    {
                        "time": f"{c['start_time']:.1f}s - {c['end_time']:.1f}s",
                        "text": c['text'],
                        "similarity": c['similarity'],
                    }
                    for c in rag_chunks
                ]
        
        # Add ALL VLM context in time window
        if self.rag_store and vlm_descriptions:
            # Retrieve ALL VLM descriptions in this time window
            vlm_context = self.rag_store.retrieve_vlm_context(
                episode_id=episode_id,
                start_time=start_time - 2.0,  # Include 2s buffer before
                end_time=end_time + 2.0,      # Include 2s buffer after
            )
            
            if vlm_context:
                # Group by cut time, then by camera for structured presentation
                by_cut_time = {}
                for v in vlm_context:
                    cut_time = v.get('transition_time', v['time_seconds'])
                    if cut_time not in by_cut_time:
                        by_cut_time[cut_time] = {}
                    
                    cam = v['camera_id']
                    if cam not in by_cut_time[cut_time]:
                        by_cut_time[cut_time][cam] = []
                    
                    by_cut_time[cut_time][cam].append({
                        "time": v['time_seconds'],
                        "offset": v.get('offset_seconds', 0),
                        "description": v['description'],
                    })
                
                context["visual_descriptions"] = by_cut_time
                context["vlm_description_count"] = len(vlm_context)
                
                logger.debug(f"Added {len(vlm_context)} VLM descriptions to context")
        
        return context
    
# processing/llm_refiner.py (UPDATED PROMPT)

    def _build_refinement_prompt(
        self,
        cuts: List[Dict[str, Any]],
        context: Dict[str, Any],
        role_mapping: Dict[str, str],
    ) -> str:
        """
        Build prompt with podcast-aware context.
        """
        
        settings = get_settings()
        
        # Build camera setup description
        camera_setup_desc = ""
        if hasattr(settings, 'CAMERA_SETUP') and settings.CAMERA_SETUP:
            camera_setup_desc = "\n**Camera Setup:**\n"
            for camera_id, description in settings.CAMERA_SETUP.items():
                camera_setup_desc += f"- {camera_id}: {description}\n"
        
        prompt = f"""You are an expert video editor specializing in multi-camera podcast and conversational content.

    **Setup:**
    Format: {settings.SETUP_TYPE}
    Participants: 3 people
    Duration: {context['duration']:.1f} seconds

    {camera_setup_desc}

    **Important Notes:**
    - This is a THREE-PERSON PODCAST (not host/guest)
    - There are THREE participants but only TWO dedicated closeup cameras
    - cam_a shows Person A (woman) in closeup
    - cam_b shows Person B (man) in closeup
    - cam_wide shows all three participants (Person A, Person B, and Person C who has no dedicated closeup)
    - Person C can only be shown via cam_wide

    **Context:**
    Time Range: {context['start_time']:.1f}s - {context['end_time']:.1f}s
    Duration: {context['duration']:.1f}s
    Speaker Count: {len(set(role_mapping.values()))}

    **Transcript:**
    {context['transcript']}

    **Current Cuts (rule-based):**
    {json.dumps(cuts, indent=2)}
    """
        
        # Add VLM context if available
        if context.get("visual_descriptions"):
            vlm_count = context.get('vlm_description_count', 0)
            prompt += f"\n**Visual Context at Cut Boundaries ({vlm_count} descriptions):**\n"
            
            for cut_time in sorted(context["visual_descriptions"].keys()):
                cut_data = context["visual_descriptions"][cut_time]
                
                prompt += f"\n--- Cut at {cut_time:.1f}s ---\n"
                
                for camera_id in sorted(cut_data.keys()):
                    camera_descriptions = cut_data[camera_id]
                    
                    prompt += f"\n{camera_id}:\n"
                    
                    sorted_descriptions = sorted(camera_descriptions, key=lambda x: x['offset'])
                    
                    for desc in sorted_descriptions:
                        offset = desc['offset']
                        time = desc['time']
                        description = desc['description']
                        
                        offset_str = f"+{offset:.1f}s" if offset >= 0 else f"{offset:.1f}s"
                        prompt += f"  [{offset_str}]: {description}\n"
        
        # Add podcast-specific editing guidelines
        prompt += """

    **Podcast Editing Guidelines:**

    1.  **Prioritize the Speaker:** The primary camera should be on the person speaking. Use their dedicated closeup (cam_a, cam_b) for engaging, personal moments.
    2.  **Show Reactions:** Cut to other participants for reactions (nodding, laughing, surprise). A reaction shot on a closeup camera is powerful. If multiple people are reacting, or the reaction is a group dynamic, `cam_wide` is a good choice.
    3.  **Use the Wide Shot Strategically:** The `cam_wide` shot is your versatile tool. Use it to:
        *   **Establish the scene:** Start segments with a wide shot to ground the viewer.
        *   **Cover transitions:** When switching between speakers, a brief cut to wide can smooth the transition.
        *   **Show the group dynamic:** If everyone is talking or laughing, the wide shot captures that energy.
        *   **Default for Person C:** When Person C (who has no closeup) is speaking, you must use `cam_wide`.
    4.  **Pacing and Rhythm:**
        *   **Avoid rapid-fire cuts:** Let the conversation breathe. Shots should be at least 3-4 seconds long.
        *   **Vary shot duration:** Mix longer closeup shots (7-15 seconds) with shorter reaction shots or wide shots.
        *   **Match the energy:** Faster-paced conversation can have slightly quicker cuts. A serious monologue should have a long, focused shot.
    5.  **Analyze the Visuals:** Use the "Visual Context" descriptions. If a person is looking away, disengaged, or not in their shot, do not cut to their camera. The visual descriptions confirm who is present and engaged.
    6.  **Continuity is Key:**
        *   Don't cut away from a speaker mid-sentence unless it's for a very compelling reaction.
        *   Ensure a smooth flow. Avoid jarring jumps between camera angles (e.g., closeup -> wide -> closeup in rapid succession).

    **Output Requirements:**
    - Keep 70-80% of original cuts unchanged
    - Only modify if it improves the viewing experience for podcast format
    - Ensure no gaps between cuts (continuous timeline)
    - Round all times to 0.033s (30fps frame boundaries)
    - Provide reasoning for changes, especially camera selections

    Respond with JSON only:
    {{
      "cuts": [
        {{"start_time": 0.0, "end_time": 4.5, "camera_id": "cam_wide", "reason": "opening shot, all three participants visible"}},
        {{"start_time": 4.5, "end_time": 9.2, "camera_id": "cam_a", "reason": "Person A speaking, cam_a closeup shows engagement"}},
        {{...}}
      ]
    }}"""
        
        estimated_tokens = len(prompt) // 4
        logger.info(f"LLM prompt estimated tokens: ~{estimated_tokens} (well under 400K limit)")
        
        return prompt
