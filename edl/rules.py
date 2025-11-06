# edl/rules.py
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Cut:
    """Represents a single video cut/clip."""
    start_time: float
    end_time: float
    camera_id: str
    reason: str = "default"
    speaker_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'camera_id': self.camera_id,
            'reason': self.reason,
        }


class RuleBasedEDLGenerator:
    """Generate EDL from speaker diarization using deterministic rules."""
    
    def __init__(
        self,
        fps: int = 30,
        min_shot_s: float = 2.0,
        wide_open_s: float = 3.0,
        rapid_window_s: float = 8.0,
        rapid_changes: int = 3,
        reaction_keywords: Optional[List[str]] = None,
    ):
        """
        Initialize EDL generator.
        
        Args:
            fps: Frame rate (default 30fps)
            min_shot_s: Minimum shot duration in seconds
            wide_open_s: Duration to show wide shot at start
            rapid_window_s: Window for detecting rapid changes
            rapid_changes: Threshold for rapid changes
            reaction_keywords: Words that trigger reaction shots
        """
        self.settings = get_settings()
        self.fps = fps
        self.min_shot_s = min_shot_s
        self.wide_open_s = wide_open_s
        self.rapid_window_s = rapid_window_s
        self.rapid_changes = rapid_changes
        self.reaction_keywords = reaction_keywords or [
            "laugh", "react", "gasp", "wow", "amazing", "shocked", "surprised"
        ]
        
        self.setup_type = getattr(self.settings, 'SETUP_TYPE', 'host_guest')
        
        logger.info(f"EDL generator initialized: {self.setup_type}")
        logger.info(f"  Min shot: {min_shot_s}s, Wide opening: {wide_open_s}s")
    
    def generate_edl(
        self,
        speaker_segments: List[Dict[str, Any]],
        role_mapping: Dict[str, str],
        transcript: Optional[List[Dict[str, Any]]] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate rule-based EDL from speaker diarization.
        
        This ALWAYS uses speaker diarization data (speaker_segments, role_mapping).
        Setup type (host_guest, podcast_3p, panel) only affects camera selection logic.
        
        Args:
            speaker_segments: List of speaker segments from diarization
            role_mapping: Mapping of speaker_id to role
            transcript: Optional transcript with word-level timing
            start_time: Start time for processing
            end_time: End time for processing
        
        Returns:
            Dictionary with 'cuts' key containing list of EDL cuts
        """
        
        if not speaker_segments:
            logger.warning("No speaker segments provided")
            return {"cuts": []}
        
        logger.info(f"Generating {self.setup_type} EDL from {len(speaker_segments)} speaker segments")
        
        # Determine the end time
        if end_time is None:
            end_time = speaker_segments[-1]['end'] + 1.0
        
        # Route to appropriate EDL generation method based on setup type
        if self.setup_type == "podcast_3p":
            cuts = self._generate_podcast_3p_edl(
                speaker_segments,
                role_mapping,
                transcript,
                start_time,
                end_time,
            )
        elif self.setup_type == "panel":
            cuts = self._generate_panel_edl(
                speaker_segments,
                role_mapping,
                transcript,
                start_time,
                end_time,
            )
        else:  # Default: host_guest
            cuts = self._generate_host_guest_edl(
                speaker_segments,
                role_mapping,
                transcript,
                start_time,
                end_time,
            )
        
        # Post-processing
        logger.info(f"Generated {len(cuts)} initial cuts")
        
        # Enforce minimum duration
        cuts = self._enforce_min_duration(cuts, start_time, end_time)
        logger.info(f"After min duration enforcement: {len(cuts)} cuts")
        
        # Eliminate gaps
        cuts = self._eliminate_gaps(cuts, start_time, end_time)
        logger.info(f"After gap elimination: {len(cuts)} cuts")
        
        # Round to frames and validate
        final = self._round_and_validate(cuts, start_time, end_time)
        logger.info(f"✅ Final EDL: {len(final)} cuts")
        
        return {"cuts": final}
    
    def _generate_host_guest_edl(
        self,
        speaker_segments: List[Dict[str, Any]],
        role_mapping: Dict[str, str],
        transcript: Optional[List[Dict[str, Any]]],
        start_time: float,
        end_time: float,
    ) -> List[Dict[str, Any]]:
        """
        Generate EDL for host-guest format.
        Two main cameras: host and guest.
        """
        cuts = []
        
        for i, seg in enumerate(speaker_segments):
            speaker_id = seg['speaker_id']
            role = role_mapping.get(speaker_id, f"speaker_{speaker_id}")
            
            # Map role to camera
            if "host" in role.lower():
                camera = "cam_a"  # Host camera
                reason = "host speaking"
            elif "guest" in role.lower():
                camera = "cam_b"  # Guest camera
                reason = "guest speaking"
            else:
                # Default to wide for unknown roles
                camera = "cam_wide"
                reason = f"{role} speaking"
            
            cut = {
                'start_time': seg['start'],
                'end_time': seg['end'],
                'camera_id': camera,
                'reason': reason,
                'speaker_id': speaker_id,
            }
            
            cuts.append(cut)
        
        return cuts
    
    def _generate_podcast_3p_edl(
        self,
        speaker_segments: List[Dict[str, Any]],
        role_mapping: Dict[str, str],
        transcript: Optional[List[Dict[str, Any]]],
        start_time: float,
        end_time: float,
    ) -> List[Dict[str, Any]]:
        """
        Generate EDL for 3-person podcast format.
        Three participants: 2 with closeup cameras + 1 only visible in wide shot.
        Uses diarization to determine who is speaking.
        """
        cuts = []
        
        logger.debug(f"Podcast 3P setup - Role mapping: {role_mapping}")
        
        for i, seg in enumerate(speaker_segments):
            speaker_id = seg['speaker_id']
            role = role_mapping.get(speaker_id, f"speaker_{speaker_id}")
            
            # Map speaker to camera based on role from diarization
            camera = self._map_speaker_to_camera_3p(role, speaker_id)
            
            cut = {
                'start_time': seg['start'],
                'end_time': seg['end'],
                'camera_id': camera,
                'reason': f"{role} speaking",
                'speaker_id': speaker_id,
            }
            
            cuts.append(cut)
        
        return cuts
    
    def _generate_panel_edl(
        self,
        speaker_segments: List[Dict[str, Any]],
        role_mapping: Dict[str, str],
        transcript: Optional[List[Dict[str, Any]]],
        start_time: float,
        end_time: float,
    ) -> List[Dict[str, Any]]:
        """
        Generate EDL for panel discussion format.
        Multiple participants, favor wide shots with occasional closeups.
        """
        cuts = []
        
        for i, seg in enumerate(speaker_segments):
            speaker_id = seg['speaker_id']
            role = role_mapping.get(speaker_id, f"speaker_{speaker_id}")
            
            # For panels, prefer wide shot by default
            camera = "cam_wide"
            
            # Only use closeup if segment is long (speaker explaining something)
            if (seg['end'] - seg['start']) > 8.0:
                # Try to find a closeup camera for this speaker
                if "person_a" in role.lower() or "speaker_1" in role.lower():
                    camera = "cam_a"
                elif "person_b" in role.lower() or "speaker_2" in role.lower():
                    camera = "cam_b"
            
            cut = {
                'start_time': seg['start'],
                'end_time': seg['end'],
                'camera_id': camera,
                'reason': f"{role} speaking",
                'speaker_id': speaker_id,
            }
            
            cuts.append(cut)
        
        return cuts
    
    def _map_speaker_to_camera_3p(self, role: str, speaker_id: str) -> str:
        """
        Map speaker to camera for 3-person podcast.
        Uses diarization roles to determine camera selection.
        
        Args:
            role: Speaker role from role_mapping (e.g., "host", "guest", "person_a")
            speaker_id: Speaker identifier (e.g., "SPEAKER_00")
        
        Returns:
            Camera ID (cam_a, cam_b, or cam_wide)
        """
        role_lower = role.lower()
        
        # Check for explicit roles first
        if "person_a" in role_lower or "host" in role_lower:
            return "cam_a"
        
        elif "person_b" in role_lower or "guest" in role_lower:
            return "cam_b"
        
        elif "person_c" in role_lower:
            return "cam_wide"  # Person C has no closeup camera
        
        # For generic roles, fall back to wide shot
        return "cam_wide"
    
    def _enforce_min_duration(
        self,
        cuts: List[Dict[str, Any]],
        start_time: float,
        end_time: float,
    ) -> List[Dict[str, Any]]:
        """
        Merge consecutive cuts from same camera if duration is below minimum.
        Prevents jarring quick cuts.
        """
        if not cuts:
            return cuts
        
        merged = []
        current = cuts[0].copy()
        
        for i in range(1, len(cuts)):
            next_cut = cuts[i]
            current_duration = current['end_time'] - current['start_time']
            
            # If current cut is too short and same camera, merge
            if (current_duration < self.min_shot_s and 
                current['camera_id'] == next_cut['camera_id']):
                # Extend current cut to include next
                current['end_time'] = next_cut['end_time']
            else:
                # Keep current cut and move to next
                merged.append(current)
                current = next_cut.copy()
        
        # Add final cut
        merged.append(current)
        
        logger.debug(f"After min duration: {len(cuts)} → {len(merged)} cuts")
        return merged
    
    def _eliminate_gaps(
        self,
        cuts: List[Dict[str, Any]],
        start_time: float,
        end_time: float,
    ) -> List[Dict[str, Any]]:
        """
        Eliminate gaps between cuts by extending clips to touch each other.
        Ensures continuous timeline with no blank frames.
        """
        if not cuts:
            return cuts
        
        result = []
        
        for i, cut in enumerate(cuts):
            adjusted_cut = cut.copy()
            
            if i == 0:
                # First cut - ensure it starts at segment start
                adjusted_cut['start_time'] = start_time
            else:
                # Subsequent cuts - start exactly where previous ended
                adjusted_cut['start_time'] = result[-1]['end_time']
            
            if i == len(cuts) - 1:
                # Last cut - extend to segment end
                adjusted_cut['end_time'] = end_time
            else:
                # Check for gap with next cut
                next_cut = cuts[i + 1]
                if adjusted_cut['end_time'] < next_cut['start_time']:
                    # Gap detected - extend this cut to touch next one
                    adjusted_cut['end_time'] = next_cut['start_time']
            
            result.append(adjusted_cut)
        
        logger.debug(f"Gaps eliminated: {len(cuts)} cuts verified")
        return result
    
    def _round_and_validate(
        self,
        cuts: List[Dict[str, Any]],
        start_time: float,
        end_time: float,
    ) -> List[Dict[str, Any]]:
        """
        Round timestamps to frame boundaries and validate cuts.
        Ensures all times align to 1/fps precision.
        """
        frame_duration = 1.0 / self.fps
        validated = []
        
        for i, cut in enumerate(cuts):
            # Round to nearest frame
            start = round(cut['start_time'] / frame_duration) * frame_duration
            end = round(cut['end_time'] / frame_duration) * frame_duration
            
            # Ensure minimum duration
            if (end - start) < self.min_shot_s:
                logger.warning(f"Cut {i} too short ({end - start:.3f}s), extending")
                end = start + self.min_shot_s
            
            # Ensure within bounds
            start = max(start, start_time)
            end = min(end, end_time)
            
            # Skip invalid cuts
            if start >= end:
                logger.warning(f"Cut {i} is invalid (start >= end), skipping")
                continue
            
            validated_cut = cut.copy()
            validated_cut['start_time'] = start
            validated_cut['end_time'] = end
            
            validated.append(validated_cut)
        
        return validated
    
    def _has_reaction(self, text: str) -> bool:
        """Check if text contains reaction keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.reaction_keywords)
