# edl/rules.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Cut:
    """Represents a single cut in the edit decision list."""
    start_time: float
    end_time: float
    camera_id: str
    reason: str = "speaker"  # For debugging


def round_to_frames(t: float, fps: float) -> float:
    """Round timestamp to nearest frame boundary."""
    return round(t * fps) / fps


class RuleBasedEDLGenerator:
    """Generate EDL from speaker diarization using deterministic rules."""
    
    def __init__(
        self,
        fps: float,
        min_shot_s: float = 2.0,
        wide_open_s: float = 3.0,
        rapid_window_s: float = 8.0,
        rapid_changes: int = 3,
        reaction_keywords: Optional[List[str]] = None,
    ):
        self.fps = fps
        self.min_shot_s = min_shot_s
        self.wide_open_s = wide_open_s
        self.rapid_window_s = rapid_window_s
        self.rapid_changes = rapid_changes
        self.reaction_keywords = reaction_keywords or []
        
    def _eliminate_gaps(
        self,
        cuts: List[Cut],
        start: float,
        end: float,
    ) -> List[Cut]:
        """
        Eliminate gaps between cuts by extending clips to touch each other.
        Ensures continuous timeline with no blank frames.
        """
        if not cuts:
            return cuts
        
        result = []
        
        for i, cut in enumerate(cuts):
            if i == 0:
                # First cut - ensure it starts at timeline start
                cut.start_time = start
            else:
                # Subsequent cuts - start exactly where previous cut ended
                cut.start_time = result[-1].end_time
            
            if i == len(cuts) - 1:
                # Last cut - extend to end of timeline
                cut.end_time = end
            else:
                # Check for gap with next cut
                next_cut = cuts[i + 1]
                if cut.end_time < next_cut.start_time:
                    # Gap detected - extend this cut to touch next one
                    gap_size = next_cut.start_time - cut.end_time
                    if gap_size <= 0.5:  # Small gap - extend current cut
                        cut.end_time = next_cut.start_time
                    else:
                        # Large gap - could be intentional, but still fill it
                        logger.warning(f"Large gap detected: {gap_size:.2f}s at {cut.end_time:.1f}s")
                        cut.end_time = next_cut.start_time
            
            result.append(cut)
        
        return result
    
    def generate_edl(
        self,
        speaker_segments: List[Dict[str, Any]],
        role_mapping: Dict[str, str],
        role_camera_map: Dict[str, str],
        transcript: Optional[List[Dict[str, Any]]] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate rule-based EDL from speaker segments.
        
        Args:
            speaker_segments: List of diarization segments
            role_mapping: Maps speaker_id to role (host/guest)
            transcript: Optional transcript for reaction detection
            start_time: Start time of segment to process
            end_time: End time of segment to process
        
        Returns:
            Dictionary with 'cuts' list
        """
        if end_time is None:
            end_time = max(seg['end'] for seg in speaker_segments)
        
        logger.info(f"Generating EDL for {start_time:.1f}s - {end_time:.1f}s")
        
        # Get default wide cam
        wide_cam = role_camera_map.get("default_wide", "cam_wide")
        
        # Step 1: Build speaker turns
        turns = self._build_turns(speaker_segments, role_mapping, role_camera_map, wide_cam, start_time, end_time)
        if not turns:
            return {"cuts": []}
        
        # Step 2: Merge consecutive same-camera shots
        merged = self._merge_consecutive(turns)
        
        # Step 3: Insert opening wide shot
        merged = self._insert_opening_wide(merged, wide_cam, start_time, end_time)
        
        # Step 4: Detect rapid exchanges and use wide shot
        merged = self._handle_rapid_exchanges(merged, wide_cam, start_time, end_time)
        
        # Step 5: Insert reaction shots (if transcript provided)
        if transcript:
            merged = self._insert_reaction_shots(merged, transcript, role_mapping, role_camera_map, start_time, end_time)
        # Step 6: Enforce minimum shot duration
        merged = self._enforce_min_duration(merged, start_time, end_time)
        
        # Step 7: Eliminate gaps (NEW)
        merged = self._eliminate_gaps(merged, start_time, end_time)
        
        # Step 8: Round to frames and validate
        final = self._round_and_validate(merged, start_time, end_time)
        
        logger.info(f"✅ Generated {len(final)} cuts")
        return {"cuts": final}
    
    def _build_turns(
        self,
        speaker_segments: List[Dict[str, Any]],
        role_mapping: Dict[str, str],
        role_camera_map: Dict[str, str],
        wide_cam: str,
        start: float,
        end: float,
    ) -> List[Cut]:
        """Build initial camera cuts from speaker segments."""
        turns = []
        for seg in speaker_segments:
            seg_start = max(seg['start'], start)
            seg_end = min(seg['end'], end)
            if seg_end <= seg_start:
                continue
            
            role = role_mapping.get(seg['speaker_id'], 'unknown')
            camera = role_camera_map.get(role, wide_cam)
            turns.append(Cut(seg_start, seg_end, camera, f"speaker_{role}"))
        
        return turns
    
    def _role_to_camera(self, role: str) -> str:
        """Map speaker role to camera ID."""
        if role == 'host':
            return 'cam_host'
        elif role == 'guest':
            return 'cam_guest'
        else:
            return 'cam_wide'
    
    def _merge_consecutive(self, cuts: List[Cut]) -> List[Cut]:
        """Merge consecutive cuts from the same camera."""
        if not cuts:
            return []
        
        merged = []
        current = cuts[0]
        
        for next_cut in cuts[1:]:
            # Merge if same camera and close in time
            if next_cut.camera_id == current.camera_id and next_cut.start_time <= current.end_time + 0.25:
                current.end_time = max(current.end_time, next_cut.end_time)
            else:
                merged.append(current)
                current = next_cut
        
        merged.append(current)
        return merged
    
    def _insert_opening_wide(self, cuts: List[Cut], wide_cam: str, start: float, end: float) -> List[Cut]:
        """Insert wide shot at the beginning."""
        if not cuts:
            return cuts
        
        first_cut = cuts[0]
        open_end = min(first_cut.start_time + self.wide_open_s, first_cut.end_time, end)
        
        if open_end - start >= 1.0:
            # Insert opening wide
            opening = Cut(start, open_end, wide_cam, 'opening')
            
            # Adjust first cut if it overlaps
            if first_cut.start_time < open_end:
                first_cut.start_time = open_end
            
            return [opening] + cuts
        
        return cuts
    
    def _handle_rapid_exchanges(self, cuts: List[Cut], wide_cam: str, start: float, end: float) -> List[Cut]:
        """Replace rapid back-and-forth with wide shot."""
        if len(cuts) < 2:
            return cuts
        
        result = []
        i = 0
        
        while i < len(cuts):
            window_start = cuts[i].start_time
            j = i
            changes = 0
            last_camera = cuts[i].camera_id
            
            # Count camera changes within window
            while j < len(cuts) and cuts[j].start_time - window_start <= self.rapid_window_s:
                if cuts[j].camera_id != last_camera:
                    changes += 1
                    last_camera = cuts[j].camera_id
                j += 1
            
            # If rapid changes detected, replace with wide shot
            if changes >= self.rapid_changes:
                window_end = cuts[min(j - 1, len(cuts) - 1)].end_time
                result.append(Cut(window_start, window_end, wide_cam, 'rapid_exchange'))
                i = j
            else:
                result.append(cuts[i])
                i += 1
        
        return result
    
    def _insert_reaction_shots(
        self,
        cuts: List[Cut],
        transcript: List[Dict[str, Any]],
        role_mapping: Dict[str, str],
        role_camera_map: Dict[str, str],
        start: float,
        end: float,
    ) -> List[Cut]:
        """Insert reaction shots based on keywords in transcript."""
        # Detect reaction moments
        reaction_times = []
        for word_data in transcript:
            if word_data['start'] < start or word_data['start'] >= end:
                continue
            word = word_data['word'].lower()
            if any(keyword in word for keyword in self.reaction_keywords):
                speaker = word_data.get('speaker', 'unknown')
                reaction_times.append({
                    'time': word_data['start'],
                    'speaker': speaker,
                })
        
        if not reaction_times:
            return cuts
        
        # Insert reaction shots (showing non-speaker)
        result = []
        for cut in cuts:
            result.append(cut)
            
            # Check if any reactions fall within this cut
            for reaction in reaction_times:
                if cut.start_time <= reaction['time'] <= cut.end_time:
                    
                    # Find role of speaker (e.g., 'host')
                    speaker_role = role_mapping.get(reaction['speaker'], 'unknown')
                    
                    # Find role of non-speaker (e.g., 'guest')
                    # This is a simple toggle, assumes 2 people
                    reactor_role = 'guest' if speaker_role == 'host' else 'host'
                    
                    # Get the camera for the reactor
                    reaction_camera = role_camera_map.get(reactor_role)
                    
                    # Only insert if we have a camera for the reactor
                    # and it's different from the current camera
                    if reaction_camera and reaction_camera != cut.camera_id:
                        reaction_start = reaction['time']
                        reaction_end = min(reaction_start + 2.0, cut.end_time)
                        
                        if reaction_end - reaction_start >= 1.0:
                            result.append(Cut(
                                reaction_start,
                                reaction_end,
                                reaction_camera,
                                'reaction'
                            ))
        
        # Sort and remove overlaps
        result.sort(key=lambda x: x.start_time)
        return self._remove_overlaps(result)
    
    def _remove_overlaps(self, cuts: List[Cut]) -> List[Cut]:
        """Remove overlapping cuts, keeping the first one."""
        if not cuts:
            return []
        
        result = [cuts[0]]
        for cut in cuts[1:]:
            if cut.start_time >= result[-1].end_time:
                result.append(cut)
            else:
                # Overlap detected - adjust or skip
                if cut.end_time > result[-1].end_time:
                    cut.start_time = result[-1].end_time
                    if cut.end_time - cut.start_time >= 1.0:
                        result.append(cut)
        
        return result
    
    def _enforce_min_duration(self, cuts: List[Cut], start: float, end: float) -> List[Cut]:
        """Enforce minimum shot duration."""
        result = []
        
        for cut in cuts:
            duration = cut.end_time - cut.start_time
            
            if duration < self.min_shot_s:
                # Try to merge with previous if same camera
                if result and result[-1].camera_id == cut.camera_id:
                    result[-1].end_time = max(result[-1].end_time, cut.end_time)
                    continue
                else:
                    # Extend to minimum duration
                    cut.end_time = min(end, cut.start_time + self.min_shot_s)
            
            result.append(cut)
        
        return result
    
    def _round_and_validate(self, cuts: List[Cut], start: float, end: float) -> List[Dict[str, Any]]:
        """Round to frames and validate timeline."""
        result = []
        last_end = start
        
        for cut in cuts:
            s = round_to_frames(max(last_end, cut.start_time), self.fps)
            e = round_to_frames(cut.end_time, self.fps)
            
            # Ensure at least 1 frame
            if e <= s:
                e = s + (1.0 / self.fps)
            
            # Clamp to bounds
            s = max(start, s)
            e = min(end, e)
            
            if e > s:
                result.append({
                    'start_time': s,
                    'end_time': e,
                    'camera_id': cut.camera_id,
                })
                last_end = e
        
        return result
