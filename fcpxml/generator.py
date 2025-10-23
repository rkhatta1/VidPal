# fcpxml/generator.py (COMPLETE with master audio)
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom
from config import get_settings

logger = logging.getLogger(__name__)


class FCPXMLGenerator:
    """Generate FCPXML files compatible with Final Cut Pro and DaVinci Resolve."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def generate(
        self,
        cuts: List[Dict[str, Any]],
        video_paths: Dict[str, Path],
        output_path: Path,
        episode_id: str,
        master_audio_path: Optional[Path] = None,
    ) -> None:
        """
        Generate FCPXML file from EDL cuts.
        
        Args:
            cuts: List of cut dictionaries with start_time, end_time, camera_id
            video_paths: Dictionary of camera_id -> video file path
            output_path: Output path for FCPXML file
            episode_id: Episode identifier
            master_audio_path: Optional path to master audio file
        """
        logger.info(f"Generating FCPXML with {len(cuts)} cuts...")
        
        fps = self.settings.FRAME_RATE
        
        # Calculate total duration from cuts
        if cuts:
            total_duration = max(cut['end_time'] for cut in cuts)
            total_frames = int(total_duration * fps)
        else:
            total_frames = 7200  # Default 4 minutes at 30fps
        
        # Create XML structure
        fcpxml = ET.Element('fcpxml', version='1.10')
        
        # Resources section
        resources = ET.SubElement(fcpxml, 'resources')
        
        # Format resource
        fmt_id = "r1"
        ET.SubElement(
            resources, 'format',
            id=fmt_id,
            name=f"FFVideoFormat{self.settings.VIDEO_HEIGHT}p{int(fps)}",
            frameDuration=f"100/{int(fps * 100)}s",
            width=str(self.settings.VIDEO_WIDTH),
            height=str(self.settings.VIDEO_HEIGHT),
        )
        
        # Add video assets with media-rep
        asset_ids = {}
        next_asset_id = 2
        
        for camera_id, video_path in video_paths.items():
            asset_id = f"r{next_asset_id}"
            asset_ids[camera_id] = asset_id
            next_asset_id += 1
            
            # Convert to Windows path if needed
            windows_path = self._to_windows_path(video_path)
            
            # Create asset element
            asset = ET.SubElement(
                resources, 'asset',
                id=asset_id,
                name=camera_id,
                hasVideo="1",
                hasAudio="1",
                format=fmt_id,
                duration=f"{total_frames}/{int(fps)}s",
            )
            
            # Add media-rep child element
            ET.SubElement(
                asset, 'media-rep',
                kind="original-media",
                src=f"file:///{windows_path}",
            )
        
        # Add master audio asset if provided
        master_audio_id = None
        if master_audio_path:
            master_audio_id = f"r{next_asset_id}"
            windows_audio_path = self._to_windows_path(master_audio_path)
            
            audio_asset = ET.SubElement(
                resources, 'asset',
                id=master_audio_id,
                name="master_audio",
                hasVideo="0",
                hasAudio="1",
                audioSources="1",
                audioChannels="2",
                duration=f"{total_frames}/{int(fps)}s",
            )
            
            ET.SubElement(
                audio_asset, 'media-rep',
                kind="original-media",
                src=f"file:///{windows_audio_path}",
            )
        
        # Library and event structure
        library = ET.SubElement(fcpxml, 'library')
        event = ET.SubElement(library, 'event', name=f"VidPalAI_{episode_id}")
        project = ET.SubElement(
            event, 'project',
            name=f"{episode_id}_multicam",
        )
        
        sequence = ET.SubElement(
            project, 'sequence',
            format=fmt_id,
            duration=f"{total_frames}/{int(fps)}s",
            tcStart="0s",
            tcFormat="NDF",
        )
        
        spine = ET.SubElement(sequence, 'spine')
        
        # Add master audio as base layer (full duration, no cuts)
        if master_audio_id:
            master_audio_clip = ET.SubElement(
                spine, 'audio',
                name="master_audio",
                ref=master_audio_id,
                offset="0/30s",
                duration=f"{total_frames}/{int(fps)}s",
                start="0/30s",
            )
        
        # Add video clips (without audio - master audio is separate)
        timeline_offset = 0
        for cut in cuts:
            self._add_clip(
                spine,
                cut,
                asset_ids,
                fmt_id,
                fps,
                timeline_offset,
                include_audio=(master_audio_id is None),  # Only include clip audio if no master audio
            )
            # Update timeline offset for next clip
            cut_duration = cut['end_time'] - cut['start_time']
            timeline_offset += cut_duration
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Pretty print XML
        xml_str = minidom.parseString(ET.tostring(fcpxml)).toprettyxml(indent="  ")
        
        # Remove extra blank lines
        xml_lines = [line for line in xml_str.split('\n') if line.strip()]
        xml_str = '\n'.join(xml_lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        
        logger.info(f"✅ FCPXML written to {output_path}")
    
    def _add_clip(
        self,
        spine: ET.Element,
        cut: Dict[str, Any],
        asset_ids: Dict[str, str],
        format_id: str,
        fps: float,
        timeline_offset: float,
        include_audio: bool = False,
    ) -> None:
        """Add a clip to the timeline spine with proper video/audio refs."""
        camera_id = cut['camera_id']
        asset_id = asset_ids.get(camera_id)
        
        if not asset_id:
            logger.warning(f"No asset ID for camera {camera_id}, skipping")
            return
        
        start_time = cut['start_time']
        end_time = cut['end_time']
        duration = end_time - start_time
        
        # Convert to frames
        offset_frames = int(timeline_offset * fps)
        duration_frames = int(duration * fps)
        start_frames = int(start_time * fps)
        
        # Create clip element
        clip = ET.SubElement(
            spine, 'clip',
            name=camera_id,
            offset=f"{offset_frames}/{int(fps)}s",
            duration=f"{duration_frames}/{int(fps)}s",
            format=format_id,
        )
        
        # Add video reference
        ET.SubElement(
            clip, 'video',
            ref=asset_id,
            start=f"{start_frames}/{int(fps)}s",
            duration=f"{duration_frames}/{int(fps)}s",
        )
        
        # Only add audio if no master audio track
        if include_audio:
            ET.SubElement(
                clip, 'audio',
                ref=asset_id,
                start=f"{start_frames}/{int(fps)}s",
                duration=f"{duration_frames}/{int(fps)}s",
            )
    
    def _to_windows_path(self, path: Path) -> str:
        """Convert path to Windows format if needed."""
        if self.settings.WINDOWS_PROJECT_ROOT:
            # Use configured Windows path
            path_str = self.settings.WINDOWS_PROJECT_ROOT.replace('\\', '/')
            filename = path.name
            return f"{path_str}/{filename}"
        
        # Default: convert to forward slashes
        return str(path).replace('\\', '/')
