
# fcpxml/generator.py
import logging
from pathlib import Path
from typing import Dict, List, Any
import xml.etree.ElementTree as ET
from xml.dom import minidom
from config import get_settings

logger = logging.getLogger(__name__)


class FCPXMLGenerator:
    """Generate FCPXML files for Final Cut Pro and DaVinci Resolve."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def generate(
        self,
        cuts: List[Dict[str, Any]],
        video_paths: Dict[str, Path],
        output_path: Path,
        episode_id: str,
    ) -> None:
        """
        Generate FCPXML file from EDL cuts.
        
        Args:
            cuts: List of cut dictionaries
            video_paths: Dictionary of camera_id -> video file path
            output_path: Output path for FCPXML file
            episode_id: Episode identifier
        """
        logger.info(f"Generating FCPXML with {len(cuts)} cuts...")
        
        # Create XML structure
        fcpxml = ET.Element('fcpxml', version='1.10')
        
        # Resources section
        resources = ET.SubElement(fcpxml, 'resources')
        fmt_id = "r1"
        ET.SubElement(
            resources, 'format',
            id=fmt_id,
            name=f"FFVideoFormat{self.settings.VIDEO_HEIGHT}p{int(self.settings.FRAME_RATE)}",
            frameDuration=f"{100}/{int(self.settings.FRAME_RATE * 100)}s",
            width=str(self.settings.VIDEO_WIDTH),
            height=str(self.settings.VIDEO_HEIGHT),
        )
        
        # Add video assets
        asset_ids = {}
        for i, (camera_id, video_path) in enumerate(video_paths.items(), start=2):
            asset_id = f"r{i}"
            asset_ids[camera_id] = asset_id
            
            # Convert to Windows path if needed
            windows_path = self._to_windows_path(video_path)
            
            ET.SubElement(
                resources, 'asset',
                id=asset_id,
                name=camera_id,
                uid=f"{episode_id}_{camera_id}",
                src=f"file://{windows_path}",
                hasVideo="1",
                hasAudio="1",
                format=fmt_id,
            )
        
        # Library and event structure
        library = ET.SubElement(fcpxml, 'library')
        event = ET.SubElement(library, 'event', name=f"VidPalAI_{episode_id}")
        project = ET.SubElement(
            event, 'project',
            name=f"{episode_id}_multicam",
            uid=f"{episode_id}_project",
        )
        
        sequence = ET.SubElement(
            project, 'sequence',
            format=fmt_id,
            tcStart="0s",
            tcFormat="NDF",
            audioLayout="stereo",
            audioRate="48k",
        )
        
        spine = ET.SubElement(sequence, 'spine')
        
        # Add clips
        for cut in cuts:
            self._add_clip(
                spine,
                cut,
                asset_ids,
                fmt_id,
            )
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Pretty print XML
        xml_str = minidom.parseString(ET.tostring(fcpxml)).toprettyxml(indent="  ")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        
        logger.info(f"✅ FCPXML written to {output_path}")
    
    def _add_clip(
        self,
        spine: ET.Element,
        cut: Dict[str, Any],
        asset_ids: Dict[str, str],
        format_id: str,
    ) -> None:
        """Add a clip to the timeline spine."""
        camera_id = cut['camera_id']
        asset_id = asset_ids.get(camera_id)
        
        if not asset_id:
            logger.warning(f"No asset ID for camera {camera_id}, skipping")
            return
        
        start_time = cut['start_time']
        end_time = cut['end_time']
        duration = end_time - start_time
        
        # Convert to rational time (seconds * frame_rate / frame_rate)
        fps = self.settings.FRAME_RATE
        start_frames = int(start_time * fps)
        duration_frames = int(duration * fps)
        
        # Create clip element
        clip = ET.SubElement(
            spine, 'asset-clip',
            name=f"{camera_id}_{start_time:.2f}",
            ref=asset_id,
            offset=f"{start_frames}/{int(fps)}s",
            duration=f"{duration_frames}/{int(fps)}s",
            start=f"{start_frames}/{int(fps)}s",
            format=format_id,
        )
    
    def _to_windows_path(self, path: Path) -> str:
        """Convert path to Windows format if needed."""
        if self.settings.WINDOWS_PROJECT_ROOT:
            # Replace project root with Windows root
            path_str = str(path).replace('\\', '/')
            return f"{self.settings.WINDOWS_PROJECT_ROOT}/{path.name}"
        return str(path).replace('\\', '/')
