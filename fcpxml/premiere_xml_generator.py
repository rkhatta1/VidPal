# fcpxml/premiere_xml_generator.py
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom
from config import get_settings #

logger = logging.getLogger(__name__)


class PremiereXMLGenerator:
    """
    Generate Adobe Premiere Pro-compatible XMEML (version 4) files.
    This format is different from FCPXML and is based on frame-based <clipitem>s.
    """
    
    def __init__(self):
        self.settings = get_settings() #
        self.fps = self.settings.FRAME_RATE #
        self.timebase = int(self.fps)
        self.ntsc = "TRUE" if self.fps in (29.97, 59.94) else "FALSE"
        
        # Track which file IDs have been fully defined
        self.defined_file_ids = set()
        self.file_id_map = {}
        self.master_clip_id_map = {}
        self.clip_item_count = 0

    def _get_file_id(self, video_path: Path) -> str:
        """Get or create a file ID for a given path."""
        path_str = str(video_path)
        if path_str not in self.file_id_map:
            self.file_id_map[path_str] = f"file-{len(self.file_id_map) + 1}"
        return self.file_id_map[path_str]

    def _get_master_clip_id(self, video_path: Path) -> str:
        """Get or create a masterclip ID for a given path."""
        path_str = str(video_path)
        if path_str not in self.master_clip_id_map:
            self.master_clip_id_map[path_str] = f"masterclip-{len(self.master_clip_id_map) + 1}"
        return self.master_clip_id_map[path_str]

    def _to_windows_path(self, path: Path) -> str:
        """Convert path to Windows format if needed."""
        # This logic is copied from the original FCPXMLGenerator
        if self.settings.WINDOWS_PROJECT_ROOT: #
            path_str = self.settings.WINDOWS_PROJECT_ROOT.replace('\\', '/') #
            filename = path.name
            return f"file://localhost/{path_str}/{filename}"
        
        # Default: convert to forward slashes and add file://localhost/
        path_str = str(path.resolve()).replace('\\', '/')
        if not path_str.startswith('/'):
            path_str = '/' + path_str
        return f"file://localhost{path_str}"

    def _add_file_definition(
        self,
        file_el: ET.Element,
        file_path: Path,
        total_duration_frames: int,
        is_audio: bool = False
    ):
        """Adds the full <file> definition if not already defined."""
        file_id = file_el.attrib['id']
        if file_id in self.defined_file_ids:
            return  # This file has already been defined

        ET.SubElement(file_el, 'name').text = file_path.name
        ET.SubElement(file_el, 'pathurl').text = self._to_windows_path(file_path) #
        
        rate_el = ET.SubElement(file_el, 'rate')
        ET.SubElement(rate_el, 'timebase').text = str(self.timebase)
        ET.SubElement(rate_el, 'ntsc').text = self.ntsc
        
        # This should ideally be the media's duration, but sequence duration is a safe fallback
        ET.SubElement(file_el, 'duration').text = str(total_duration_frames)
        
        media_el = ET.SubElement(file_el, 'media')
        
        if is_audio:
            audio_el = ET.SubElement(media_el, 'audio')
            sample_el = ET.SubElement(audio_el, 'samplecharacteristics')
            ET.SubElement(sample_el, 'depth').text = "16"
            ET.SubElement(sample_el, 'samplerate').text = "48000" # Common default
            ET.SubElement(audio_el, 'channelcount').text = "2"
        else:
            video_el = ET.SubElement(media_el, 'video')
            sample_el = ET.SubElement(video_el, 'samplecharacteristics')
            rate_el_video = ET.SubElement(sample_el, 'rate')
            ET.SubElement(rate_el_video, 'timebase').text = str(self.timebase)
            ET.SubElement(rate_el_video, 'ntsc').text = self.ntsc
            ET.SubElement(sample_el, 'width').text = str(self.settings.VIDEO_WIDTH) #
            ET.SubElement(sample_el, 'height').text = str(self.settings.VIDEO_HEIGHT) #
            ET.SubElement(sample_el, 'anamorphic').text = "FALSE"
            ET.SubElement(sample_el, 'pixelaspectratio').text = "square"
            ET.SubElement(sample_el, 'fielddominance').text = "none"
        
        self.defined_file_ids.add(file_id)

    def generate(
        self,
        cuts: List[Dict[str, Any]],
        video_paths: Dict[str, Path],
        output_path: Path,
        episode_id: str,
        master_audio_path: Optional[Path] = None,
    ) -> None:
        """
        Generate Premiere Pro XMEML file from EDL cuts.
        """
        logger.info(f"Generating Premiere XMEML with {len(cuts)} cuts...")
        
        # Reset trackers
        self.defined_file_ids = set()
        self.file_id_map = {}
        self.master_clip_id_map = {}
        self.clip_item_count = 0
        
        if not cuts and not master_audio_path:
            logger.warning("No cuts or audio provided. Aborting XML generation.")
            return

        # Calculate total duration in frames
        if cuts:
            total_duration_seconds = max(cut['end_time'] for cut in cuts)
        elif master_audio_path:
            # Fallback: We need a duration. Use config.
            total_duration_seconds = self.settings.PROCESS_DURATION_MINUTES * 60 #
        else:
            total_duration_seconds = 1 # Avoid division by zero
            
        total_duration_frames = int(total_duration_seconds * self.fps)

        # Create root element
        xmeml = ET.Element('xmeml', version='4')
        sequence = ET.SubElement(
            xmeml, 'sequence',
            id="sequence-1",
        )
        ET.SubElement(sequence, 'uuid').text = episode_id
        ET.SubElement(sequence, 'duration').text = str(total_duration_frames)
        
        rate_el = ET.SubElement(sequence, 'rate')
        ET.SubElement(rate_el, 'timebase').text = str(self.timebase)
        ET.SubElement(rate_el, 'ntsc').text = self.ntsc
        
        ET.SubElement(sequence, 'name').text = episode_id
        
        media_el = ET.SubElement(sequence, 'media')
        
        # --- Video Tracks ---
        video_media_el = ET.SubElement(media_el, 'video')
        video_format_el = ET.SubElement(video_media_el, 'format')
        video_sample_el = ET.SubElement(video_format_el, 'samplecharacteristics')
        video_rate_el = ET.SubElement(video_sample_el, 'rate')
        ET.SubElement(video_rate_el, 'timebase').text = str(self.timebase)
        ET.SubElement(video_rate_el, 'ntsc').text = self.ntsc
        ET.SubElement(video_sample_el, 'width').text = str(self.settings.VIDEO_WIDTH) #
        ET.SubElement(video_sample_el, 'height').text = str(self.settings.VIDEO_HEIGHT) #
        ET.SubElement(video_sample_el, 'anamorphic').text = "FALSE"
        ET.SubElement(video_sample_el, 'pixelaspectratio').text = "square"
        ET.SubElement(video_sample_el, 'fielddominance').text = "none"

        video_track_el = ET.SubElement(video_media_el, 'track')
        ET.SubElement(video_track_el, 'enabled').text = "TRUE"
        ET.SubElement(video_track_el, 'locked').text = "FALSE"

        timeline_start_frame = 0
        for cut in cuts:
            self.clip_item_count += 1
            
            video_path = video_paths[cut['camera_id']]
            file_id = self._get_file_id(video_path)
            master_clip_id = self._get_master_clip_id(video_path)
            
            # Calculate frame times
            in_point_frames = int(cut['start_time'] * self.fps)
            out_point_frames = int(cut['end_time'] * self.fps)
            duration_frames = out_point_frames - in_point_frames
            timeline_end_frame = timeline_start_frame + duration_frames

            clipitem = ET.SubElement(
                video_track_el, 'clipitem',
                id=f"clipitem-{self.clip_item_count}"
            )
            ET.SubElement(clipitem, 'masterclipid').text = master_clip_id
            ET.SubElement(clipitem, 'name').text = video_path.name
            ET.SubElement(clipitem, 'enabled').text = "TRUE"
            ET.SubElement(clipitem, 'duration').text = str(total_duration_frames) # Media duration
            
            clip_rate_el = ET.SubElement(clipitem, 'rate')
            ET.SubElement(clip_rate_el, 'timebase').text = str(self.timebase)
            ET.SubElement(clip_rate_el, 'ntsc').text = self.ntsc

            # Timeline position
            ET.SubElement(clipitem, 'start').text = str(timeline_start_frame)
            ET.SubElement(clipitem, 'end').text = str(timeline_end_frame)
            # Media file in/out
            ET.SubElement(clipitem, 'in').text = str(in_point_frames)
            ET.SubElement(clipitem, 'out').text = str(out_point_frames)

            # Add file definition
            file_el = ET.SubElement(clipitem, 'file', id=file_id)
            self._add_file_definition(file_el, video_path, total_duration_frames, is_audio=False)
            
            # Update timeline position
            timeline_start_frame = timeline_end_frame

        # --- Audio Tracks ---
        audio_media_el = ET.SubElement(media_el, 'audio')
        ET.SubElement(audio_media_el, 'numOutputChannels').text = "2"
        audio_format_el = ET.SubElement(audio_media_el, 'format')
        audio_sample_el = ET.SubElement(audio_format_el, 'samplecharacteristics')
        ET.SubElement(audio_sample_el, 'depth').text = "16"
        ET.SubElement(audio_sample_el, 'samplerate').text = "48000"
        
        # Create two tracks for stereo
        audio_track_1 = ET.SubElement(audio_media_el, 'track')
        ET.SubElement(audio_track_1, 'enabled').text = "TRUE"
        ET.SubElement(audio_track_1, 'locked').text = "FALSE"
        ET.SubElement(audio_track_1, 'outputchannelindex').text = "1"
        
        audio_track_2 = ET.SubElement(audio_media_el, 'track')
        ET.SubElement(audio_track_2, 'enabled').text = "TRUE"
        ET.SubElement(audio_track_2, 'locked').text = "FALSE"
        ET.SubElement(audio_track_2, 'outputchannelindex').text = "2"

        if master_audio_path:
            self.clip_item_count += 1
            clip_item_id_a1 = f"clipitem-{self.clip_item_count}"
            self.clip_item_count += 1
            clip_item_id_a2 = f"clipitem-{self.clip_item_count}"
            
            file_id = self._get_file_id(master_audio_path)
            master_clip_id = self._get_master_clip_id(master_audio_path)

            # Add clip to Track 1
            clipitem_a1 = ET.SubElement(audio_track_1, 'clipitem', id=clip_item_id_a1)
            ET.SubElement(clipitem_a1, 'masterclipid').text = master_clip_id
            ET.SubElement(clipitem_a1, 'name').text = master_audio_path.name
            ET.SubElement(clipitem_a1, 'enabled').text = "TRUE"
            ET.SubElement(clipitem_a1, 'duration').text = str(total_duration_frames)
            
            clip_rate_a1 = ET.SubElement(clipitem_a1, 'rate')
            ET.SubElement(clip_rate_a1, 'timebase').text = str(self.timebase)
            ET.SubElement(clip_rate_a1, 'ntsc').text = self.ntsc
            
            ET.SubElement(clipitem_a1, 'start').text = "0"
            ET.SubElement(clipitem_a1, 'end').text = str(total_duration_frames)
            ET.SubElement(clipitem_a1, 'in').text = "0"
            ET.SubElement(clipitem_a1, 'out').text = str(total_duration_frames)
            
            file_el_a1 = ET.SubElement(clipitem_a1, 'file', id=file_id)
            self._add_file_definition(file_el_a1, master_audio_path, total_duration_frames, is_audio=True)
            
            ET.SubElement(clipitem_a1, 'sourcetrack').text = "1"
            
            # Add clip to Track 2
            clipitem_a2 = ET.SubElement(audio_track_2, 'clipitem', id=clip_item_id_a2)
            ET.SubElement(clipitem_a2, 'masterclipid').text = master_clip_id
            ET.SubElement(clipitem_a2, 'name').text = master_audio_path.name
            ET.SubElement(clipitem_a2, 'enabled').text = "TRUE"
            ET.SubElement(clipitem_a2, 'duration').text = str(total_duration_frames)

            clip_rate_a2 = ET.SubElement(clipitem_a2, 'rate')
            ET.SubElement(clip_rate_a2, 'timebase').text = str(self.timebase)
            ET.SubElement(clip_rate_a2, 'ntsc').text = self.ntsc

            ET.SubElement(clipitem_a2, 'start').text = "0"
            ET.SubElement(clipitem_a2, 'end').text = str(total_duration_frames)
            ET.SubElement(clipitem_a2, 'in').text = "0"
            ET.SubElement(clipitem_a2, 'out').text = str(total_duration_frames)
            
            ET.SubElement(clipitem_a2, 'file', id=file_id) # No definition needed here
            
            ET.SubElement(clipitem_a2, 'sourcetrack').text = "2"
            
            # Link the two audio clips
            link1 = ET.SubElement(clipitem_a1, 'link')
            ET.SubElement(link1, 'linkclipref').text = clip_item_id_a1
            ET.SubElement(link1, 'mediatype').text = "audio"
            ET.SubElement(link1, 'trackindex').text = "1"
            ET.SubElement(link1, 'clipindex').text = "1"
            
            link2 = ET.SubElement(clipitem_a1, 'link')
            ET.SubElement(link2, 'linkclipref').text = clip_item_id_a2
            ET.SubElement(link2, 'mediatype').text = "audio"
            ET.SubElement(link2, 'trackindex').text = "2"
            ET.SubElement(link2, 'clipindex').text = "1"

            link3 = ET.SubElement(clipitem_a2, 'link')
            ET.SubElement(link3, 'linkclipref').text = clip_item_id_a1
            ET.SubElement(link3, 'mediatype').text = "audio"
            ET.SubElement(link3, 'trackindex').text = "1"
            ET.SubElement(link3, 'clipindex').text = "1"

            link4 = ET.SubElement(clipitem_a2, 'link')
            ET.SubElement(link4, 'linkclipref').text = clip_item_id_a2
            ET.SubElement(link4, 'mediatype').text = "audio"
            ET.SubElement(link4, 'trackindex').text = "2"
            ET.SubElement(link4, 'clipindex').text = "1"
            
        # --- Timecode ---
        timecode_el = ET.SubElement(sequence, 'timecode')
        timecode_rate_el = ET.SubElement(timecode_el, 'rate')
        ET.SubElement(timecode_rate_el, 'timebase').text = str(self.timebase)
        ET.SubElement(timecode_rate_el, 'ntsc').text = self.ntsc
        ET.SubElement(timecode_el, 'string').text = "00:00:00:00"
        ET.SubElement(timecode_el, 'frame').text = "0"
        ET.SubElement(timecode_el, 'displayformat').text = "NDF" # Non-Drop Frame
        
        # --- Write to file ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        xml_str = minidom.parseString(ET.tostring(xmeml)).toprettyxml(indent="  ")
        
        # Remove extra blank lines
        xml_lines = [line for line in xml_str.split('\n') if line.strip()]
        xml_str = '\n'.join(xml_lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        
        logger.info(f"✅ Premiere XMEML written to {output_path}")
