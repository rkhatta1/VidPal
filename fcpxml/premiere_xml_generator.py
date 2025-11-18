# fcpxml/premiere_xml_generator.py
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom
from config import get_settings

logger = logging.getLogger(__name__)

class PremiereXMLGenerator:
    """
    Generate Adobe Premiere Pro-compatible XMEML (version 4).
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.fps = self.settings.FRAME_RATE
        self.timebase = int(self.fps)
        self.ntsc = "TRUE" if self.fps in (29.97, 59.94) else "FALSE"
        
        self.defined_file_ids = set()
        self.file_id_map = {}
        self.master_clip_id_map = {}
        self.clip_item_count = 0

    def _get_file_id(self, video_path: Path) -> str:
        path_str = str(video_path)
        if path_str not in self.file_id_map:
            self.file_id_map[path_str] = f"file-{len(self.file_id_map) + 1}"
        return self.file_id_map[path_str]

    def _get_master_clip_id(self, video_path: Path) -> str:
        path_str = str(video_path)
        if path_str not in self.master_clip_id_map:
            self.master_clip_id_map[path_str] = f"masterclip-{len(self.master_clip_id_map) + 1}"
        return self.master_clip_id_map[path_str]

    def _to_windows_path(self, path: Path) -> str:
        if self.settings.WINDOWS_PROJECT_ROOT:
            path_str = self.settings.WINDOWS_PROJECT_ROOT.replace('\\', '/')
            filename = path.name
            return f"file://localhost/{path_str}/{filename}"
        
        path_str = str(path.resolve()).replace('\\', '/')
        if not path_str.startswith('/'):
            path_str = '/' + path_str
        return f"file://localhost{path_str}"

    def _add_file_definition(self, file_el, file_path, total_frames, is_audio=False):
        # ... (Same logic as before, just helper method) ...
        file_id = file_el.attrib['id']
        if file_id in self.defined_file_ids:
            return

        ET.SubElement(file_el, 'name').text = file_path.name
        ET.SubElement(file_el, 'pathurl').text = self._to_windows_path(file_path)
        
        rate_el = ET.SubElement(file_el, 'rate')
        ET.SubElement(rate_el, 'timebase').text = str(self.timebase)
        ET.SubElement(rate_el, 'ntsc').text = self.ntsc
        ET.SubElement(file_el, 'duration').text = str(total_frames)
        
        media_el = ET.SubElement(file_el, 'media')
        if is_audio:
            audio_el = ET.SubElement(media_el, 'audio')
            # ... audio sample characteristics ...
            sample_el = ET.SubElement(audio_el, 'samplecharacteristics')
            ET.SubElement(sample_el, 'depth').text = "16"
            ET.SubElement(sample_el, 'samplerate').text = "48000"
            ET.SubElement(audio_el, 'channelcount').text = "2"
        else:
            video_el = ET.SubElement(media_el, 'video')
            sample_el = ET.SubElement(video_el, 'samplecharacteristics')
            ET.SubElement(sample_el, 'rate').extend([
                ET.Element('timebase', text=str(self.timebase)),
                ET.Element('ntsc', text=self.ntsc)
            ])
            ET.SubElement(sample_el, 'width').text = str(self.settings.VIDEO_WIDTH)
            ET.SubElement(sample_el, 'height').text = str(self.settings.VIDEO_HEIGHT)
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
        emotion_clusters: Optional[List[Dict[str, float]]] = None, # NEW
    ) -> None:
        logger.info(f"Generating Premiere XMEML with {len(cuts)} cuts...")
        
        self.defined_file_ids = set()
        self.file_id_map = {}
        self.master_clip_id_map = {}
        self.clip_item_count = 0
        
        # Duration calculation
        total_duration_seconds = 1
        if cuts:
            total_duration_seconds = max(cut['end_time'] for cut in cuts)
        elif master_audio_path:
            total_duration_seconds = self.settings.PROCESS_DURATION_MINUTES * 60
            
        total_duration_frames = int(total_duration_seconds * self.fps)

        # Root
        xmeml = ET.Element('xmeml', version='4')
        sequence = ET.SubElement(xmeml, 'sequence', id="sequence-1")
        ET.SubElement(sequence, 'uuid').text = episode_id
        ET.SubElement(sequence, 'duration').text = str(total_duration_frames)
        ET.SubElement(sequence, 'rate').extend([
            ET.Element('timebase', text=str(self.timebase)),
            ET.Element('ntsc', text=self.ntsc)
        ])
        ET.SubElement(sequence, 'name').text = episode_id
        
        media_el = ET.SubElement(sequence, 'media')
        
        # === VIDEO ===
        video_media_el = ET.SubElement(media_el, 'video')
        # Format (Global for sequence)
        fmt = ET.SubElement(video_media_el, 'format')
        sample = ET.SubElement(fmt, 'samplecharacteristics')
        sample.extend([
            ET.Element('rate', children=[
                ET.Element('timebase', text=str(self.timebase)),
                ET.Element('ntsc', text=self.ntsc)
            ]),
            ET.Element('width', text=str(self.settings.VIDEO_WIDTH)),
            ET.Element('height', text=str(self.settings.VIDEO_HEIGHT)),
            ET.Element('anamorphic', text="FALSE"),
            ET.Element('pixelaspectratio', text="square"),
            ET.Element('fielddominance', text="none"),
            ET.Element('colordepth', text="24")
        ])

        # --- Track 1: MAIN CUTS ---
        video_track_1 = ET.SubElement(video_media_el, 'track')
        ET.SubElement(video_track_1, 'enabled').text = "TRUE"
        ET.SubElement(video_track_1, 'locked').text = "FALSE"

        timeline_start_frame = 0
        for cut in cuts:
            self.clip_item_count += 1
            video_path = video_paths[cut['camera_id']]
            file_id = self._get_file_id(video_path)
            master_clip_id = self._get_master_clip_id(video_path)
            
            in_frames = int(cut['start_time'] * self.fps)
            out_frames = int(cut['end_time'] * self.fps)
            duration = out_frames - in_frames
            end_frame = timeline_start_frame + duration

            clip = ET.SubElement(video_track_1, 'clipitem', id=f"clipitem-{self.clip_item_count}")
            ET.SubElement(clip, 'masterclipid').text = master_clip_id
            ET.SubElement(clip, 'name').text = video_path.name
            ET.SubElement(clip, 'enabled').text = "TRUE"
            ET.SubElement(clip, 'duration').text = str(total_duration_frames)
            ET.SubElement(clip, 'rate').extend([
                ET.Element('timebase', text=str(self.timebase)),
                ET.Element('ntsc', text=self.ntsc)
            ])
            ET.SubElement(clip, 'start').text = str(timeline_start_frame)
            ET.SubElement(clip, 'end').text = str(end_frame)
            ET.SubElement(clip, 'in').text = str(in_frames)
            ET.SubElement(clip, 'out').text = str(out_frames)
            
            file_ref = ET.SubElement(clip, 'file', id=file_id)
            self._add_file_definition(file_ref, video_path, total_duration_frames, is_audio=False)
            
            timeline_start_frame = end_frame

        # --- Track 2: EMOTION CLUSTERS (Adjustment Layers) ---
        if emotion_clusters:
            video_track_2 = ET.SubElement(video_media_el, 'track')
            ET.SubElement(video_track_2, 'enabled').text = "TRUE"
            ET.SubElement(video_track_2, 'locked').text = "FALSE"
            
            slug_file_id = "file-slug-emotion"
            slug_master_id = "masterclip-slug-emotion"

            for cluster in emotion_clusters:
                self.clip_item_count += 1
                start_f = int(cluster['start'] * self.fps)
                end_f = int(cluster['end'] * self.fps)
                dur_f = end_f - start_f
                
                if dur_f <= 0: continue
                
                clip = ET.SubElement(video_track_2, 'clipitem', id=f"clipitem-{self.clip_item_count}")
                ET.SubElement(clip, 'masterclipid').text = slug_master_id
                ET.SubElement(clip, 'name').text = "emotion_event" # From sample
                ET.SubElement(clip, 'enabled').text = "TRUE"
                ET.SubElement(clip, 'duration').text = str(total_duration_frames) # Slug duration arbitrary
                
                ET.SubElement(clip, 'rate').extend([
                    ET.Element('timebase', text=str(self.timebase)),
                    ET.Element('ntsc', text=self.ntsc)
                ])
                
                ET.SubElement(clip, 'start').text = str(start_f)
                ET.SubElement(clip, 'end').text = str(end_f)
                ET.SubElement(clip, 'in').text = "0"
                ET.SubElement(clip, 'out').text = str(dur_f)
                
                # Slug File Definition
                file_ref = ET.SubElement(clip, 'file', id=slug_file_id)
                
                if slug_file_id not in self.defined_file_ids:
                    ET.SubElement(file_ref, 'name').text = "Black Video" # Standard slug name
                    ET.SubElement(file_ref, 'mediaSource').text = "Slug" # CRITICAL for AL behavior
                    ET.SubElement(file_ref, 'rate').extend([
                        ET.Element('timebase', text=str(self.timebase)),
                        ET.Element('ntsc', text=self.ntsc)
                    ])
                    # Standard timecode for slugs
                    tc = ET.SubElement(file_ref, 'timecode')
                    ET.SubElement(tc, 'rate').extend([
                        ET.Element('timebase', text=str(self.timebase)),
                        ET.Element('ntsc', text=self.ntsc)
                    ])
                    ET.SubElement(tc, 'string').text = "00:00:00:00"
                    ET.SubElement(tc, 'frame').text = "0"
                    ET.SubElement(tc, 'displayformat').text = "DF" # Drop Frame often standard for slugs
                    
                    # Slug Media Definition
                    m = ET.SubElement(file_ref, 'media')
                    v = ET.SubElement(m, 'video')
                    s = ET.SubElement(v, 'samplecharacteristics')
                    ET.SubElement(s, 'rate').extend([
                        ET.Element('timebase', text=str(self.timebase)),
                        ET.Element('ntsc', text=self.ntsc)
                    ])
                    ET.SubElement(s, 'width').text = str(self.settings.VIDEO_WIDTH)
                    ET.SubElement(s, 'height').text = str(self.settings.VIDEO_HEIGHT)
                    ET.SubElement(s, 'anamorphic').text = "FALSE"
                    ET.SubElement(s, 'pixelaspectratio').text = "square"
                    ET.SubElement(s, 'fielddominance').text = "none"
                    
                    self.defined_file_ids.add(slug_file_id)
                
                # Add Label (Teal)
                labels = ET.SubElement(clip, 'labels')
                ET.SubElement(labels, 'label2').text = "Teal"


        # === AUDIO ===
        # ... (Standard Audio Logic from previous implementation) ...
        audio_media_el = ET.SubElement(media_el, 'audio')
        ET.SubElement(audio_media_el, 'numOutputChannels').text = "2"
        fmt = ET.SubElement(audio_media_el, 'format')
        s = ET.SubElement(fmt, 'samplecharacteristics')
        ET.SubElement(s, 'depth').text = "16"
        ET.SubElement(s, 'samplerate').text = "48000"
        
        # Audio Tracks 1 & 2
        for i in range(1, 3):
            t = ET.SubElement(audio_media_el, 'track')
            ET.SubElement(t, 'enabled').text = "TRUE"
            ET.SubElement(t, 'locked').text = "FALSE"
            ET.SubElement(t, 'outputchannelindex').text = str(i)
            
            # If master audio, add clips
            if master_audio_path:
                self.clip_item_count += 1
                clip_id = f"clipitem-{self.clip_item_count}"
                
                # For linking, we need to know the ID of the other channel's clip.
                # Since we iterate 1 then 2, we can calculate the IDs.
                # If i=1, next is +1. If i=2, prev is -1.
                
                clip = ET.SubElement(t, 'clipitem', id=clip_id, premiereChannelType="mono")
                master_clip_id = self._get_master_clip_id(master_audio_path)
                file_id = self._get_file_id(master_audio_path)
                
                ET.SubElement(clip, 'masterclipid').text = master_clip_id
                ET.SubElement(clip, 'name').text = master_audio_path.name
                ET.SubElement(clip, 'enabled').text = "TRUE"
                ET.SubElement(clip, 'duration').text = str(total_duration_frames)
                ET.SubElement(clip, 'rate').extend([
                    ET.Element('timebase', text=str(self.timebase)),
                    ET.Element('ntsc', text=self.ntsc)
                ])
                ET.SubElement(clip, 'start').text = "0"
                ET.SubElement(clip, 'end').text = str(total_duration_frames)
                ET.SubElement(clip, 'in').text = "0"
                ET.SubElement(clip, 'out').text = str(total_duration_frames)
                
                f = ET.SubElement(clip, 'file', id=file_id)
                self._add_file_definition(f, master_audio_path, total_duration_frames, is_audio=True)
                
                ET.SubElement(clip, 'sourcetrack').text = "1" # Source is stereo, but treated as mono tracks
                
                # Link Logic (Simplified for stereo pair)
                # In a robust system, we'd track IDs. For now, assuming uniform creation order:
                # Track 1 clip is ID X. Track 2 clip is ID X+1.
                
        # Write File
        output_path.parent.mkdir(parents=True, exist_ok=True)
        xml_str = minidom.parseString(ET.tostring(xmeml)).toprettyxml(indent="  ")
        xml_lines = [line for line in xml_str.split('\n') if line.strip()]
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(xml_lines))
        
        logger.info(f"✅ Premiere XMEML written to {output_path}")
