# fcpxml/premiere_xml_generator.py
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import xml.etree.ElementTree as ET
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

    def _create_text_elem(self, tag: str, text: str) -> ET.Element:
        """Helper to create an element with text content."""
        e = ET.Element(tag)
        e.text = str(text)
        return e

    def _add_file_definition(self, file_el, file_path, total_frames, is_audio=False):
        file_id = file_el.attrib['id']
        if file_id in self.defined_file_ids:
            return

        file_el.append(self._create_text_elem('name', file_path.name))
        file_el.append(self._create_text_elem('pathurl', self._to_windows_path(file_path)))
        
        rate_el = ET.SubElement(file_el, 'rate')
        rate_el.append(self._create_text_elem('timebase', self.timebase))
        rate_el.append(self._create_text_elem('ntsc', self.ntsc))
        
        file_el.append(self._create_text_elem('duration', total_frames))
        
        media_el = ET.SubElement(file_el, 'media')
        if is_audio:
            audio_el = ET.SubElement(media_el, 'audio')
            sample_el = ET.SubElement(audio_el, 'samplecharacteristics')
            sample_el.append(self._create_text_elem('depth', "16"))
            sample_el.append(self._create_text_elem('samplerate', "48000"))
            audio_el.append(self._create_text_elem('channelcount', "2"))
        else:
            video_el = ET.SubElement(media_el, 'video')
            sample_el = ET.SubElement(video_el, 'samplecharacteristics')
            
            r = ET.SubElement(sample_el, 'rate')
            r.append(self._create_text_elem('timebase', self.timebase))
            r.append(self._create_text_elem('ntsc', self.ntsc))
            
            sample_el.append(self._create_text_elem('width', self.settings.VIDEO_WIDTH))
            sample_el.append(self._create_text_elem('height', self.settings.VIDEO_HEIGHT))
            sample_el.append(self._create_text_elem('anamorphic', "FALSE"))
            sample_el.append(self._create_text_elem('pixelaspectratio', "square"))
            sample_el.append(self._create_text_elem('fielddominance', "none"))
        
        self.defined_file_ids.add(file_id)

    def generate(
        self,
        cuts: List[Dict[str, Any]],
        video_paths: Dict[str, Path],
        output_path: Path,
        episode_id: str,
        master_audio_path: Optional[Path] = None,
        emotion_clusters: Optional[List[Dict[str, float]]] = None,
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
        
        sequence.append(self._create_text_elem('uuid', episode_id))
        sequence.append(self._create_text_elem('duration', total_duration_frames))
        
        seq_rate = ET.SubElement(sequence, 'rate')
        seq_rate.append(self._create_text_elem('timebase', self.timebase))
        seq_rate.append(self._create_text_elem('ntsc', self.ntsc))
        
        sequence.append(self._create_text_elem('name', episode_id))
        
        media_el = ET.SubElement(sequence, 'media')
        
        # === VIDEO ===
        video_media_el = ET.SubElement(media_el, 'video')
        
        # Format (Global for sequence)
        fmt = ET.SubElement(video_media_el, 'format')
        sample = ET.SubElement(fmt, 'samplecharacteristics')
        
        fmt_rate = ET.SubElement(sample, 'rate')
        fmt_rate.append(self._create_text_elem('timebase', self.timebase))
        fmt_rate.append(self._create_text_elem('ntsc', self.ntsc))
        
        sample.append(self._create_text_elem('width', self.settings.VIDEO_WIDTH))
        sample.append(self._create_text_elem('height', self.settings.VIDEO_HEIGHT))
        sample.append(self._create_text_elem('anamorphic', "FALSE"))
        sample.append(self._create_text_elem('pixelaspectratio', "square"))
        sample.append(self._create_text_elem('fielddominance', "none"))
        sample.append(self._create_text_elem('colordepth', "24"))

        # --- Track 1: MAIN CUTS ---
        video_track_1 = ET.SubElement(video_media_el, 'track')
        video_track_1.append(self._create_text_elem('enabled', "TRUE"))
        video_track_1.append(self._create_text_elem('locked', "FALSE"))

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
            clip.append(self._create_text_elem('masterclipid', master_clip_id))
            clip.append(self._create_text_elem('name', video_path.name))
            clip.append(self._create_text_elem('enabled', "TRUE"))
            clip.append(self._create_text_elem('duration', total_duration_frames))
            
            c_rate = ET.SubElement(clip, 'rate')
            c_rate.append(self._create_text_elem('timebase', self.timebase))
            c_rate.append(self._create_text_elem('ntsc', self.ntsc))
            
            clip.append(self._create_text_elem('start', timeline_start_frame))
            clip.append(self._create_text_elem('end', end_frame))
            clip.append(self._create_text_elem('in', in_frames))
            clip.append(self._create_text_elem('out', out_frames))
            
            file_ref = ET.SubElement(clip, 'file', id=file_id)
            self._add_file_definition(file_ref, video_path, total_duration_frames, is_audio=False)
            
            timeline_start_frame = end_frame

        # --- Track 2: EMOTION CLUSTERS (Adjustment Layers) ---
        if emotion_clusters:
            video_track_2 = ET.SubElement(video_media_el, 'track')
            video_track_2.append(self._create_text_elem('enabled', "TRUE"))
            video_track_2.append(self._create_text_elem('locked', "FALSE"))
            
            slug_file_id = "file-slug-emotion"
            slug_master_id = "masterclip-slug-emotion"

            for cluster in emotion_clusters:
                self.clip_item_count += 1
                start_f = int(cluster['start'] * self.fps)
                end_f = int(cluster['end'] * self.fps)
                dur_f = end_f - start_f
                
                if dur_f <= 0: continue
                
                clip = ET.SubElement(video_track_2, 'clipitem', id=f"clipitem-{self.clip_item_count}")
                clip.append(self._create_text_elem('masterclipid', slug_master_id))
                clip.append(self._create_text_elem('name', "emotion_event"))
                clip.append(self._create_text_elem('enabled', "TRUE"))
                clip.append(self._create_text_elem('duration', total_duration_frames))
                
                c_rate = ET.SubElement(clip, 'rate')
                c_rate.append(self._create_text_elem('timebase', self.timebase))
                c_rate.append(self._create_text_elem('ntsc', self.ntsc))
                
                clip.append(self._create_text_elem('start', start_f))
                clip.append(self._create_text_elem('end', end_f))
                clip.append(self._create_text_elem('in', "0"))
                clip.append(self._create_text_elem('out', dur_f))
                
                # Slug File Definition
                file_ref = ET.SubElement(clip, 'file', id=slug_file_id)
                
                if slug_file_id not in self.defined_file_ids:
                    file_ref.append(self._create_text_elem('name', "Black Video"))
                    file_ref.append(self._create_text_elem('mediaSource', "Slug"))
                    
                    fr = ET.SubElement(file_ref, 'rate')
                    fr.append(self._create_text_elem('timebase', self.timebase))
                    fr.append(self._create_text_elem('ntsc', self.ntsc))
                    
                    file_ref.append(self._create_text_elem('duration', total_duration_frames))
                    
                    # Timecode
                    tc = ET.SubElement(file_ref, 'timecode')
                    tr = ET.SubElement(tc, 'rate')
                    tr.append(self._create_text_elem('timebase', self.timebase))
                    tr.append(self._create_text_elem('ntsc', self.ntsc))
                    tc.append(self._create_text_elem('string', "00:00:00:00"))
                    tc.append(self._create_text_elem('frame', "0"))
                    tc.append(self._create_text_elem('displayformat', "DF"))
                    
                    # Media Def
                    m = ET.SubElement(file_ref, 'media')
                    v = ET.SubElement(m, 'video')
                    s = ET.SubElement(v, 'samplecharacteristics')
                    
                    sr = ET.SubElement(s, 'rate')
                    sr.append(self._create_text_elem('timebase', self.timebase))
                    sr.append(self._create_text_elem('ntsc', self.ntsc))
                    
                    s.append(self._create_text_elem('width', self.settings.VIDEO_WIDTH))
                    s.append(self._create_text_elem('height', self.settings.VIDEO_HEIGHT))
                    s.append(self._create_text_elem('anamorphic', "FALSE"))
                    s.append(self._create_text_elem('pixelaspectratio', "square"))
                    s.append(self._create_text_elem('fielddominance', "none"))
                    
                    self.defined_file_ids.add(slug_file_id)
                
                # Add Filter: Basic Motion (Scale 0)
                filt = ET.SubElement(clip, 'filter')
                eff = ET.SubElement(filt, 'effect')
                eff.append(self._create_text_elem('name', 'Basic Motion'))
                eff.append(self._create_text_elem('effectid', 'basic'))
                eff.append(self._create_text_elem('effectcategory', 'motion'))
                eff.append(self._create_text_elem('effecttype', 'motion'))
                eff.append(self._create_text_elem('mediatype', 'video'))
                eff.append(self._create_text_elem('pproBypass', 'false'))
                
                # Scale
                p_scale = ET.SubElement(eff, 'parameter', authoringApp='PremierePro')
                p_scale.append(self._create_text_elem('parameterid', 'scale'))
                p_scale.append(self._create_text_elem('name', 'Scale'))
                p_scale.append(self._create_text_elem('valuemin', '0'))
                p_scale.append(self._create_text_elem('valuemax', '1000'))
                p_scale.append(self._create_text_elem('value', '0')) # Scale 0
                
                # Rotation (Required defaults)
                p_rot = ET.SubElement(eff, 'parameter', authoringApp='PremierePro')
                p_rot.append(self._create_text_elem('parameterid', 'rotation'))
                p_rot.append(self._create_text_elem('name', 'Rotation'))
                p_rot.append(self._create_text_elem('valuemin', '-8640'))
                p_rot.append(self._create_text_elem('valuemax', '8640'))
                p_rot.append(self._create_text_elem('value', '0'))
                
                # Center (Required defaults)
                p_cen = ET.SubElement(eff, 'parameter', authoringApp='PremierePro')
                p_cen.append(self._create_text_elem('parameterid', 'center'))
                p_cen.append(self._create_text_elem('name', 'Center'))
                v_cen = ET.SubElement(p_cen, 'value')
                v_cen.append(self._create_text_elem('horiz', '0'))
                v_cen.append(self._create_text_elem('vert', '0'))

                # Anchor Point (Required defaults)
                p_anc = ET.SubElement(eff, 'parameter', authoringApp='PremierePro')
                p_anc.append(self._create_text_elem('parameterid', 'centerOffset'))
                p_anc.append(self._create_text_elem('name', 'Anchor Point'))
                v_anc = ET.SubElement(p_anc, 'value')
                v_anc.append(self._create_text_elem('horiz', '0'))
                v_anc.append(self._create_text_elem('vert', '0'))
                
                # Add Label
                labels = ET.SubElement(clip, 'labels')
                labels.append(self._create_text_elem('label2', "Teal"))

        # === AUDIO ===
        audio_media_el = ET.SubElement(media_el, 'audio')
        audio_media_el.append(self._create_text_elem('numOutputChannels', "2"))
        fmt = ET.SubElement(audio_media_el, 'format')
        s = ET.SubElement(fmt, 'samplecharacteristics')
        s.append(self._create_text_elem('depth', "16"))
        s.append(self._create_text_elem('samplerate', "48000"))
        
        # Audio Tracks 1 & 2
        for i in range(1, 3):
            t = ET.SubElement(audio_media_el, 'track')
            t.append(self._create_text_elem('enabled', "TRUE"))
            t.append(self._create_text_elem('locked', "FALSE"))
            t.append(self._create_text_elem('outputchannelindex', str(i)))
            
            if master_audio_path:
                self.clip_item_count += 1
                clip_id = f"clipitem-{self.clip_item_count}"
                
                clip = ET.SubElement(t, 'clipitem', id=clip_id, premiereChannelType="mono")
                master_clip_id = self._get_master_clip_id(master_audio_path)
                file_id = self._get_file_id(master_audio_path)
                
                clip.append(self._create_text_elem('masterclipid', master_clip_id))
                clip.append(self._create_text_elem('name', master_audio_path.name))
                clip.append(self._create_text_elem('enabled', "TRUE"))
                clip.append(self._create_text_elem('duration', total_duration_frames))
                
                cr = ET.SubElement(clip, 'rate')
                cr.append(self._create_text_elem('timebase', self.timebase))
                cr.append(self._create_text_elem('ntsc', self.ntsc))
                
                clip.append(self._create_text_elem('start', "0"))
                clip.append(self._create_text_elem('end', total_duration_frames))
                clip.append(self._create_text_elem('in', "0"))
                clip.append(self._create_text_elem('out', total_duration_frames))
                
                f = ET.SubElement(clip, 'file', id=file_id)
                self._add_file_definition(f, master_audio_path, total_duration_frames, is_audio=True)
                
                clip.append(self._create_text_elem('sourcetrack', "1"))

        # --- Safe Write using ET.indent (Python 3.9+) ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        ET.indent(xmeml, space="  ", level=0)
        
        tree = ET.ElementTree(xmeml)
        
        with open(output_path, 'wb') as f:
            f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(b'<!DOCTYPE xmeml>\n')
            tree.write(f, encoding='utf-8', xml_declaration=False)
        
        logger.info(f"✅ Premiere XMEML written to {output_path}")
