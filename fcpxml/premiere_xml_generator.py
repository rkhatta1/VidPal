import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import xml.etree.ElementTree as ET

from config import get_settings
from processing.sync.models import SyncResult

logger = logging.getLogger(__name__)


class PremiereXMLGenerator:
    """
    Generate Adobe Premiere Pro-compatible XMEML (version 4).

    Layout:
      - V1..Vn  : one full-length track per camera
      - V(n+1)  : main cut sequence
      - V(n+2)  : emotion overlays (slug/adjustment layer)

    Audio:
      - If master_audio_path is provided  -> use that as separate master audio.
      - If master_audio_path is None     -> reuse master video file's audio
                                           (or cam_wide / first camera).
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.fps = self.settings.FRAME_RATE
        self.timebase = int(self.fps)
        self.ntsc = "TRUE" if self.fps in (29.97, 59.94) else "FALSE"
        
        self.defined_file_ids: set[str] = set()
        self.file_id_map: Dict[str, str] = {}
        self.master_clip_id_map: Dict[str, str] = {}
        self.clip_item_count: int = 0

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

    def _create_text_elem(self, tag: str, text: Any) -> ET.Element:
        """Helper to create an element with text content."""
        e = ET.Element(tag)
        e.text = str(text)
        return e

    def _add_file_definition(
        self,
        file_el: ET.Element,
        file_path: Path,
        total_frames: int,
        is_audio: bool = False,
    ) -> None:
        """
        Define a <file> element once per file_id.

        For AV (video) files (is_audio=False), define BOTH video and audio
        characteristics so the same file id can be used on video and audio tracks.
        For pure audio files (is_audio=True), define only audio.
        """
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

        # Video characteristics for AV files
        if not is_audio:
            video_el = ET.SubElement(media_el, 'video')
            sample_el_v = ET.SubElement(video_el, 'samplecharacteristics')
            
            r = ET.SubElement(sample_el_v, 'rate')
            r.append(self._create_text_elem('timebase', self.timebase))
            r.append(self._create_text_elem('ntsc', self.ntsc))
            
            sample_el_v.append(self._create_text_elem('width', self.settings.VIDEO_WIDTH))
            sample_el_v.append(self._create_text_elem('height', self.settings.VIDEO_HEIGHT))
            sample_el_v.append(self._create_text_elem('anamorphic', "FALSE"))
            sample_el_v.append(self._create_text_elem('pixelaspectratio', "square"))
            sample_el_v.append(self._create_text_elem('fielddominance', "none"))
        
        # Audio characteristics (for both AV and audio-only files)
        audio_el = ET.SubElement(media_el, 'audio')
        sample_el_a = ET.SubElement(audio_el, 'samplecharacteristics')
        sample_el_a.append(self._create_text_elem('depth', "16"))
        sample_el_a.append(self._create_text_elem('samplerate', "48000"))
        audio_el.append(self._create_text_elem('channelcount', "2"))
        
        self.defined_file_ids.add(file_id)

    def generate(
        self,
        cuts: List[Dict[str, Any]],
        video_paths: Dict[str, Path],
        output_path: Path,
        episode_id: str,
        master_audio_path: Optional[Path] = None,
        emotion_clusters: Optional[List[Dict[str, float]]] = None,
        sync_result: Optional[SyncResult] = None,
    ) -> None:
        logger.info(f"Generating Premiere XMEML with {len(cuts)} cuts...")
        
        # Reset internal state
        self.defined_file_ids = set()
        self.file_id_map = {}
        self.master_clip_id_map = {}
        self.clip_item_count = 0

        camera_ids = list(video_paths.keys())
        
        # --- Determine timeline bounds from cuts ---
        if cuts:
            min_start = min(c['start_time'] for c in cuts)
            max_end = max(c['end_time'] for c in cuts)
            if max_end < min_start:
                max_end = min_start
            total_duration_seconds = max(max_end - min_start, 1.0 / self.fps)
        else:
            min_start = 0.0
            if master_audio_path is not None:
                total_duration_seconds = self.settings.PROCESS_DURATION_MINUTES * 60
            else:
                total_duration_seconds = 1.0
            max_end = min_start + total_duration_seconds
        
        total_duration_frames = max(1, int(round(total_duration_seconds * self.fps)))

        # --- Root / sequence ---
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
        
        # Global sequence video format
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

        # --- Tracks 1..N: full-length camera layers ---
        for camera_id in camera_ids:
            track = ET.SubElement(video_media_el, 'track')
            track.append(self._create_text_elem('enabled', "TRUE"))
            track.append(self._create_text_elem('locked', "FALSE"))
            
            video_path = video_paths[camera_id]
            file_id = self._get_file_id(video_path)
            master_clip_id = self._get_master_clip_id(video_path)

            # Target global range we care about
            if cuts:
                target_global_start = min_start
                target_global_end = max_end
            else:
                target_global_start = 0.0
                target_global_end = max_end

            offset = sync_result.get_offset(camera_id) if sync_result else None

            if offset:
                global_start = max(target_global_start, offset.global_in_point)
                global_end = min(target_global_end, offset.global_out_point)
            else:
                global_start = target_global_start
                global_end = target_global_end

            if global_end <= global_start:
                logger.warning(f"{camera_id}: no overlap with main timeline; skipping base track.")
                continue

            clip_global_duration = global_end - global_start

            # Timeline placement (rebased so min_start -> 0)
            if cuts:
                timeline_start_sec = global_start - min_start
            else:
                timeline_start_sec = global_start - target_global_start  # usually 0

            timeline_end_sec = timeline_start_sec + clip_global_duration

            start_frame = int(round(timeline_start_sec * self.fps))
            end_frame = int(round(timeline_end_sec * self.fps))
            if end_frame <= start_frame:
                end_frame = start_frame + 1

            # Source trimming
            if offset:
                source_in_sec = global_start - offset.global_in_point
            else:
                source_in_sec = global_start

            if source_in_sec < 0:
                logger.warning(
                    f"{camera_id}: negative source_in_sec {source_in_sec:.3f}s "
                    f"on base track; clamping to 0."
                )
                source_in_sec = 0.0

            in_frames = int(round(source_in_sec * self.fps))
            clip_duration_frames = end_frame - start_frame
            out_frames = in_frames + clip_duration_frames

            self.clip_item_count += 1
            clip = ET.SubElement(track, 'clipitem', id=f"clipitem-{self.clip_item_count}")
            clip.append(self._create_text_elem('masterclipid', master_clip_id))
            clip.append(self._create_text_elem('name', video_path.name))
            clip.append(self._create_text_elem('enabled', "TRUE"))
            clip.append(self._create_text_elem('duration', total_duration_frames))
            
            c_rate = ET.SubElement(clip, 'rate')
            c_rate.append(self._create_text_elem('timebase', self.timebase))
            c_rate.append(self._create_text_elem('ntsc', self.ntsc))
            
            clip.append(self._create_text_elem('start', start_frame))
            clip.append(self._create_text_elem('end', end_frame))
            clip.append(self._create_text_elem('in', in_frames))
            clip.append(self._create_text_elem('out', out_frames))
            
            file_ref = ET.SubElement(clip, 'file', id=file_id)
            self._add_file_definition(file_ref, video_path, total_duration_frames, is_audio=False)

        # --- Track N+1: MAIN CUTS ---
        cuts_track = ET.SubElement(video_media_el, 'track')
        cuts_track.append(self._create_text_elem('enabled', "TRUE"))
        cuts_track.append(self._create_text_elem('locked', "FALSE"))

        if cuts:
            sorted_cuts = sorted(cuts, key=lambda c: c['start_time'])

            for cut in sorted_cuts:
                camera_id = cut['camera_id']
                if camera_id not in video_paths:
                    logger.warning(f"Cut references unknown camera '{camera_id}', skipping")
                    continue

                video_path = video_paths[camera_id]
                file_id = self._get_file_id(video_path)
                master_clip_id = self._get_master_clip_id(video_path)

                global_start = cut['start_time']
                global_end = cut['end_time']
                if global_end <= global_start:
                    continue

                clip_duration_sec = global_end - global_start

                # Timeline placement
                timeline_start_sec = global_start - min_start
                timeline_end_sec = timeline_start_sec + clip_duration_sec

                start_frame = int(round(timeline_start_sec * self.fps))
                end_frame = int(round(timeline_end_sec * self.fps))
                if end_frame <= start_frame:
                    end_frame = start_frame + 1

                clip_duration_frames = end_frame - start_frame

                # Source trimming via sync
                offset = sync_result.get_offset(camera_id) if sync_result else None
                if offset:
                    source_in_sec = global_start - offset.global_in_point
                else:
                    source_in_sec = global_start

                if source_in_sec < 0:
                    logger.warning(
                        f"{camera_id}: negative source_in_sec {source_in_sec:.3f}s "
                        f"at cut start {global_start:.3f}s; clamping to 0."
                    )
                    source_in_sec = 0.0

                in_frames = int(round(source_in_sec * self.fps))
                out_frames = in_frames + clip_duration_frames

                self.clip_item_count += 1
                clip = ET.SubElement(cuts_track, 'clipitem', id=f"clipitem-{self.clip_item_count}")
                clip.append(self._create_text_elem('masterclipid', master_clip_id))
                clip.append(self._create_text_elem('name', video_path.name))
                clip.append(self._create_text_elem('enabled', "TRUE"))
                clip.append(self._create_text_elem('duration', total_duration_frames))
                
                c_rate = ET.SubElement(clip, 'rate')
                c_rate.append(self._create_text_elem('timebase', self.timebase))
                c_rate.append(self._create_text_elem('ntsc', self.ntsc))
                
                clip.append(self._create_text_elem('start', start_frame))
                clip.append(self._create_text_elem('end', end_frame))
                clip.append(self._create_text_elem('in', in_frames))
                clip.append(self._create_text_elem('out', out_frames))
                
                file_ref = ET.SubElement(clip, 'file', id=file_id)
                self._add_file_definition(file_ref, video_path, total_duration_frames, is_audio=False)

        # --- Track N+2: EMOTION CLUSTERS (Adjustment Layer) ---
        if emotion_clusters:
            emotion_track = ET.SubElement(video_media_el, 'track')
            emotion_track.append(self._create_text_elem('enabled', "TRUE"))
            emotion_track.append(self._create_text_elem('locked', "FALSE"))
            
            slug_file_id = "file-slug-emotion"
            slug_master_id = "masterclip-slug-emotion"

            for cluster in emotion_clusters:
                start_f = int(round((cluster['start'] - min_start) * self.fps))
                end_f = int(round((cluster['end'] - min_start) * self.fps))
                dur_f = end_f - start_f
                if dur_f <= 0:
                    continue
                if start_f < 0:
                    start_f = 0

                self.clip_item_count += 1
                clip = ET.SubElement(emotion_track, 'clipitem', id=f"clipitem-{self.clip_item_count}")
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
                
                file_ref = ET.SubElement(clip, 'file', id=slug_file_id)
                
                if slug_file_id not in self.defined_file_ids:
                    file_ref.append(self._create_text_elem('name', "Black Video"))
                    file_ref.append(self._create_text_elem('mediaSource', "Slug"))
                    
                    fr = ET.SubElement(file_ref, 'rate')
                    fr.append(self._create_text_elem('timebase', self.timebase))
                    fr.append(self._create_text_elem('ntsc', self.ntsc))
                    
                    file_ref.append(self._create_text_elem('duration', total_duration_frames))
                    
                    tc = ET.SubElement(file_ref, 'timecode')
                    tr = ET.SubElement(tc, 'rate')
                    tr.append(self._create_text_elem('timebase', self.timebase))
                    tr.append(self._create_text_elem('ntsc', self.ntsc))
                    tc.append(self._create_text_elem('string', "00:00:00:00"))
                    tc.append(self._create_text_elem('frame', "0"))
                    tc.append(self._create_text_elem('displayformat', "DF"))
                    
                    m = ET.SubElement(file_ref, 'media')
                    v = ET.SubElement(m, 'video')
                    s_char = ET.SubElement(v, 'samplecharacteristics')
                    
                    sr = ET.SubElement(s_char, 'rate')
                    sr.append(self._create_text_elem('timebase', self.timebase))
                    sr.append(self._create_text_elem('ntsc', self.ntsc))
                    
                    s_char.append(self._create_text_elem('width', self.settings.VIDEO_WIDTH))
                    s_char.append(self._create_text_elem('height', self.settings.VIDEO_HEIGHT))
                    s_char.append(self._create_text_elem('anamorphic', "FALSE"))
                    s_char.append(self._create_text_elem('pixelaspectratio', "square"))
                    s_char.append(self._create_text_elem('fielddominance', "none"))
                    
                    self.defined_file_ids.add(slug_file_id)
                
                # Basic Motion hack to make this an "adjustment-layer"-like clip
                filt = ET.SubElement(clip, 'filter')
                eff = ET.SubElement(filt, 'effect')
                eff.append(self._create_text_elem('name', 'Basic Motion'))
                eff.append(self._create_text_elem('effectid', 'basic'))
                eff.append(self._create_text_elem('effectcategory', 'motion'))
                eff.append(self._create_text_elem('effecttype', 'motion'))
                eff.append(self._create_text_elem('mediatype', 'video'))
                eff.append(self._create_text_elem('pproBypass', 'false'))
                
                p_scale = ET.SubElement(eff, 'parameter', authoringApp='PremierePro')
                p_scale.append(self._create_text_elem('parameterid', 'scale'))
                p_scale.append(self._create_text_elem('name', 'Scale'))
                p_scale.append(self._create_text_elem('valuemin', '0'))
                p_scale.append(self._create_text_elem('valuemax', '1000'))
                p_scale.append(self._create_text_elem('value', '0'))

        # === AUDIO ===
        audio_media_el = ET.SubElement(media_el, 'audio')
        audio_media_el.append(self._create_text_elem('numOutputChannels', "2"))
        afmt = ET.SubElement(audio_media_el, 'format')
        s_char_a = ET.SubElement(afmt, 'samplecharacteristics')
        s_char_a.append(self._create_text_elem('depth', "16"))
        s_char_a.append(self._create_text_elem('samplerate', "48000"))
        
        # --- Determine audio source (Option B logic) ---
        audio_source_path: Optional[Path] = None
        audio_file_id: Optional[str] = None
        audio_master_clip_id: Optional[str] = None
        audio_in_frames = 0
        audio_out_frames = total_duration_frames

        if master_audio_path is not None:
            # Separate master audio file exists
            audio_source_path = master_audio_path
            audio_file_id = self._get_file_id(master_audio_path)
            audio_master_clip_id = self._get_master_clip_id(master_audio_path)
        else:
            # Use a video's audio instead
            audio_cam_id: Optional[str] = None
            if sync_result and sync_result.master_file_id in video_paths:
                audio_cam_id = sync_result.master_file_id
            elif "cam_wide" in video_paths:
                audio_cam_id = "cam_wide"
            elif camera_ids:
                audio_cam_id = camera_ids[0]

            if audio_cam_id:
                audio_source_path = video_paths[audio_cam_id]
                audio_file_id = self._get_file_id(audio_source_path)
                audio_master_clip_id = self._get_master_clip_id(audio_source_path)

                if sync_result and cuts:
                    offset = sync_result.get_offset(audio_cam_id)
                    if offset:
                        local_in_sec = min_start - offset.global_in_point
                        if local_in_sec < 0:
                            logger.warning(
                                f"Audio local_in_sec < 0 ({local_in_sec:.3f}s) "
                                f"for {audio_cam_id}; clamping to 0."
                            )
                            local_in_sec = 0.0
                        audio_in_frames = int(round(local_in_sec * self.fps))
                        audio_out_frames = audio_in_frames + total_duration_frames
                    else:
                        audio_in_frames = 0
                        audio_out_frames = total_duration_frames
                else:
                    audio_in_frames = 0
                    audio_out_frames = total_duration_frames

        has_audio_source = audio_source_path is not None

        if audio_out_frames <= audio_in_frames:
            audio_out_frames = audio_in_frames + max(1, total_duration_frames)

        # 2 mono tracks (A1/A2) using the same source
        for i in range(1, 3):
            track = ET.SubElement(audio_media_el, 'track')
            track.append(self._create_text_elem('enabled', "TRUE"))
            track.append(self._create_text_elem('locked', "FALSE"))
            track.append(self._create_text_elem('outputchannelindex', str(i)))
            
            if not has_audio_source:
                continue
            
            self.clip_item_count += 1
            clip_id = f"clipitem-{self.clip_item_count}"
            clip = ET.SubElement(track, 'clipitem', id=clip_id, premiereChannelType="mono")
            
            clip.append(self._create_text_elem('masterclipid', audio_master_clip_id))
            clip.append(self._create_text_elem('name', audio_source_path.name))
            clip.append(self._create_text_elem('enabled', "TRUE"))
            clip.append(self._create_text_elem('duration', total_duration_frames))
            
            cr = ET.SubElement(clip, 'rate')
            cr.append(self._create_text_elem('timebase', self.timebase))
            cr.append(self._create_text_elem('ntsc', self.ntsc))
            
            clip.append(self._create_text_elem('start', "0"))
            clip.append(self._create_text_elem('end', total_duration_frames))
            clip.append(self._create_text_elem('in', audio_in_frames))
            clip.append(self._create_text_elem('out', audio_out_frames))
            
            f_el = ET.SubElement(clip, 'file', id=audio_file_id)
            self._add_file_definition(
                f_el,
                audio_source_path,
                total_duration_frames,
                is_audio=(master_audio_path is not None),
            )
            
            clip.append(self._create_text_elem('sourcetrack', "1"))

        # --- Write XML ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ET.indent(xmeml, space="  ", level=0)
        tree = ET.ElementTree(xmeml)
        
        with open(output_path, 'wb') as f:
            f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(b'<!DOCTYPE xmeml>\n')
            tree.write(f, encoding='utf-8', xml_declaration=False)
        
        logger.info(f"✅ Premiere XMEML written to {output_path}")
