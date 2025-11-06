# processing/vlm_processor.py

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import cv2
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from config import get_settings
from utils.caching import Cache, compute_file_hash

logger = logging.getLogger(__name__)

IMAGE_TOKEN_INDEX = -200  # FastVLM's special image token


class VLMProcessor:
    """Process video frames with VLM (Visual Language Model) at cut boundaries."""
    
    def __init__(self, cache: Optional[Cache] = None):
        self.settings = get_settings()
        self.device = "cuda" if (torch.cuda.is_available() and self.settings.USE_GPU) else "cpu"
        self.cache = cache  # ✅ ADD THIS
        
        # Load FastVLM model
        logger.info("Loading FastVLM model...")
        self.model_id = "apple/FastVLM-1.5B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        
        logger.info(f"✅ VLM processor initialized (device: {self.device})")
    
    def process_scene_transitions(
        self,
        video_paths: Dict[str, Path],
        cuts: List[Dict[str, Any]],
        episode_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Process video frames at EVERY cut boundary for ALL cameras.
        Extracts 2 frames per boundary: 1 before and 1 after the cut.
        
        Uses caching to avoid reprocessing the same videos.
        
        Args:
            video_paths: Dictionary of camera_id -> video file path
            cuts: List of EDL cuts
            episode_id: Episode identifier
        
        Returns:
            List of all VLM descriptions with metadata
        """
        if not self.settings.ENABLE_VLM_PROCESSING:
            logger.info("VLM processing disabled, skipping")
            return []
        
        logger.info("Processing cut boundaries with VLM for all cameras...")
        
        # Check cache first
        if self.cache and self.settings.ENABLE_CACHING:
            # Create cache key based on video files and cuts
            cache_key = self._create_cache_key(video_paths, cuts)
            cached_descriptions = self.cache.get(cache_key, "vlm_descriptions")
            
            if cached_descriptions:
                logger.info(f"✅ Using cached VLM descriptions ({len(cached_descriptions)} descriptions)")
                return cached_descriptions
        
        # Extract ALL cut boundaries (where ANY camera changes)
        cut_boundaries = self._extract_cut_boundaries(cuts)
        logger.info(f"Found {len(cut_boundaries)} cut boundaries")
        
        # Log expected workload (2 frames per boundary per camera)
        total_expected = len(video_paths) * len(cut_boundaries) * 2
        logger.info(f"Expected VLM descriptions: {len(video_paths)} cameras × {len(cut_boundaries)} boundaries × 2 frames = {total_expected} descriptions")
        
        # Process each camera - describe ALL boundaries
        all_descriptions = []
        
        for camera_id, video_path in tqdm(
            video_paths.items(),
            desc="Processing cameras",
            unit="camera"
        ):
            logger.info(f"Processing {camera_id} ({len(cut_boundaries)} boundaries)...")
            descriptions = self._process_video_all_boundaries(
                video_path,
                camera_id,
                cut_boundaries,
            )
            all_descriptions.extend(descriptions)
        
        logger.info(f"✅ VLM processing complete: {len(all_descriptions)} descriptions generated")
        
        # Cache the results
        if self.cache and self.settings.ENABLE_CACHING:
            cache_key = self._create_cache_key(video_paths, cuts)
            self.cache.set(cache_key, "vlm_descriptions", all_descriptions)
            logger.info(f"💾 Cached VLM descriptions ({len(all_descriptions)} descriptions)")
        
        return all_descriptions
    
    def _create_cache_key(self, video_paths: Dict[str, Path], cuts: List[Dict[str, Any]]) -> str:
        """
        Create a cache key based on video files and cut boundaries.
        
        If videos change or cuts change significantly, the key changes.
        """
        # Hash the video file hashes
        video_hashes = []
        for cam_id in sorted(video_paths.keys()):
            video_path = video_paths[cam_id]
            try:
                file_hash = compute_file_hash(video_path)
                video_hashes.append(f"{cam_id}:{file_hash}")
            except Exception as e:
                logger.warning(f"Could not hash {cam_id}: {e}")
                return None  # Skip caching if we can't hash
        
        # Create key from video hashes and cut count
        video_key = "_".join(video_hashes)
        cut_boundaries_count = sum(1 for i in range(len(cuts) - 1) if cuts[i]['camera_id'] != cuts[i + 1]['camera_id'])
        
        cache_key = f"vlm_{video_key}_{cut_boundaries_count}cuts"
        return cache_key
    
    def _extract_cut_boundaries(
        self,
        cuts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Extract ALL times where camera changes occur.
        Every boundary is returned, regardless of which cameras are involved.
        
        Args:
            cuts: List of EDL cuts
        
        Returns:
            List of cut boundaries with camera transition info
        """
        boundaries = []
        
        for i in range(len(cuts) - 1):
            current_cut = cuts[i]
            next_cut = cuts[i + 1]
            
            current_camera = current_cut['camera_id']
            next_camera = next_cut['camera_id']
            
            # Camera change detected
            if current_camera != next_camera:
                boundary_time = current_cut['end_time']
                boundaries.append({
                    'time': boundary_time,
                    'from_camera': current_camera,
                    'to_camera': next_camera,
                    'cut_index': i,
                })
        
        return boundaries
    
    def _process_video_all_boundaries(
        self,
        video_path: Path,
        camera_id: str,
        cut_boundaries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Process video and extract frames at ALL cut boundaries.
        Each camera processes every boundary to provide complete context.
        Samples 2 frames: one 0.5s before the cut, one 0.5s after.
        
        Args:
            video_path: Path to video file
            camera_id: Camera identifier
            cut_boundaries: List of all cut boundaries
        
        Returns:
            List of VLM descriptions for this camera
        """
        descriptions = []
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps == 0:
            logger.error(f"Could not get FPS for {camera_id}")
            cap.release()
            return descriptions
        
        logger.info(f"{camera_id}: FPS={fps:.2f}, processing {len(cut_boundaries)} boundaries")
        
        # Sample: -0.5s (before cut) and +0.5s (after cut)
        sample_offsets = [-0.5, 0.5]
        total_samples = len(cut_boundaries) * len(sample_offsets)
        
        with tqdm(
            total=total_samples,
            desc=f"VLM {camera_id}",
            unit="frame",
            leave=False
        ) as pbar:
            for boundary in cut_boundaries:
                boundary_time = boundary['time']
                
                for offset in sample_offsets:
                    sample_time = boundary_time + offset
                    
                    if sample_time < 0:
                        pbar.update(1)
                        continue
                    
                    # Extract frame from THIS camera
                    frame = self._extract_frame(cap, sample_time, fps)
                    
                    if frame is None:
                        pbar.update(1)
                        continue
                    
                    # Get VLM description for this camera at this boundary
                    description = self._describe_frame(frame, camera_id)
                    
                    # Label offset as "before" or "after"
                    offset_label = "before cut" if offset < 0 else "after cut"
                    
                    descriptions.append({
                        'camera_id': camera_id,
                        'time': sample_time,
                        'cut_time': boundary_time,
                        'offset': offset,
                        'offset_label': offset_label,
                        'from_camera': boundary['from_camera'],
                        'to_camera': boundary['to_camera'],
                        'description': description,
                    })
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'boundary': f"{boundary['from_camera']}→{boundary['to_camera']}",
                        'offset': offset_label,
                    })
        
        cap.release()
        logger.info(f"{camera_id}: Extracted {len(descriptions)} descriptions for {len(cut_boundaries)} boundaries")
        
        return descriptions
    
    def _extract_frame(
        self,
        cap: cv2.VideoCapture,
        time_seconds: float,
        fps: float,
    ) -> Optional[Image.Image]:
        """
        Extract a single frame at specified time.
        
        Args:
            cap: OpenCV video capture object
            time_seconds: Time in seconds
            fps: Frames per second
        
        Returns:
            PIL Image or None if frame couldn't be extracted
        """
        frame_number = int(time_seconds * fps)
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        return Image.fromarray(frame_rgb)
    
    def _describe_frame(
        self,
        image: Image.Image,
        camera_id: str,
    ) -> str:
        """
        Generate VLM description for a frame using FastVLM's API.
        
        Args:
            image: PIL Image to describe
            camera_id: Camera identifier for context
        
        Returns:
            VLM description text
        """
        
        # Build chat message with image placeholder
        messages = [
            {
                "role": "user",
                "content": f"<image>\nDescribe what's happening in this video frame from {camera_id}. Focus on: people visible, their actions, facial expressions, body language, and scene composition. Be concise (1-2 sentences)."
            }
        ]
        
        # Render to string (not tokens yet)
        rendered = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Split around the <image> placeholder
        try:
            pre, post = rendered.split("<image>", 1)
        except ValueError:
            logger.error("Could not find <image> placeholder in template")
            return "Error: Invalid template"
        
        # Tokenize the text around the image token (no extra special tokens)
        pre_ids = self.tokenizer(
            pre,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids
        
        post_ids = self.tokenizer(
            post,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids
        
        # Splice in the IMAGE token ID (-200) at the placeholder position
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.model.device)
        attention_mask = torch.ones_like(input_ids, device=self.model.device)
        
        # Preprocess image via the model's own processor
        pixel_values = self.model.get_vision_tower().image_processor(
            images=image.convert("RGB"),
            return_tensors="pt"
        )["pixel_values"]
        pixel_values = pixel_values.to(self.model.device, dtype=self.model.dtype)
        
        # Generate description
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=pixel_values,
                    max_new_tokens=128,
                    do_sample=False,
                )
            
            # Decode output
            description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from output
            if "<image>" in description:
                description = description.split("<image>", 1)[1].strip()
            
            # Remove any remaining prompt text
            for prompt_part in ["Describe what's happening", "Focus on:", camera_id, "video frame from"]:
                if prompt_part in description:
                    description = description.replace(prompt_part, "").strip()
            
            # Clean up multiple spaces
            description = " ".join(description.split())
            
            return description
            
        except Exception as e:
            logger.error(f"VLM generation failed: {e}")
            return f"Error generating description: {str(e)[:100]}"
