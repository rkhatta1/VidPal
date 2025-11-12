# processing/vlm_processor.py (CORRECTED VERSION)
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import cv2
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import get_settings
from db.models import EpisodeRepository
from tqdm import tqdm

logger = logging.getLogger(__name__)

IMAGE_TOKEN_INDEX = -200  # FastVLM's special image token


class VLMProcessor:
    """Process video frames with VLM (Visual Language Model) at scene transitions."""
    
    def __init__(self):
        self.settings = get_settings()
        self.device = "cuda" if (torch.cuda.is_available() and self.settings.USE_GPU) else "cpu"
        
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
        Process video frames at scene transitions with VLM.
        Analyzes 2 seconds before and after each cut.
        
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
        
        logger.info("Processing scene transitions with VLM...")
        
        # Extract transition times (where camera changes)
        transition_times = self._extract_transitions(cuts)
        logger.info(f"Found {len(transition_times)} scene transitions")
        
        # Process each camera
        all_descriptions = []
        
        for camera_id, video_path in tqdm(video_paths.items(), desc="Processing cameras", unit="camera"):
            logger.info(f"Processing {camera_id}...")
            descriptions = self._process_video(
                video_path,
                camera_id,
                transition_times,
            )
            all_descriptions.extend(descriptions)
        
        logger.info(f"✅ VLM processing complete: {len(all_descriptions)} descriptions")
        return all_descriptions
    
    def _extract_transitions(
        self,
        cuts: List[Dict[str, Any]],
    ) -> List[float]:
        """Extract timestamps where camera changes occur."""
        transitions = []
        
        for i in range(len(cuts) - 1):
            current_camera = cuts[i]['camera_id']
            next_camera = cuts[i + 1]['camera_id']
            
            # Camera change detected
            if current_camera != next_camera:
                transition_time = cuts[i]['end_time']
                transitions.append(transition_time)
        
        return transitions
    
    def _process_video(
        self,
        video_path: Path,
        camera_id: str,
        transition_times: List[float],
    ) -> List[Dict[str, Any]]:
        """
        Process video file and extract VLM descriptions around transitions.
        
        Captures frames at:
        - 2 seconds before transition
        - 1 second before transition
        - At transition
        - 1 second after transition
        - 2 seconds after transition
        """
        descriptions = []
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps == 0:
            logger.error(f"Could not get FPS for {camera_id}")
            cap.release()
            return descriptions
        
        logger.info(f"{camera_id}: FPS={fps:.2f}")
        
        # Calculate total samples
        sample_offsets = [-2, -1, 0, 1, 2]
        total_samples = len(transition_times) * len(sample_offsets)
        
        # Process each transition with progress bar
        with tqdm(
            total=total_samples,
            desc=f"VLM {camera_id}",
            unit="frame",
            leave=False
        ) as pbar:
            for transition_time in transition_times:
                for offset in sample_offsets:
                    sample_time = transition_time + offset
                    
                    if sample_time < 0:
                        pbar.update(1)
                        continue
                    
                    # Extract frame
                    frame = self._extract_frame(cap, sample_time, fps)
                    
                    if frame is None:
                        pbar.update(1)
                        continue
                    
                    # Get VLM description
                    description = self._describe_frame(frame, camera_id)
                    
                    descriptions.append({
                        'camera_id': camera_id,
                        'time': sample_time,
                        'transition_time': transition_time,
                        'offset': offset,
                        'description': description,
                    })
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'time': f"{sample_time:.1f}s",
                        'desc_len': len(description)
                    })
        
        cap.release()
        logger.info(f"{camera_id}: Extracted {len(descriptions)} VLM descriptions")
        
        return descriptions
    
    def _extract_frame(
        self,
        cap: cv2.VideoCapture,
        time_seconds: float,
        fps: float,
    ) -> Optional[Image.Image]:
        """Extract a single frame at specified time."""
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
        """Generate VLM description for a frame using FastVLM's API."""
        
        # Build chat message with image placeholder
        messages = [
            {
                "role": "user",
                "content": f"<image>\nDescribe this video frame from {camera_id}. Focus on: people visible, their actions, facial expressions, body language, and scene composition. Be concise."
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
            for prompt_part in ["Describe this video frame", "Focus on:", camera_id]:
                description = description.replace(prompt_part, "").strip()
            
            return description
            
        except Exception as e:
            logger.error(f"VLM generation failed: {e}")
            return f"Error generating description: {str(e)}"
