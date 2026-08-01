"""
Multimodal processing module for images and voice notes.
Handles text extraction (OCR/transcription) and media asset loading for Gemini Vision/Audio models.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import config

logger = logging.getLogger(__name__)


class MultimodalProcessor:
    """Handles feature extraction and media file loading for multimodal content."""

    def __init__(self, media_dir: Path = config.MEDIA_DIR):
        self.media_dir = media_dir
        self.images_dir = config.IMAGES_DIR
        self.audio_dir = config.AUDIO_DIR

    def get_image_file_path(self, image_id: str, image_info: Dict[str, Any]) -> Optional[Path]:
        """Resolves absolute Path for an image file."""
        file_path_str = image_info.get("file_path", "")
        rel_path = image_info.get("relative_path", file_path_str)

        candidates = []
        if rel_path:
            candidates.append(self.media_dir / rel_path)
            candidates.append(self.images_dir / Path(rel_path).name)
        candidates.append(self.images_dir / f"{image_id}.jpg")
        candidates.append(self.images_dir / f"{image_id}.png")

        for path in candidates:
            if path.exists() and path.is_file():
                return path

        return None

    def get_audio_file_path(self, voice_note_id: str, voice_info: Dict[str, Any]) -> Optional[Path]:
        """Resolves absolute Path for an audio voice note file."""
        file_path_str = voice_info.get("file_path", "")
        rel_path = voice_info.get("relative_path", file_path_str)

        candidates = []
        if rel_path:
            candidates.append(self.media_dir / rel_path)
            candidates.append(self.audio_dir / Path(rel_path).name)
        candidates.append(self.audio_dir / f"{voice_note_id}.ogg")
        candidates.append(self.audio_dir / f"{voice_note_id}.mp3")
        candidates.append(self.audio_dir / f"{voice_note_id}.wav")

        for path in candidates:
            if path.exists() and path.is_file():
                return path

        return None

    def load_pil_image(self, image_id: str, image_info: Dict[str, Any]):
        """Loads PIL Image object for multimodal Gemini API inspection."""
        path = self.get_image_file_path(image_id, image_info)
        if path:
            try:
                from PIL import Image

                return Image.open(path)
            except Exception as e:
                logger.warning(f"Could not load image {path}: {e}")
        return None

    def extract_image_content(self, image_id: str, image_info: Dict[str, Any]) -> str:
        """Extracts OCR text or metadata summary from image poster / screenshot."""
        path = self.get_image_file_path(image_id, image_info)
        description = image_info.get("ocr_text", "") or image_info.get("caption", "")

        if path:
            try:
                from PIL import Image

                with Image.open(path) as img:
                    width, height = img.size
                    fmt = img.format

                try:
                    import pytesseract

                    ocr_text = pytesseract.image_to_string(Image.open(path))
                    if ocr_text.strip():
                        return ocr_text.strip()
                except Exception:
                    pass

                return f"[Image: {fmt} {width}x{height} - Path: {path.name}] {description}".strip()
            except Exception as e:
                logger.debug(f"Image read error: {e}")

        return description or f"[Image ID: {image_id}]"

    def extract_voice_note_content(self, voice_note_id: str, voice_info: Dict[str, Any]) -> str:
        """Extracts transcription or audio characteristics from voice note."""
        transcript = voice_info.get("transcript", "") or voice_info.get("transcription", "")
        duration_sec = voice_info.get("duration_seconds", 0)

        if transcript:
            return transcript.strip()

        path = self.get_audio_file_path(voice_note_id, voice_info)
        file_desc = f"File: {path.name}" if path else "No local file"

        return f"[Voice Note ID: {voice_note_id}, Duration: {duration_sec}s, {file_desc}]"
