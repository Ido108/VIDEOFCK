"""
Input validation and security utilities for VideoFck.
"""
import os
import re
import logging
from pathlib import Path
from typing import Optional, List
from constants import (
    MAX_VIDEO_FILE_SIZE_MB,
    MIN_VIDEO_DURATION_SECONDS,
    MAX_VIDEO_DURATION_SECONDS,
    VIDEO_EXTENSIONS,
    KEYFRAMES_PER_SEGMENT_MIN,
    KEYFRAMES_PER_SEGMENT_MAX,
    PLAYBACK_SPEED_MIN,
    PLAYBACK_SPEED_MAX
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def sanitize_file_path(file_path: str, base_dir: Optional[str] = None) -> str:
    """
    Sanitize file path to prevent path traversal attacks.

    Args:
        file_path: File path to sanitize
        base_dir: Optional base directory to restrict access to

    Returns:
        Sanitized absolute file path

    Raises:
        ValidationError: If path is invalid or contains traversal attempts
    """
    try:
        # Convert to Path object for safer handling
        path = Path(file_path).resolve()

        # Check for path traversal attempts
        if ".." in str(file_path):
            logger.warning(f"Path traversal attempt detected: {file_path}")
            raise ValidationError("Path traversal not allowed")

        # If base_dir specified, ensure path is within it
        if base_dir:
            base = Path(base_dir).resolve()
            if not str(path).startswith(str(base)):
                logger.warning(f"Path outside base directory: {file_path}")
                raise ValidationError(f"Path must be within {base_dir}")

        return str(path)

    except (ValueError, OSError) as e:
        logger.error(f"Invalid file path: {file_path} - {str(e)}")
        raise ValidationError(f"Invalid file path: {str(e)}")


def validate_video_file(file_path: str) -> bool:
    """
    Validate video file exists and has valid extension.

    Args:
        file_path: Path to video file

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if not file_path:
        raise ValidationError("Video file path is required")

    # Sanitize path
    safe_path = sanitize_file_path(file_path)

    # Check file exists
    if not os.path.exists(safe_path):
        raise ValidationError(f"Video file not found: {safe_path}")

    # Check file extension
    ext = os.path.splitext(safe_path)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        raise ValidationError(
            f"Invalid video extension '{ext}'. Allowed: {', '.join(VIDEO_EXTENSIONS)}"
        )

    # Check file size
    file_size_mb = os.path.getsize(safe_path) / (1024 * 1024)
    if file_size_mb > MAX_VIDEO_FILE_SIZE_MB:
        raise ValidationError(
            f"Video file too large ({file_size_mb:.1f}MB). "
            f"Maximum: {MAX_VIDEO_FILE_SIZE_MB}MB"
        )

    logger.info(f"Video file validated: {safe_path} ({file_size_mb:.1f}MB)")
    return True


def validate_video_duration(duration: float) -> bool:
    """
    Validate video duration is within acceptable range.

    Args:
        duration: Video duration in seconds

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if duration < MIN_VIDEO_DURATION_SECONDS:
        raise ValidationError(
            f"Video too short ({duration:.1f}s). "
            f"Minimum: {MIN_VIDEO_DURATION_SECONDS}s"
        )

    if duration > MAX_VIDEO_DURATION_SECONDS:
        raise ValidationError(
            f"Video too long ({duration:.1f}s). "
            f"Maximum: {MAX_VIDEO_DURATION_SECONDS}s"
        )

    return True


def validate_keyframes_per_segment(value: int) -> bool:
    """
    Validate keyframes per segment parameter.

    Args:
        value: Number of keyframes per segment

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, int):
        raise ValidationError("Keyframes per segment must be an integer")

    if value < KEYFRAMES_PER_SEGMENT_MIN or value > KEYFRAMES_PER_SEGMENT_MAX:
        raise ValidationError(
            f"Keyframes per segment must be between "
            f"{KEYFRAMES_PER_SEGMENT_MIN} and {KEYFRAMES_PER_SEGMENT_MAX}"
        )

    return True


def validate_playback_speed(speed: float) -> bool:
    """
    Validate playback speed factor.

    Args:
        speed: Playback speed multiplier

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(speed, (int, float)):
        raise ValidationError("Playback speed must be a number")

    if speed < PLAYBACK_SPEED_MIN or speed > PLAYBACK_SPEED_MAX:
        raise ValidationError(
            f"Playback speed must be between "
            f"{PLAYBACK_SPEED_MIN}x and {PLAYBACK_SPEED_MAX}x"
        )

    return True


def validate_api_key(api_key: Optional[str], service_name: str) -> bool:
    """
    Validate API key is present and has reasonable format.

    Args:
        api_key: API key string
        service_name: Name of the service (for error messages)

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if not api_key:
        raise ValidationError(f"{service_name} API key is required")

    # Basic format check - should be non-empty and reasonable length
    if len(api_key.strip()) < 10:
        raise ValidationError(f"Invalid {service_name} API key format")

    # Check for placeholder values
    placeholder_patterns = ["test", "your", "key", "xxx", "placeholder"]
    if any(pattern in api_key.lower() for pattern in placeholder_patterns):
        logger.warning(f"Potential placeholder API key detected for {service_name}")

    return True


def sanitize_text_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize text input to prevent injection attacks.

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed text length

    Returns:
        Sanitized text

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(text, str):
        raise ValidationError("Input must be a string")

    # Trim to max length
    if len(text) > max_length:
        logger.warning(f"Text input truncated from {len(text)} to {max_length} chars")
        text = text[:max_length]

    # Remove null bytes and other control characters (except newlines/tabs)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)

    return text.strip()


def validate_model_choice(model: str, allowed_models: List[str]) -> bool:
    """
    Validate model selection.

    Args:
        model: Selected model name
        allowed_models: List of allowed model names

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if not model:
        raise ValidationError("Model choice is required")

    if model not in allowed_models:
        raise ValidationError(
            f"Invalid model '{model}'. Allowed: {', '.join(allowed_models)}"
        )

    return True


def validate_output_directory(output_dir: str) -> bool:
    """
    Validate and create output directory if needed.

    Args:
        output_dir: Path to output directory

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    try:
        # Sanitize path
        safe_dir = sanitize_file_path(output_dir)

        # Create if doesn't exist
        os.makedirs(safe_dir, exist_ok=True)

        # Check write permissions
        if not os.access(safe_dir, os.W_OK):
            raise ValidationError(f"No write permission for directory: {safe_dir}")

        return True

    except (OSError, PermissionError) as e:
        raise ValidationError(f"Cannot create output directory: {str(e)}")


def is_safe_filename(filename: str) -> bool:
    """
    Check if filename is safe (no path traversal, special chars).

    Args:
        filename: Filename to check

    Returns:
        True if safe, False otherwise
    """
    # Check for path separators
    if os.sep in filename or (os.altsep and os.altsep in filename):
        return False

    # Check for path traversal
    if ".." in filename:
        return False

    # Check for null bytes
    if "\x00" in filename:
        return False

    # Check for reserved names on Windows
    reserved_names = [
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    ]
    name_without_ext = os.path.splitext(filename)[0].upper()
    if name_without_ext in reserved_names:
        return False

    return True
