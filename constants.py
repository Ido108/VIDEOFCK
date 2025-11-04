"""
Constants and configuration values for VideoFck application.

This module centralizes all magic numbers and configuration values to improve
maintainability and make the codebase easier to understand and modify.
"""

# ============================================================================
# Video Processing Constants
# ============================================================================

# Keyframe Extraction
KEYFRAME_THRESHOLD_DEFAULT = 20  # Default frame difference threshold
MIN_TIME_BETWEEN_KEYFRAMES = 3.0  # Minimum seconds between keyframes
KEYFRAMES_PER_SEGMENT_DEFAULT = 3  # Default number of keyframes per segment
KEYFRAMES_PER_SEGMENT_MIN = 1
KEYFRAMES_PER_SEGMENT_MAX = 10

# Video Codec Settings
VIDEO_CODEC = 'libx264'
AUDIO_CODEC = 'aac'
TEMP_AUDIO_FILE = 'temp-audio.m4a'

# Audio Settings
AUDIO_SAMPLE_RATE = 44100  # Hz
DEFAULT_ORIGINAL_AUDIO_VOLUME = 1.0

# ============================================================================
# Narration Generation Constants
# ============================================================================

# Speech Rate
WORDS_PER_SECOND = 2.5  # Target speaking rate for TTS

# API Retry Configuration
API_MAX_RETRIES = 3
API_RETRY_DELAY_SECONDS = 2
API_TIMEOUT_SECONDS = 120

# Model Names
ANTHROPIC_MODEL_NAME = "claude-3-5-sonnet-20241022"
OPENAI_MODEL_NAME = "gpt-4o-2024-11-20"

# API Token Limits
OPENAI_MAX_TOKENS = 2000
ANTHROPIC_MAX_TOKENS = 2000

# Model Temperature Settings
OPENAI_TEMPERATURE = 0.6
ANTHROPIC_TEMPERATURE = 0.7

# ============================================================================
# Image Processing Constants
# ============================================================================

# Sequence Image Settings
SEQUENCE_IMAGE_BORDER_SIZE = 2
SEQUENCE_IMAGE_TARGET_WIDTH = 1200
SEQUENCE_IMAGE_QUALITY = 100

# Font Settings
DEFAULT_FONT_NAME = "arial.ttf"
DEFAULT_FONT_SIZE = 35
TIMESTAMP_TEXT_PADDING = 15
FRAME_NUMBER_TEXT_SPACING = 10

# Colors
BORDER_COLOR = "white"
BACKGROUND_COLOR = "black"
TEXT_COLOR_WHITE = "white"
TEXT_COLOR_YELLOW = "yellow"
TEXT_BACKGROUND_COLOR = (0, 0, 0, 180)

# ============================================================================
# Subtitle Processing Constants
# ============================================================================

# FFmpeg Subtitle Style
SUBTITLE_FONT_SIZE = 18
SUBTITLE_BORDER_COLOR = "&H000000"
SUBTITLE_BORDER_STYLE = 3
SUBTITLE_OUTLINE = 1
SUBTITLE_SHADOW = 1
SUBTITLE_MARGIN_V = 10

# ============================================================================
# Logging Constants
# ============================================================================

# Log File Settings
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3
LOG_FILE_NAME = "VIDEOFCK.log"

# ============================================================================
# Playback Speed Constants
# ============================================================================

PLAYBACK_SPEED_MIN = 0.5
PLAYBACK_SPEED_MAX = 2.0
PLAYBACK_SPEED_DEFAULT = 1.0

# ============================================================================
# File Extensions
# ============================================================================

KEYFRAME_EXTENSION = ".jpg"
SUBTITLE_EXTENSION = ".srt"
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
AUDIO_EXTENSIONS = [".mp3", ".wav", ".m4a", ".aac"]

# ============================================================================
# Validation Constants
# ============================================================================

MAX_VIDEO_FILE_SIZE_MB = 500  # Maximum upload size in MB
MIN_VIDEO_DURATION_SECONDS = 1  # Minimum video duration
MAX_VIDEO_DURATION_SECONDS = 600  # Maximum video duration (10 minutes)

# ============================================================================
# Error Messages
# ============================================================================

ERROR_VIDEO_NOT_FOUND = "Video file not found at specified path"
ERROR_API_KEY_MISSING = "API key not found in environment"
ERROR_INVALID_MODEL_CHOICE = "Invalid model choice specified"
ERROR_KEYFRAME_EXTRACTION_FAILED = "Failed to extract keyframes from video"
ERROR_NARRATION_GENERATION_FAILED = "Failed to generate narration"
ERROR_TTS_GENERATION_FAILED = "Failed to generate text-to-speech audio"
ERROR_SUBTITLE_GENERATION_FAILED = "Failed to generate subtitles"
