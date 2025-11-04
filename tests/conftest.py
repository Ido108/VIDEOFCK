"""
Pytest configuration and fixtures for VideoFck test suite.
"""
import os
import sys
import pytest
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from PIL import Image
import cv2

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_video_path(temp_dir):
    """Create a mock video file path."""
    video_path = os.path.join(temp_dir, "test_video.mp4")

    # Create a simple test video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))

    # Write 90 frames (3 seconds at 30fps)
    for i in range(90):
        # Create a frame with changing content
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some variation every 30 frames
        if i < 30:
            frame[:, :] = [255, 0, 0]  # Blue
        elif i < 60:
            frame[:, :] = [0, 255, 0]  # Green
        else:
            frame[:, :] = [0, 0, 255]  # Red
        out.write(frame)

    out.release()
    return video_path


@pytest.fixture
def mock_keyframe_image(temp_dir):
    """Create a mock keyframe image."""
    image_path = os.path.join(temp_dir, "keyframe_0_0.00.jpg")
    img = Image.new('RGB', (640, 480), color='blue')
    img.save(image_path)
    return image_path


@pytest.fixture
def mock_keyframes_folder(temp_dir):
    """Create a folder with mock keyframe images."""
    keyframes_folder = os.path.join(temp_dir, "keyframes")
    os.makedirs(keyframes_folder, exist_ok=True)

    # Create multiple keyframes with different timestamps
    timestamps = [0.00, 3.00, 6.00, 9.00]
    for i, ts in enumerate(timestamps):
        img_path = os.path.join(keyframes_folder, f"keyframe_{i}_{ts:.2f}.jpg")
        img = Image.new('RGB', (640, 480), color=['blue', 'green', 'red', 'yellow'][i])
        img.save(img_path)

    return keyframes_folder


@pytest.fixture
def mock_audio_path(temp_dir):
    """Create a mock audio file."""
    audio_path = os.path.join(temp_dir, "test_audio.mp3")
    # Create empty file (actual audio generation not needed for most tests)
    open(audio_path, 'w').close()
    return audio_path


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic Claude API client."""
    with patch('utils.gemini_utils.anthropic_client') as mock_client:
        mock_response = Mock()
        mock_response.content = [Mock(text="<BEGIN SEGMENT>\nFrame Numbers: [0, 1, 2]\nTimestamp: 0.00-3.00\nNarration: Test narration\n<END SEGMENT>")]
        mock_client.messages.create.return_value = mock_response
        yield mock_client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI GPT API client."""
    with patch('utils.gemini_utils.openai_client') as mock_client:
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="<BEGIN SEGMENT>\nFrame Numbers: [0, 1, 2]\nTimestamp: 0.00-3.00\nNarration: Test narration\n<END SEGMENT>"))]
        mock_client.chat.completions.create.return_value = mock_response
        yield mock_client


@pytest.fixture
def mock_google_tts_client():
    """Mock Google Cloud TTS client."""
    with patch('utils.tts_utils.texttospeech.TextToSpeechClient') as mock_client:
        mock_instance = Mock()
        mock_response = Mock()
        mock_response.audio_content = b"fake audio content"
        mock_instance.synthesize_speech.return_value = mock_response
        mock_client.return_value = mock_instance
        yield mock_client


@pytest.fixture
def mock_elevenlabs_client():
    """Mock ElevenLabs TTS client."""
    with patch('utils.tts_utils.generate') as mock_generate:
        mock_generate.return_value = [b"fake", b"audio", b"content"]
        yield mock_generate


@pytest.fixture
def sample_script_data():
    """Sample script data for testing."""
    return [
        {
            "frame_numbers": [0, 1, 2],
            "timestamp": "0.00-3.00",
            "narration": "This is the first narration segment.",
            "OST": 2
        },
        {
            "frame_numbers": [2, 3, 4],
            "timestamp": "3.00-6.00",
            "narration": "This is the second narration segment.",
            "OST": 2
        }
    ]


@pytest.fixture
def sample_segments():
    """Sample segment data for testing."""
    return [
        {
            "frame_numbers": [0, 1, 2],
            "start_time": 0.0,
            "end_time": 3.0,
            "duration": 3.0,
            "max_words": 7,
            "image_path": "/fake/path/segment_0-2.jpg"
        },
        {
            "frame_numbers": [2, 3, 4],
            "start_time": 3.0,
            "end_time": 6.0,
            "duration": 3.0,
            "max_words": 7,
            "image_path": "/fake/path/segment_2-4.jpg"
        }
    ]


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before each test."""
    import logging
    # Clear handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Reset level
    logging.root.setLevel(logging.WARNING)
    yield
    # Clean up after test
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


@pytest.fixture
def mock_video_clip():
    """Mock MoviePy VideoFileClip."""
    with patch('utils.video_utils.VideoFileClip') as mock_clip:
        mock_instance = Mock()
        mock_instance.duration = 10.0
        mock_instance.fps = 30.0
        mock_instance.audio = None
        mock_clip.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_audio_clip():
    """Mock MoviePy AudioFileClip."""
    with patch('utils.video_utils.AudioFileClip') as mock_clip:
        mock_instance = Mock()
        mock_instance.to_soundarray.return_value = np.zeros((44100, 2), dtype=np.float32)
        mock_clip.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def set_env_vars():
    """Set environment variables for testing."""
    original_env = os.environ.copy()
    os.environ['CLAUDE_API_KEY'] = 'test_claude_key'
    os.environ['OPENAI_API_KEY'] = 'test_openai_key'
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'test_google_creds'
    os.environ['ELEVENLABS_API_KEY'] = 'test_elevenlabs_key'
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
