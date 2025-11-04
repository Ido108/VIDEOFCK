"""
Unit tests for video_utils module.
"""
import os
import pytest
import cv2
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from utils.video_utils import (
    extract_timestamp,
    extract_key_frames,
    extract_keyframes_by_number,
    extract_keyframes_by_threshold,
    save_keyframe,
    create_sequence_image_for_segment,
    create_video_with_audio,
    add_subtitles_to_video,
    process_video_with_subtitles
)


class TestExtractTimestamp:
    """Tests for extract_timestamp function."""

    @pytest.mark.unit
    def test_extract_valid_timestamp(self):
        """Test extracting timestamp from valid filename."""
        assert extract_timestamp("keyframe_0_3.50.jpg") == 3.50
        assert extract_timestamp("keyframe_5_120.75.jpg") == 120.75
        assert extract_timestamp("keyframe_10_0.00.jpg") == 0.00

    @pytest.mark.unit
    def test_extract_timestamp_no_match(self):
        """Test timestamp extraction from invalid filename."""
        assert extract_timestamp("invalid_filename.jpg") is None
        assert extract_timestamp("keyframe_0.jpg") is None
        assert extract_timestamp("random.txt") is None


class TestExtractKeyFrames:
    """Tests for keyframe extraction functions."""

    @pytest.mark.unit
    def test_extract_key_frames_threshold_mode(self, temp_dir, mock_video_path):
        """Test keyframe extraction using threshold method."""
        key_frames, timestamps = extract_key_frames(
            mock_video_path,
            temp_dir,
            keyframe_threshold=20,
            num_keyframes=None,
            keyframes_per_segment=3
        )

        assert isinstance(key_frames, list)
        assert isinstance(timestamps, list)
        assert len(key_frames) == len(timestamps)
        assert len(key_frames) > 0

        # Check that keyframes folder was created
        keyframes_folder = os.path.join(temp_dir, "keyframes")
        assert os.path.exists(keyframes_folder)

    @pytest.mark.unit
    def test_extract_key_frames_fixed_number(self, temp_dir, mock_video_path):
        """Test keyframe extraction with fixed number."""
        num_keyframes = 5
        key_frames, timestamps = extract_key_frames(
            mock_video_path,
            temp_dir,
            num_keyframes=num_keyframes,
            keyframes_per_segment=3
        )

        assert isinstance(key_frames, list)
        assert isinstance(timestamps, list)
        assert len(key_frames) >= num_keyframes
        assert len(timestamps) >= num_keyframes

    @pytest.mark.unit
    def test_extract_key_frames_invalid_video(self, temp_dir):
        """Test keyframe extraction with invalid video path."""
        key_frames, timestamps = extract_key_frames(
            "/nonexistent/video.mp4",
            temp_dir,
            keyframe_threshold=20
        )

        assert key_frames == []
        assert timestamps == []

    @pytest.mark.unit
    def test_extract_keyframes_by_number(self, temp_dir):
        """Test extracting fixed number of keyframes."""
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(temp_dir, "test.mp4")
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))

        for i in range(90):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        cap = cv2.VideoCapture(video_path)
        keyframes_folder = os.path.join(temp_dir, "keyframes")
        os.makedirs(keyframes_folder, exist_ok=True)

        key_frames, timestamps = extract_keyframes_by_number(
            cap, 5, keyframes_folder
        )

        cap.release()

        assert len(key_frames) >= 5
        assert len(timestamps) >= 5
        # Verify timestamps are increasing
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i-1]

    @pytest.mark.unit
    def test_save_keyframe(self, temp_dir):
        """Test saving a keyframe."""
        key_frames = []
        keyframe_timestamps = []
        keyframes_folder = os.path.join(temp_dir, "keyframes")
        os.makedirs(keyframes_folder, exist_ok=True)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp = 3.456

        save_keyframe(frame, timestamp, keyframes_folder, key_frames, keyframe_timestamps)

        assert len(key_frames) == 1
        assert len(keyframe_timestamps) == 1
        assert keyframe_timestamps[0] == timestamp

        # Check file was created
        expected_file = os.path.join(keyframes_folder, f"keyframe_0_{timestamp:.3f}.jpg")
        assert os.path.exists(expected_file)


class TestCreateSequenceImage:
    """Tests for create_sequence_image_for_segment function."""

    @pytest.mark.unit
    def test_create_sequence_image_success(self, temp_dir, mock_keyframes_folder):
        """Test creating sequence image for a segment."""
        segment = {
            "frame_numbers": [0, 1, 2],
            "start_time": 0.0,
            "end_time": 6.0
        }

        output_path = create_sequence_image_for_segment(
            segment,
            mock_keyframes_folder,
            temp_dir,
            keyframes_per_segment=3
        )

        assert output_path is not None
        assert os.path.exists(output_path)
        assert output_path.endswith(".jpg")

        # Verify image was created correctly
        img = Image.open(output_path)
        assert img.size[0] > 0
        assert img.size[1] > 0

    @pytest.mark.unit
    def test_create_sequence_image_no_keyframes(self, temp_dir):
        """Test creating sequence image when no keyframes exist."""
        keyframes_folder = os.path.join(temp_dir, "empty_keyframes")
        os.makedirs(keyframes_folder, exist_ok=True)

        segment = {
            "frame_numbers": [0, 1, 2],
            "start_time": 0.0,
            "end_time": 3.0
        }

        output_path = create_sequence_image_for_segment(
            segment,
            keyframes_folder,
            temp_dir,
            keyframes_per_segment=3
        )

        assert output_path is None


class TestVideoWithAudio:
    """Tests for create_video_with_audio function."""

    @pytest.mark.unit
    @patch('utils.video_utils.VideoFileClip')
    @patch('utils.video_utils.AudioFileClip')
    @patch('utils.video_utils.AudioArrayClip')
    def test_create_video_with_audio_success(
        self, mock_audio_array_clip, mock_audio_file_clip, mock_video_file_clip, temp_dir
    ):
        """Test creating video with audio narration."""
        # Setup mocks
        mock_video = Mock()
        mock_video.duration = 10.0
        mock_video.fps = 30.0
        mock_video.audio = None
        mock_video_file_clip.return_value = mock_video

        mock_audio = Mock()
        mock_audio.to_soundarray.return_value = np.zeros((44100, 2), dtype=np.float32)
        mock_audio_file_clip.return_value = mock_audio

        mock_composite_audio = Mock()
        mock_audio_array_clip.return_value = mock_composite_audio

        mock_final = Mock()
        mock_video.with_audio.return_value = mock_final

        video_path = os.path.join(temp_dir, "input.mp4")
        audio_paths = [os.path.join(temp_dir, "audio1.mp3")]
        timestamps = [0.0, 5.0]
        output_path = os.path.join(temp_dir, "output.mp4")

        result = create_video_with_audio(
            video_path, audio_paths, timestamps, output_path, original_audio_volume=0.5
        )

        assert result == output_path
        mock_video.with_audio.assert_called_once()
        mock_final.write_videofile.assert_called_once()

    @pytest.mark.unit
    @patch('utils.video_utils.VideoFileClip')
    def test_create_video_with_audio_error(self, mock_video_file_clip, temp_dir):
        """Test error handling in create_video_with_audio."""
        mock_video_file_clip.side_effect = Exception("Video loading failed")

        video_path = os.path.join(temp_dir, "input.mp4")
        audio_paths = [os.path.join(temp_dir, "audio1.mp3")]
        timestamps = [0.0]
        output_path = os.path.join(temp_dir, "output.mp4")

        result = create_video_with_audio(
            video_path, audio_paths, timestamps, output_path
        )

        assert result is None


class TestSubtitles:
    """Tests for subtitle-related functions."""

    @pytest.mark.unit
    @patch('utils.video_utils.subprocess.Popen')
    def test_add_subtitles_success(self, mock_popen, temp_dir):
        """Test adding subtitles to video."""
        mock_process = Mock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        input_video = os.path.join(temp_dir, "input.mp4")
        srt_file = os.path.join(temp_dir, "subs.srt")
        output_video = os.path.join(temp_dir, "output.mp4")

        # Create dummy files
        open(input_video, 'w').close()
        open(srt_file, 'w').close()

        result = add_subtitles_to_video(input_video, srt_file, output_video)

        assert result == output_video
        mock_popen.assert_called_once()

    @pytest.mark.unit
    def test_add_subtitles_missing_srt(self, temp_dir):
        """Test adding subtitles when SRT file is missing."""
        input_video = os.path.join(temp_dir, "input.mp4")
        srt_file = os.path.join(temp_dir, "nonexistent.srt")
        output_video = os.path.join(temp_dir, "output.mp4")

        result = add_subtitles_to_video(input_video, srt_file, output_video)

        # Should return input video without modification
        assert result == input_video

    @pytest.mark.unit
    @patch('utils.video_utils.subprocess.Popen')
    def test_add_subtitles_ffmpeg_error(self, mock_popen, temp_dir):
        """Test subtitle burning with FFmpeg error."""
        mock_process = Mock()
        mock_process.communicate.return_value = (b"", b"Error message")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        input_video = os.path.join(temp_dir, "input.mp4")
        srt_file = os.path.join(temp_dir, "subs.srt")
        output_video = os.path.join(temp_dir, "output.mp4")

        # Create dummy files
        open(input_video, 'w').close()
        open(srt_file, 'w').close()

        result = add_subtitles_to_video(input_video, srt_file, output_video)

        # Should return input video on failure
        assert result == input_video

    @pytest.mark.unit
    @patch('utils.video_utils.transcribe_audio')
    @patch('utils.video_utils.format_transcript_to_srt')
    @patch('utils.video_utils.add_subtitles_to_video')
    def test_process_video_with_subtitles_success(
        self, mock_add_subs, mock_format_srt, mock_transcribe, temp_dir
    ):
        """Test complete subtitle processing workflow."""
        mock_transcribe.return_value = {"segments": [{"text": "Test"}]}
        mock_add_subs.return_value = os.path.join(temp_dir, "output.mp4")

        audio_path = os.path.join(temp_dir, "audio.mp3")
        video_path = os.path.join(temp_dir, "video.mp4")
        output_path = os.path.join(temp_dir, "output.mp4")

        result = process_video_with_subtitles(audio_path, video_path, output_path)

        assert result is not None
        mock_transcribe.assert_called_once_with(audio_path)
        mock_format_srt.assert_called_once()
        mock_add_subs.assert_called_once()

    @pytest.mark.unit
    @patch('utils.video_utils.transcribe_audio')
    def test_process_video_with_subtitles_transcription_fail(
        self, mock_transcribe, temp_dir
    ):
        """Test subtitle processing when transcription fails."""
        mock_transcribe.return_value = None

        audio_path = os.path.join(temp_dir, "audio.mp3")
        video_path = os.path.join(temp_dir, "video.mp4")
        output_path = os.path.join(temp_dir, "output.mp4")

        result = process_video_with_subtitles(audio_path, video_path, output_path)

        assert result is None
