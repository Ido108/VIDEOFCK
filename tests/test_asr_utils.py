"""
Unit tests for asr_utils module.
"""
import os
import json
import pytest
from unittest.mock import Mock, patch, mock_open

from utils.asr_utils import transcribe_audio, format_transcript_to_srt


class TestTranscribeAudio:
    """Tests for transcribe_audio function."""

    @pytest.mark.unit
    @patch('utils.asr_utils.subprocess.Popen')
    @patch('utils.asr_utils.torch.cuda.is_available')
    def test_transcribe_success(self, mock_cuda, mock_popen, temp_dir):
        """Test successful audio transcription."""
        mock_cuda.return_value = False

        # Setup mock process
        mock_process = Mock()
        mock_process.communicate.return_value = (b"", b"")
        mock_popen.return_value = mock_process

        audio_path = os.path.join(temp_dir, "test.mp3")
        json_path = os.path.join(temp_dir, "test.json")

        # Create mock JSON output
        mock_data = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "Hello world"}
            ]
        }

        with open(json_path, 'w') as f:
            json.dump(mock_data, f)

        result = transcribe_audio(audio_path)

        assert result is not None
        assert "segments" in result
        assert len(result["segments"]) == 1
        mock_popen.assert_called_once()

    @pytest.mark.unit
    @patch('utils.asr_utils.subprocess.Popen')
    @patch('utils.asr_utils.torch.cuda.is_available')
    def test_transcribe_with_cuda(self, mock_cuda, mock_popen, temp_dir):
        """Test transcription with CUDA available."""
        mock_cuda.return_value = True

        mock_process = Mock()
        mock_process.communicate.return_value = (b"", b"")
        mock_popen.return_value = mock_process

        audio_path = os.path.join(temp_dir, "test.mp3")
        json_path = os.path.join(temp_dir, "test.json")

        with open(json_path, 'w') as f:
            json.dump({"segments": []}, f)

        result = transcribe_audio(audio_path)

        # Check that CUDA was used in command
        call_args = mock_popen.call_args[0][0]
        assert "cuda" in call_args

    @pytest.mark.unit
    @patch('utils.asr_utils.subprocess.Popen')
    @patch('utils.asr_utils.torch.cuda.is_available')
    def test_transcribe_no_json_output(self, mock_cuda, mock_popen, temp_dir):
        """Test transcription when JSON file is not created."""
        mock_cuda.return_value = False

        mock_process = Mock()
        mock_process.communicate.return_value = (b"", b"")
        mock_popen.return_value = mock_process

        audio_path = os.path.join(temp_dir, "test.mp3")

        # Don't create JSON file
        result = transcribe_audio(audio_path)

        assert result is None

    @pytest.mark.unit
    @patch('utils.asr_utils.subprocess.Popen')
    def test_transcribe_error(self, mock_popen, temp_dir):
        """Test transcription with subprocess error."""
        mock_popen.side_effect = Exception("Subprocess error")

        audio_path = os.path.join(temp_dir, "test.mp3")
        result = transcribe_audio(audio_path)

        assert result is None

    @pytest.mark.unit
    @patch('utils.asr_utils.subprocess.Popen')
    @patch('utils.asr_utils.torch.cuda.is_available')
    def test_transcribe_stderr_output(self, mock_cuda, mock_popen, temp_dir):
        """Test transcription with stderr output."""
        mock_cuda.return_value = False

        mock_process = Mock()
        mock_process.communicate.return_value = (b"", b"Warning: Some warning message")
        mock_popen.return_value = mock_process

        audio_path = os.path.join(temp_dir, "test.mp3")
        json_path = os.path.join(temp_dir, "test.json")

        with open(json_path, 'w') as f:
            json.dump({"segments": []}, f)

        result = transcribe_audio(audio_path)

        # Should still succeed despite stderr
        assert result is not None


class TestFormatTranscriptToSrt:
    """Tests for format_transcript_to_srt function."""

    @pytest.mark.unit
    def test_format_success(self, temp_dir):
        """Test successful SRT formatting."""
        transcript_data = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "Hello world"},
                {"start": 2.5, "end": 5.0, "text": "This is a test"}
            ]
        }

        srt_path = os.path.join(temp_dir, "output.srt")
        result = format_transcript_to_srt(transcript_data, srt_path)

        assert result == srt_path
        assert os.path.exists(srt_path)

        # Verify SRT content
        with open(srt_path, 'r') as f:
            content = f.read()
            assert "1\n" in content
            assert "00:00:00,000 --> 00:00:02,500" in content
            assert "Hello world" in content
            assert "2\n" in content
            assert "This is a test" in content

    @pytest.mark.unit
    def test_format_none_data(self, temp_dir):
        """Test SRT formatting with None data."""
        srt_path = os.path.join(temp_dir, "output.srt")
        result = format_transcript_to_srt(None, srt_path)

        assert result is None

    @pytest.mark.unit
    def test_format_empty_segments(self, temp_dir):
        """Test SRT formatting with empty segments."""
        transcript_data = {"segments": []}

        srt_path = os.path.join(temp_dir, "output.srt")
        result = format_transcript_to_srt(transcript_data, srt_path)

        assert result is None

    @pytest.mark.unit
    def test_format_no_segments_key(self, temp_dir):
        """Test SRT formatting without segments key."""
        transcript_data = {"text": "Some text"}

        srt_path = os.path.join(temp_dir, "output.srt")
        result = format_transcript_to_srt(transcript_data, srt_path)

        assert result is None

    @pytest.mark.unit
    def test_format_time_conversion(self, temp_dir):
        """Test SRT time conversion accuracy."""
        transcript_data = {
            "segments": [
                {"start": 3661.123, "end": 3725.456, "text": "Test"}
            ]
        }

        srt_path = os.path.join(temp_dir, "output.srt")
        result = format_transcript_to_srt(transcript_data, srt_path)

        assert result == srt_path

        with open(srt_path, 'r') as f:
            content = f.read()
            # 3661.123 seconds = 1 hour, 1 minute, 1 second, 123 milliseconds
            assert "01:01:01,123" in content
            # 3725.456 seconds = 1 hour, 2 minutes, 5 seconds, 456 milliseconds
            assert "01:02:05,456" in content

    @pytest.mark.unit
    def test_format_creates_directory(self, temp_dir):
        """Test that SRT formatting creates necessary directories."""
        nested_path = os.path.join(temp_dir, "nested", "dir", "output.srt")

        transcript_data = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Test"}
            ]
        }

        result = format_transcript_to_srt(transcript_data, nested_path)

        assert result == nested_path
        assert os.path.exists(nested_path)
        assert os.path.exists(os.path.dirname(nested_path))

    @pytest.mark.unit
    def test_format_strips_whitespace(self, temp_dir):
        """Test that text whitespace is stripped."""
        transcript_data = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "  Hello world  \n"}
            ]
        }

        srt_path = os.path.join(temp_dir, "output.srt")
        result = format_transcript_to_srt(transcript_data, srt_path)

        with open(srt_path, 'r') as f:
            content = f.read()
            # Text should be stripped
            assert "Hello world\n\n" in content
            assert "  Hello world" not in content

    @pytest.mark.unit
    def test_format_exception_handling(self, temp_dir):
        """Test SRT formatting with malformed data."""
        transcript_data = {
            "segments": [
                {"start": 0.0}  # Missing 'end' and 'text'
            ]
        }

        srt_path = os.path.join(temp_dir, "output.srt")
        result = format_transcript_to_srt(transcript_data, srt_path)

        assert result is None
