"""
Unit tests for tts_utils module.
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pydub import AudioSegment

from utils.tts_utils import generate_text_to_speech


class TestGenerateTextToSpeech:
    """Tests for generate_text_to_speech function."""

    @pytest.mark.unit
    def test_generate_empty_text(self, temp_dir):
        """Test TTS generation with empty text."""
        output_path = os.path.join(temp_dir, "output.mp3")
        result = generate_text_to_speech(
            "", output_path, "Google Cloud TTS"
        )
        assert result is None

    @pytest.mark.unit
    def test_generate_none_text(self, temp_dir):
        """Test TTS generation with None text."""
        output_path = os.path.join(temp_dir, "output.mp3")
        result = generate_text_to_speech(
            None, output_path, "Google Cloud TTS"
        )
        assert result is None

    @pytest.mark.unit
    @patch('utils.tts_utils.texttospeech.TextToSpeechClient')
    def test_generate_google_tts_success(self, mock_tts_client, temp_dir):
        """Test successful Google Cloud TTS generation."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.audio_content = b"fake audio content"
        mock_client.synthesize_speech.return_value = mock_response
        mock_tts_client.return_value = mock_client

        output_path = os.path.join(temp_dir, "output.mp3")

        # Create a dummy audio file for speed adjustment
        with patch('utils.tts_utils.AudioSegment.from_file') as mock_audio:
            mock_segment = Mock()
            mock_segment.frame_rate = 44100
            mock_segment.raw_data = b"data"
            mock_segment._spawn.return_value = mock_segment
            mock_segment.set_frame_rate.return_value = mock_segment
            mock_audio.return_value = mock_segment

            result = generate_text_to_speech(
                "Hello world",
                output_path,
                "Google Cloud TTS",
                speed_factor=1.0
            )

        assert result == output_path
        assert os.path.exists(output_path)
        mock_client.synthesize_speech.assert_called_once()

    @pytest.mark.unit
    @patch('utils.tts_utils.texttospeech.TextToSpeechClient')
    def test_generate_google_tts_with_voice_name(self, mock_tts_client, temp_dir):
        """Test Google TTS with custom voice name."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.audio_content = b"fake audio content"
        mock_client.synthesize_speech.return_value = mock_response
        mock_tts_client.return_value = mock_client

        output_path = os.path.join(temp_dir, "output.mp3")

        with patch('utils.tts_utils.AudioSegment.from_file'):
            result = generate_text_to_speech(
                "Hello world",
                output_path,
                "Google Cloud TTS",
                speed_factor=1.0,
                google_tts_voice_name="en-US-Studio-O (en-US)"
            )

        assert result == output_path
        mock_client.synthesize_speech.assert_called_once()

    @pytest.mark.unit
    @patch('utils.tts_utils.texttospeech.TextToSpeechClient')
    def test_generate_google_tts_error(self, mock_tts_client, temp_dir):
        """Test Google TTS with error."""
        mock_tts_client.side_effect = Exception("API Error")

        output_path = os.path.join(temp_dir, "output.mp3")
        result = generate_text_to_speech(
            "Hello world",
            output_path,
            "Google Cloud TTS"
        )

        assert result is None

    @pytest.mark.unit
    @patch('utils.tts_utils.ELEVENLABS_API_KEY', 'test_key')
    @patch('utils.tts_utils.ElevenLabs')
    def test_generate_elevenlabs_success(self, mock_elevenlabs, temp_dir):
        """Test successful ElevenLabs TTS generation."""
        # Setup mock
        mock_client = Mock()
        mock_client.generate.return_value = [b"fake", b"audio", b"content"]
        mock_elevenlabs.return_value = mock_client

        output_path = os.path.join(temp_dir, "output.mp3")

        with patch('utils.tts_utils.AudioSegment.from_file'):
            result = generate_text_to_speech(
                "Hello world",
                output_path,
                "ElevenLabs",
                speed_factor=1.0,
                elevenlabs_voice_id="Daniel (voice_id_123)",
                elevenlabs_model_id="eleven_multilingual_v2"
            )

        assert result == output_path
        assert os.path.exists(output_path)
        mock_client.generate.assert_called_once()

    @pytest.mark.unit
    @patch('utils.tts_utils.ELEVENLABS_API_KEY', None)
    def test_generate_elevenlabs_no_api_key(self, temp_dir):
        """Test ElevenLabs TTS without API key."""
        output_path = os.path.join(temp_dir, "output.mp3")
        result = generate_text_to_speech(
            "Hello world",
            output_path,
            "ElevenLabs"
        )

        assert result is None

    @pytest.mark.unit
    @patch('utils.tts_utils.ELEVENLABS_API_KEY', 'test_key')
    @patch('utils.tts_utils.ElevenLabs')
    def test_generate_elevenlabs_error(self, mock_elevenlabs, temp_dir):
        """Test ElevenLabs TTS with error."""
        mock_elevenlabs.side_effect = Exception("ElevenLabs API Error")

        output_path = os.path.join(temp_dir, "output.mp3")
        result = generate_text_to_speech(
            "Hello world",
            output_path,
            "ElevenLabs"
        )

        assert result is None

    @pytest.mark.unit
    def test_generate_unsupported_engine(self, temp_dir):
        """Test TTS generation with unsupported engine."""
        output_path = os.path.join(temp_dir, "output.mp3")
        result = generate_text_to_speech(
            "Hello world",
            output_path,
            "UnsupportedEngine"
        )

        assert result is None

    @pytest.mark.unit
    @patch('utils.tts_utils.texttospeech.TextToSpeechClient')
    @patch('utils.tts_utils.AudioSegment.from_file')
    def test_speed_adjustment(self, mock_audio_from_file, mock_tts_client, temp_dir):
        """Test audio speed adjustment."""
        # Setup TTS mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.audio_content = b"fake audio content"
        mock_client.synthesize_speech.return_value = mock_response
        mock_tts_client.return_value = mock_client

        # Setup audio segment mock
        mock_segment = Mock()
        mock_segment.frame_rate = 44100
        mock_segment.raw_data = b"data"
        spawned_segment = Mock()
        spawned_segment.set_frame_rate.return_value = spawned_segment
        mock_segment._spawn.return_value = spawned_segment
        mock_audio_from_file.return_value = mock_segment

        output_path = os.path.join(temp_dir, "output.mp3")

        result = generate_text_to_speech(
            "Hello world",
            output_path,
            "Google Cloud TTS",
            speed_factor=1.5
        )

        assert result == output_path
        # Speed adjustment should be called
        mock_audio_from_file.assert_called_once()
        mock_segment._spawn.assert_called_once()

    @pytest.mark.unit
    @patch('utils.tts_utils.texttospeech.TextToSpeechClient')
    def test_invalid_voice_format(self, mock_tts_client, temp_dir):
        """Test Google TTS with invalid voice format."""
        mock_client = Mock()
        mock_tts_client.return_value = mock_client
        mock_client.synthesize_speech.side_effect = ValueError("Invalid Google TTS voice format")

        output_path = os.path.join(temp_dir, "output.mp3")

        with pytest.raises(ValueError):
            generate_text_to_speech(
                "Hello world",
                output_path,
                "Google Cloud TTS",
                google_tts_voice_name="InvalidFormat"
            )
