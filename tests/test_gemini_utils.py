"""
Unit tests for gemini_utils module.
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import base64

from utils.gemini_utils import (
    extract_timestamp,
    generate_captions_and_narrations
)


class TestExtractTimestampGemini:
    """Tests for extract_timestamp function in gemini_utils."""

    @pytest.mark.unit
    def test_extract_valid_timestamp(self):
        """Test extracting timestamp from valid filename."""
        assert extract_timestamp("keyframe_0_3.50.jpg") == 3.50
        assert extract_timestamp("keyframe_5_120.75.jpg") == 120.75
        assert extract_timestamp("keyframe_10_0.jpg") == 0.0

    @pytest.mark.unit
    def test_extract_timestamp_no_decimal(self):
        """Test timestamp extraction without decimal."""
        assert extract_timestamp("keyframe_0_5.jpg") == 5.0

    @pytest.mark.unit
    def test_extract_timestamp_invalid(self):
        """Test timestamp extraction from invalid filename."""
        assert extract_timestamp("invalid.jpg") is None
        assert extract_timestamp("keyframe.jpg") is None


class TestGenerateCaptionsAndNarrations:
    """Tests for generate_captions_and_narrations function."""

    @pytest.mark.unit
    @patch('utils.gemini_utils.openai_client')
    @patch('utils.gemini_utils.create_sequence_image_for_segment')
    def test_generate_with_gpt(
        self, mock_create_sequence, mock_openai_client, temp_dir, mock_keyframes_folder
    ):
        """Test narration generation with GPT-4o."""
        # Setup mock responses
        mock_create_sequence.return_value = os.path.join(temp_dir, "segment_0-2.jpg")

        # Create a test image for base64 encoding
        test_image_path = os.path.join(temp_dir, "segment_0-2.jpg")
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(test_image_path)

        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content="<BEGIN SEGMENT>\nFrame Numbers: [0, 1, 2]\nTimestamp: 0.00-3.00\nNarration: Test narration for segment one.\n<END SEGMENT>"
            )
        )]
        mock_openai_client.chat.completions.create.return_value = mock_response

        caption_prompt = "Generate narrations for keyframes. Max {max_words_total} words."
        narration_prompt = "Create engaging narration."
        video_duration = 10.0

        script_data, api_responses = generate_captions_and_narrations(
            mock_keyframes_folder,
            caption_prompt,
            narration_prompt,
            video_duration,
            model_choice="gpt",
            style_instructions="Be enthusiastic",
            keyframes_per_segment=3
        )

        assert len(script_data) > 0
        assert len(api_responses) > 0
        assert script_data[0]["narration"] == "Test narration for segment one."
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.unit
    @patch('utils.gemini_utils.anthropic_client')
    @patch('utils.gemini_utils.create_sequence_image_for_segment')
    def test_generate_with_claude(
        self, mock_create_sequence, mock_anthropic_client, temp_dir, mock_keyframes_folder
    ):
        """Test narration generation with Claude."""
        # Setup mock responses
        mock_create_sequence.return_value = os.path.join(temp_dir, "segment_0-2.jpg")

        # Create a test image
        test_image_path = os.path.join(temp_dir, "segment_0-2.jpg")
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(test_image_path)

        mock_response = Mock()
        mock_response.content = [Mock(
            text="<BEGIN SEGMENT>\nFrame Numbers: [0, 1, 2]\nTimestamp: 0.00-3.00\nNarration: Claude generated narration.\n<END SEGMENT>"
        )]
        mock_anthropic_client.messages.create.return_value = mock_response

        caption_prompt = "Generate narrations for keyframes."
        narration_prompt = "Create engaging narration."
        video_duration = 10.0

        script_data, api_responses = generate_captions_and_narrations(
            mock_keyframes_folder,
            caption_prompt,
            narration_prompt,
            video_duration,
            model_choice="Claude",
            style_instructions="Be professional",
            keyframes_per_segment=3
        )

        assert len(script_data) > 0
        assert len(api_responses) > 0
        assert script_data[0]["narration"] == "Claude generated narration."
        mock_anthropic_client.messages.create.assert_called_once()

    @pytest.mark.unit
    @patch('utils.gemini_utils.openai_client', None)
    def test_generate_no_openai_key(self, mock_keyframes_folder):
        """Test error when OpenAI key is missing."""
        script_data, api_responses = generate_captions_and_narrations(
            mock_keyframes_folder,
            "prompt",
            "narration",
            10.0,
            model_choice="gpt",
            keyframes_per_segment=3
        )

        assert script_data == []
        assert api_responses == []

    @pytest.mark.unit
    @patch('utils.gemini_utils.anthropic_client', None)
    def test_generate_no_claude_key(self, mock_keyframes_folder):
        """Test error when Claude key is missing."""
        script_data, api_responses = generate_captions_and_narrations(
            mock_keyframes_folder,
            "prompt",
            "narration",
            10.0,
            model_choice="Claude",
            keyframes_per_segment=3
        )

        assert script_data == []
        assert api_responses == []

    @pytest.mark.unit
    def test_generate_unsupported_model(self, mock_keyframes_folder):
        """Test error with unsupported model choice."""
        script_data, api_responses = generate_captions_and_narrations(
            mock_keyframes_folder,
            "prompt",
            "narration",
            10.0,
            model_choice="InvalidModel",
            keyframes_per_segment=3
        )

        assert script_data == []
        assert api_responses == []

    @pytest.mark.unit
    def test_generate_no_keyframes(self, temp_dir):
        """Test generation with no keyframes."""
        empty_folder = os.path.join(temp_dir, "empty")
        os.makedirs(empty_folder, exist_ok=True)

        script_data, api_responses = generate_captions_and_narrations(
            temp_dir,
            "prompt",
            "narration",
            10.0,
            model_choice="gpt",
            keyframes_per_segment=3
        )

        assert script_data == []
        assert api_responses == []

    @pytest.mark.unit
    @patch('utils.gemini_utils.openai_client')
    @patch('utils.gemini_utils.create_sequence_image_for_segment')
    def test_generate_with_retry_logic(
        self, mock_create_sequence, mock_openai_client, temp_dir, mock_keyframes_folder
    ):
        """Test API retry logic on failure."""
        mock_create_sequence.return_value = os.path.join(temp_dir, "segment_0-2.jpg")

        # Create test image
        test_image_path = os.path.join(temp_dir, "segment_0-2.jpg")
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(test_image_path)

        # First two calls fail, third succeeds
        mock_openai_client.chat.completions.create.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            Mock(choices=[Mock(message=Mock(content="<BEGIN SEGMENT>\nFrame Numbers: [0, 1, 2]\nTimestamp: 0.00-3.00\nNarration: Success after retry.\n<END SEGMENT>"))])
        ]

        script_data, api_responses = generate_captions_and_narrations(
            mock_keyframes_folder,
            "prompt",
            "narration",
            10.0,
            model_choice="gpt",
            keyframes_per_segment=3
        )

        assert len(script_data) > 0
        assert script_data[0]["narration"] == "Success after retry."
        assert mock_openai_client.chat.completions.create.call_count == 3

    @pytest.mark.unit
    @patch('utils.gemini_utils.openai_client')
    @patch('utils.gemini_utils.create_sequence_image_for_segment')
    def test_generate_all_retries_fail(
        self, mock_create_sequence, mock_openai_client, temp_dir, mock_keyframes_folder
    ):
        """Test when all API retries fail."""
        mock_create_sequence.return_value = os.path.join(temp_dir, "segment_0-2.jpg")

        # Create test image
        test_image_path = os.path.join(temp_dir, "segment_0-2.jpg")
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(test_image_path)

        # All calls fail
        mock_openai_client.chat.completions.create.side_effect = Exception("Persistent API Error")

        script_data, api_responses = generate_captions_and_narrations(
            mock_keyframes_folder,
            "prompt",
            "narration",
            10.0,
            model_choice="gpt",
            keyframes_per_segment=3
        )

        assert script_data == []
        assert api_responses == []
        assert mock_openai_client.chat.completions.create.call_count == 3
