import logging
import os
import subprocess
import json
import torch
from utils.logging_setup import setup_logging

log_file = setup_logging(app_name="videofck_asr")
logger = logging.getLogger(__name__)

def transcribe_audio(audio_path):
    """Transcribes an audio file using Whisper."""
    try:
        logging.info(f"Starting transcription for: {audio_path}")

        command = [
            "whisper",
            audio_path,
            "--model", "medium",
            "--output_dir", os.path.dirname(audio_path),
            "--output_format", "json",
            "--word_timestamps", "True",
            "--device", "cuda" if torch.cuda.is_available() else "cpu"
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if stderr:
            error_message = stderr.decode('utf-8', errors='ignore')
            logging.error(f"Whisper stderr: {error_message}")

        output_json_path = os.path.splitext(audio_path)[0] + ".json"
        if os.path.exists(output_json_path):
            with open(output_json_path, 'r', encoding='utf-8') as f:
                transcription_data = json.load(f)
                logging.info("Successfully transcribed audio.")
                return transcription_data

        logging.error("Transcription JSON file not created.")
        return None

    except Exception as e:
        logging.exception(f"An error occurred during transcription of file {audio_path}: {e}")
        return None

def format_transcript_to_srt(transcript_data, srt_output_path):
    """Formats Whisper's JSON output to SRT."""
    try:
        def to_srt_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

        if not transcript_data:
            logging.error("No transcription data to format. Skipping SRT creation.")
            return None

        segments = transcript_data.get("segments", [])
        if not segments:
            logging.error("No segments in the transcription data.")
            return None

        os.makedirs(os.path.dirname(srt_output_path), exist_ok=True)  # Ensure directory exists

        with open(srt_output_path, 'w', encoding='utf-8') as srt_file:
            for i, segment in enumerate(segments):
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text'].strip()

                srt_file.write(f"{i + 1}\n")
                srt_file.write(f"{to_srt_time(start_time)} --> {to_srt_time(end_time)}\n")
                srt_file.write(f"{text}\n\n")

        logging.info(f"SRT file created at: {srt_output_path}")
        return srt_output_path

    except Exception as e:
        logging.exception("Error occurred while formatting transcript to SRT:")
        return None
