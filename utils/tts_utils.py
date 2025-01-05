import os
from config import ELEVENLABS_API_KEY
import logging
from pydub import AudioSegment
from google.cloud import texttospeech

def generate_text_to_speech(narration_text, output_path, tts_engine, speed_factor=1.0, elevenlabs_voice_id=None, elevenlabs_model_id=None, google_tts_voice_name=None):
    """
    Generate an audio file from text using the selected TTS engine.
    Supports:
        - Google Cloud TTS (with voice name selection)
        - ElevenLabs (with voice ID and model ID selection)
    Adjusts playback speed (tempo) using pydub.
    """
    logging.info(f"Starting TTS generation with engine: {tts_engine} | speed_factor: {speed_factor}")

    if not narration_text:
        logging.warning("No narration text available.")
        return None

    # Function to adjust audio speed
    def adjust_audio_speed(file_path, factor):
        if abs(factor - 1.0) < 1e-3:
            return
        sound = AudioSegment.from_file(file_path)
        sound_with_speed = sound._spawn(
            sound.raw_data,
            overrides={
                "frame_rate": int(sound.frame_rate * factor)
            }
        )
        sound_with_speed = sound_with_speed.set_frame_rate(44100)
        sound_with_speed.export(file_path, format="mp3")
        logging.info(f"Audio speed adjusted by factor {factor} and re-exported to {file_path}")

    if tts_engine == "Google Cloud TTS":
        try:
            # Initialize Google Cloud TTS client
            client = texttospeech.TextToSpeechClient()

            # Set the text input to be synthesized
            synthesis_input = texttospeech.SynthesisInput(text=narration_text)

            # Build the voice request
            if google_tts_voice_name:
                logging.info(f"Received Google TTS voice_name: {google_tts_voice_name}")
                try:
                    voice_name = google_tts_voice_name.split(" (")[0]
                    language_code = google_tts_voice_name.split(" (")[-1].strip(")")
                    logging.info(f"Extracted voice_name: {voice_name}, language_code: {language_code}")
                except (IndexError, AttributeError) as e:
                    logging.error(f"Could not parse: '{google_tts_voice_name}'.")
                    logging.error(f"Error: {e}")
                    raise ValueError("Invalid Google TTS voice format. Ensure it is 'Voice Name (language_code)'.")
            else:
                voice_name = "en-US-Studio-O"  # Choose your default voice
                language_code = "en-US"  # Default language code
                logging.info(f"Using default Google Cloud TTS voice: {voice_name} ({language_code})")

            # Configure voice parameters
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )

            # Configure audio settings
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.25  # Set speaking rate to 1.5

            )

            # Perform the text-to-speech request
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            # Save the audio to output_path
            with open(output_path, "wb") as out:
                out.write(response.audio_content)
                logging.info(f"Audio content written to file {output_path}")

            # Adjust speed if needed
            adjust_audio_speed(output_path, speed_factor)

            return output_path

        except Exception as e:
            logging.error(f"Error with Google Cloud TTS: {e}")
            return None
        
    elif tts_engine == "ElevenLabs":
        try:
            if not ELEVENLABS_API_KEY:
                logging.error("Error: ELEVENLABS_API_KEY not found in environment variables.")
                return None

            # Import the ElevenLabs client
            from elevenlabs import ElevenLabs  # Correct import for ElevenLabs SDK

            # Instantiate the client with your API key
            client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

            # Use the selected voice ID if provided, otherwise use default voice
            if elevenlabs_voice_id:
                # Extract voice_id from the "Voice Name (voice_id)" format
                voice_id = elevenlabs_voice_id.split(" ")[-1].strip("()")
                voice = voice_id
            else:
                voice = "obami"  # Default voice ID

            # Use the selected model ID if provided, otherwise use default model
            model = elevenlabs_model_id if elevenlabs_model_id else "eleven_multilingual_v2"

            # Generate the audio
            audio_stream = client.generate(
                text=narration_text,
                voice=voice,
                model=model
            )
            
            audio_bytes = b"".join([chunk for chunk in audio_stream])

            # Save the audio to output_path
            with open(output_path, "wb") as f:
                f.write(audio_bytes)

            # Adjust speed if needed
            adjust_audio_speed(output_path, speed_factor)

            logging.info(f"ElevenLabs audio generated at: {output_path}")
            return output_path

        except Exception as e:
            logging.error(f"Error with ElevenLabs TTS: {e}")
            return None

    else:
        logging.error(f"Error: Unsupported TTS engine '{tts_engine}'.")
        return None