import gradio as gr
import os
import datetime
import logging
import json
import shutil
import requests  # For API calls
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
import numpy as np

# Utility functions
from utils.video_utils import (
    extract_key_frames,
    create_video_with_audio,
    make_srt_from_script_data,
    add_subtitles_to_video
)
from utils.gemini_utils import generate_captions_and_narrations
from utils.tts_utils import generate_text_to_speech
from config import verify_api_keys, ELEVENLABS_API_KEY

#Logo path
LOGO_PATH = "logo.png"

def fetch_elevenlabs_voices():
    """
    Fetch the list of available voices from the ElevenLabs API.
    Returns a list of tuples: (voice_id, voice_name)
    """
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY
    }
    url = "https://api.elevenlabs.io/v1/voices"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            voices = data.get("voices", [])
            # Create a list of tuples (voice_id, voice_name)
            voice_options = [(voice["voice_id"], voice["name"]) for voice in voices]
            return voice_options
        else:
            logging.error(f"Failed to fetch voices from ElevenLabs API: {response.status_code} {response.text}")
            return []
    except Exception as e:
        logging.error(f"Error fetching voices from ElevenLabs: {e}")
        return []

def fetch_elevenlabs_models():
    """
    Fetch the list of available models from the ElevenLabs API.
    Returns a list of tuples: (model_id, model_name)
    """
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY
    }
    url = "https://api.elevenlabs.io/v1/models"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            models = response.json()  # Response is a list, not a dict
            # Create a list of tuples (model_id, model_name)
            model_options = [(model["model_id"], model.get(model["model_id"])) for model in models]
            return model_options
        else:
            logging.error(f"Failed to fetch models from ElevenLabs API: {response.status_code} {response.text}")
            return []
    except Exception as e:
        logging.error(f"Error fetching models from ElevenLabs: {e}")
        return []

def fetch_google_tts_voices():
    """
    Fetches available voices from the Google Cloud TTS API.
    Returns a list of tuples: (voice_name, language_codes)
    """
    try:
        from google.cloud import texttospeech
        client = texttospeech.TextToSpeechClient()
        response = client.list_voices()  # Use the list_voices method
        voices = []
        for voice in response.voices:
            for language_code in voice.language_codes:
                voices.append((voice.name, language_code))  # Store voice name and language code
        return voices
    except Exception as e:
        logging.error(f"Error fetching voices from Google Cloud TTS: {e}")
        return []

def process_video(
    video_file,
    keyframe_extraction_method,
    keyframe_threshold,
    num_keyframes,
    tts_engine,
    elevenlabs_voice_id,  # New parameter
    elevenlabs_model_id,  # New parameter
    google_tts_voice_name, # Parameter for selected Google TTS voice name
    model_choice,
    speed_factor,
    style_instructions,
    burn_subtitles,
    keyframes_per_segment,
    original_audio_volume, # New parameter
    progress=gr.Progress(track_tqdm=True),
    processing_state=None
):

    """
    Main function to orchestrate video processing, captioning, TTS,
    with optional burned-in subtitles using MoviePy's file_to_subtitles.
    """
    # set processing status
    processing_state = "processing"

    # Create unique project folder
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    project_folder = os.path.join("output", f"project_{current_time}")
    os.makedirs(project_folder, exist_ok=True)

    # Logging
    log_file = os.path.join(project_folder, "process.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
    )

    logging.info("Starting video processing.")

    if not verify_api_keys():
        logging.error("One or more API keys are missing.")
        return "Error: Missing API keys in the .env file.", None, None, None, processing_state

    # Retrieve path from Gradio file object
    if hasattr(video_file, "name"):
        input_video_path = video_file.name
    else:
        input_video_path = video_file  # if it's already a path string

    video_path = os.path.join(project_folder, os.path.basename(input_video_path))

    # Copy video file
    try:
        with open(input_video_path, "rb") as f_in:
            with open(video_path, "wb") as f_out:
                f_out.write(f_in.read())
        logging.info(f"Copied video to: {video_path}")
    except Exception as e:
        logging.error(f"Error copying video: {e}")
        return "Error copying video file.", None, None, None, processing_state

     # Load prompts
    if model_choice == "gpt":
        with open("prompts/multi_image_segmented_prompt.txt", "r", encoding="utf-8") as f:
            base_caption_prompt = f.read().strip()
        base_tts_prompt = None
    else:
        with open("prompts/caption_prompt.txt", "r", encoding="utf-8") as f:
            base_caption_prompt = f.read().strip()
        with open("prompts/tts_prompt.txt", "r", encoding="utf-8") as f:
            base_tts_prompt = f.read().strip()

    # Extract Key Frames
    if keyframe_extraction_method == "Threshold":
        keyframe_indices, keyframe_timestamps = extract_key_frames(
            video_path, project_folder, keyframe_threshold=keyframe_threshold, keyframes_per_segment=keyframes_per_segment
        )
    else:
        keyframe_indices, keyframe_timestamps = extract_key_frames(
            video_path, project_folder, num_keyframes=num_keyframes, keyframes_per_segment=keyframes_per_segment
        )

    if not keyframe_indices:
        logging.error("Error: No key frames could be extracted.")
        return "Error: No key frames could be extracted. Please make sure the video is valid.", None, None, None, processing_state
    logging.info(f"Extracted {len(keyframe_indices)} key frames.")

    # Get video duration
    video_clip = VideoFileClip(video_path)
    video_duration = video_clip.duration
    video_clip.close()

    # Generate Captions and Narrations
    keyframes_folder = os.path.join(project_folder, "keyframes")
    script_data, api_responses = generate_captions_and_narrations(
        project_folder,
        base_caption_prompt,
        base_tts_prompt,
        video_duration,
        model_choice,
        style_instructions,
        keyframes_per_segment  # Pass the new parameter
    )

    if not script_data:
        return "Error: No script data could be generated.", None, None, None, processing_state

    # Save API responses
    responses_filename = os.path.join(project_folder, "api_responses.json")
    with open(responses_filename, "w", encoding="utf-8") as f:
        json.dump(api_responses, f, indent=2, ensure_ascii=False)  # ensure_ascii=False
    logging.info(f"Saved responses to {responses_filename}")

    # Save narration script
    script_filename = os.path.join(project_folder, "narration_script.json")
    with open(script_filename, "w", encoding="utf-8") as f:
        json.dump(script_data, f, indent=2, ensure_ascii=False)  # ensure_ascii=False
    logging.info(f"Saved narration script to {script_filename}")

    # Generate TTS for each narration segment
    audio_files = []
    for idx, entry in enumerate(script_data):
        narration_text = entry["narration"]
        audio_filename = f"narration_{idx}.mp3"
        audio_path = os.path.join(project_folder, audio_filename)
        audio_files.append(audio_path)

        if not generate_text_to_speech(
            narration_text,
            audio_path,
            tts_engine,
            speed_factor,
            elevenlabs_voice_id,  # Pass the selected voice ID
            elevenlabs_model_id,  # Pass the selected model ID
            google_tts_voice_name  # Pass the selected voice name
        ):
            logging.error(f"Error generating TTS for segment {idx}.")
            return f"Error generating TTS for segment {idx}.", None, None, None, processing_state

    timestamps = [float(entry["timestamp"].split('-')[0]) for entry in script_data]
    output_video_path = os.path.join(project_folder, "output_video.mp4")
    final_video_path = create_video_with_audio(video_path, audio_files, timestamps, output_video_path, original_audio_volume)  # Pass original_audio_volume

    if not final_video_path:
        return "Error: Could not combine video with audio.", None, None, None, processing_state

    # (Optional) Burn subtitles with MoviePy
    final_result_path = final_video_path
    if burn_subtitles:
        srt_path = os.path.join(project_folder, "output_subtitles.srt")
        if make_srt_from_script_data(script_data, srt_path):
            # Use add_subtitles_to_video with burn_in=True
            subbed_video_path = os.path.join(project_folder, "output_video_with_subs.mp4")
            sub_done = add_subtitles_to_video(final_video_path, srt_path, subbed_video_path, burn_in=True)
            if sub_done:
                final_result_path = subbed_video_path  # subbed_video_path

    # Return final result
    last_audio_path = audio_files[-1] if audio_files else None
    processing_state = "done"
    return final_result_path, script_filename, last_audio_path, get_sequence_image_path(project_folder), processing_state

def get_sequence_image_path(project_folder):
     # Extract keyframes sequence image
    return os.path.join(project_folder, "keyframes_sequence.jpg") if os.path.exists(os.path.join(project_folder, "keyframes_sequence.jpg")) else None

if __name__ == "__main__":
    # Fetch voices and models on startup
    initial_google_tts_voices = fetch_google_tts_voices()
    initial_elevenlabs_voices = fetch_elevenlabs_voices()
    initial_elevenlabs_models = fetch_elevenlabs_models()

    # Define default values for the dropdowns
    default_elevenlabs_voice = "David Attenboro (DBZ3Yn0vZCfYBbj7kyCY)"
    default_elevenlabs_model = "eleven_flash_v2_5"
    default_google_tts_voice = "he-IL-Standard-B (he-IL)"

    with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {background-color: #f0f2f5;}") as demo:
        # State to store whether processing has started
        processing_state = gr.State("idle")
        
        # Add logo
        if os.path.exists(LOGO_PATH):
            logo = gr.Image(LOGO_PATH, show_label=False, container=False, width=200)
        else:
            logo = gr.Markdown("## VideoMaster")

        with gr.Row():
            with gr.Column(scale=3):  # Input elements column (wider)
                with gr.Column():
                    with gr.Row():
                        gr.Markdown('<p style="font-size: 15px; color: black">Upload a video to get started</p>')
                    with gr.Row():
                        uploaded_video_preview = gr.Video(label="Uploaded Video", width=150, height=100)  # small preview
                    with gr.Row():
                        video_input = gr.File(file_count='single', file_types=["video"], label="Upload/Drag a Video")

                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Column():
                            gr.Markdown("## Keyframe Extraction Settings")
                            with gr.Row():
                                keyframe_extraction_method_input = gr.Radio(
                                    label="Keyframe Extraction Method",
                                    choices=["Threshold", "Number"],
                                    value="Number"
                                )
                            with gr.Row():
                                keyframe_threshold_input = gr.Slider(
                                    label="Keyframe Change Threshold",
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    value=30,
                                    visible=False
                                )
                                num_keyframes_input = gr.Slider(
                                    label="Number of Keyframes",
                                    minimum=2,
                                    maximum=50,
                                    step=1,
                                    value=20,
                                    visible=True
                                )
                            with gr.Row():
                                keyframes_per_segment_input = gr.Slider(
                                    label="Keyframes per Segment",
                                    minimum=1,
                                    maximum=10,
                                    step=1,
                                    value=3
                                )

                        gr.Markdown("## Advanced Settings")
                        with gr.Column():
                            burn_subtitles_input = gr.Checkbox(label="Burn Subtitles?", value=False, visible=False)
                            original_audio_volume_input = gr.Slider(
                                label="Original Audio Volume",
                                minimum=0.0,
                                maximum=0.4,
                                step=0.01,
                                value=0.02
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("## Model Settings")
                        with gr.Column():
                            with gr.Row():
                                model_choice_input = gr.Radio(
                                    label="Model Choice",
                                    choices=["Claude", "gpt"],
                                    value="Claude"
                                )
                                speed_factor_input = gr.Slider(
                                    label="Playback Speed",
                                    minimum=0.5,
                                    maximum=2.0,
                                    step=0.1,
                                    value=1.0
                                )
                            style_instructions_input = gr.Textbox(
                                label="Style Instructions",
                                placeholder="e.g. 'Narrate like a sports commentator' or 'Use a reflective, internal monologue style'",
                                value=""
                            )
                        
                        with gr.Column():
                            gr.Markdown("## Text-to-Speech Engine Settings")
                            with gr.Column():
                                with gr.Row():
                                    tts_engine_input = gr.Radio(
                                        label="TTS Engine",
                                        choices=["Google Cloud TTS", "ElevenLabs"],
                                        value="ElevenLabs"
                                    )
                                with gr.Column():
                                    elevenlabs_voice_input = gr.Dropdown(
                                        label="ElevenLabs Voice",
                                        choices=[f"{voice_name} ({voice_id})" for voice_id, voice_name in initial_elevenlabs_voices],
                                        value=default_elevenlabs_voice,
                                        allow_custom_value=True,
                                        visible=True
                                    )
                                    elevenlabs_model_input = gr.Dropdown(
                                        label="ElevenLabs Model",
                                        choices=initial_elevenlabs_models,
                                        value=default_elevenlabs_model,
                                        allow_custom_value=True,
                                        visible=True
                                    )
                                    google_tts_voice_input = gr.Dropdown(
                                        label="Google TTS Voice",
                                        choices=[f"{voice_name} ({lang_code})" for voice_name, lang_code in initial_google_tts_voices],
                                        value=default_google_tts_voice,
                                        allow_custom_value=True,
                                        visible=False
                                    )

                with gr.Column():
                    gr.Markdown("## Examples")
                    gr.Examples(
                        examples=[
                            ["examples/vid1.mp4", "Number", 20, None, "ElevenLabs", "David Attenboro (DBZ3Yn0vZCfYBbj7kyCY)", 
                             "eleven_flash_v2_5", None, "gpt", 1.0, "make it funny", False, 3, 0.02],
                            ["examples/vid2.mp4", "Threshold", 20, None, "ElevenLabs", "David Attenboro (DBZ3Yn0vZCfYBbj7kyCY)",
                             "eleven_flash_v2_5", "gpt", 1.0, "you are a FOULED MOUTH gangsta rapper using NONSTOP profanity", True, 3, 0.02],
                            ["examples/vid3.mp4", "Number", 10, None, "ElevenLabs", "David Attenboro (DBZ3Yn0vZCfYBbj7kyCY)",
                             "eleven_flash_v2_5", None, "gpt", 1.0, "you are a gangsta rapper using profanity", False, 3, 0.02]
                        ],
                        inputs=[
                            video_input,
                            keyframe_extraction_method_input,
                            keyframe_threshold_input,
                            num_keyframes_input,
                            tts_engine_input,
                            elevenlabs_voice_input,
                            elevenlabs_model_input,
                            google_tts_voice_input,
                            model_choice_input,
                            speed_factor_input,
                            style_instructions_input,
                            burn_subtitles_input,
                            keyframes_per_segment_input,
                            original_audio_volume_input
                        ],
                    )

            with gr.Column(scale=1):  # Output elements column (narrower, on right)
                processing_feedback = gr.Markdown("")
                processed_video_output = gr.Video(label="Processed Video with Narration")
                submit_button = gr.Button("Process Video", variant="primary", icon="logo.png")

        def show_processing_output(state, video_path=None):
            """Show processing status and video output."""
            if state == "processing":
                return gr.Markdown("Processing... Please wait, this can take a while."), None
            elif state == "done":
                if video_path:
                    return gr.Markdown("Done! Processing is completed and the video is available."), video_path
                else:
                    return gr.Markdown(""), None
            elif state == "idle":
                return gr.Markdown("Ready! Please provide a video input and click the \"Process Video\" Button"), None
            else:
                return gr.Markdown("Error Occurred"), None

        def update_video_preview(video_input):
            """Update the video preview when a new video is uploaded."""
            if video_input:
                return gr.update(width=150, height=100, value=video_input)
            return gr.update(value=None)

        # Event handlers
        keyframe_extraction_method_input.change(
            fn=lambda method: (gr.update(visible=method == "Threshold"), gr.update(visible=method == "Number")),
            inputs=keyframe_extraction_method_input,
            outputs=[keyframe_threshold_input, num_keyframes_input]
        )

        tts_engine_input.change(
            fn=lambda engine: (
                gr.update(visible=engine == "Google Cloud TTS"),
                gr.update(visible=engine == "ElevenLabs"),
                gr.update(visible=engine == "ElevenLabs")
            ),
            inputs=tts_engine_input,
            outputs=[google_tts_voice_input, elevenlabs_voice_input, elevenlabs_model_input]
        )

        video_input.change(
            fn=update_video_preview,
            inputs=video_input,
            outputs=uploaded_video_preview
        )

        submit_button.click(
            fn=process_video,
            inputs=[
                video_input,
                keyframe_extraction_method_input,
                keyframe_threshold_input,
                num_keyframes_input,
                tts_engine_input,
                elevenlabs_voice_input,
                elevenlabs_model_input,
                google_tts_voice_input,
                model_choice_input,
                speed_factor_input,
                style_instructions_input,
                burn_subtitles_input,
                keyframes_per_segment_input,
                original_audio_volume_input,
                processing_state
            ],
            outputs=[
                processed_video_output,  # Output video path
                gr.State(),  # script_filename
                gr.State(),  # last_audio_path
                gr.State(),  # sequence_image_path
                processing_state
            ],
            show_progress="full"
        )

        # Monitor processing state changes
        processing_state.change(
            fn=show_processing_output,
            inputs=[processing_state, processed_video_output],
            outputs=[processing_feedback, processed_video_output],
            show_progress=False
        )

        # Update video display when processing is done
        def update_video_display(state, video_path):
            if state == "done" and video_path:
                return video_path
            return None

        processing_state.change(
            fn=update_video_display,
            inputs=[processing_state, processed_video_output],
            outputs=processed_video_output,
            show_progress=False
        )

        # Initialize the UI
        demo.load(
            fn=lambda: ("idle", None, gr.Markdown("Ready to process videos"), None),
            outputs=[processing_state, processed_video_output, processing_feedback, uploaded_video_preview]
        )

    # Launch the demo
    demo.launch(debug=True, favicon_path="favicon.ico", inbrowser=True)