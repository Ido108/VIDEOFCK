import logging
import os
from config import CLAUDE_API_KEY, OPENAI_API_KEY, LLAMA_MODEL_PATH
import anthropic
from openai import OpenAI
from llama_cpp import Llama
import time
import datetime
import re
import base64
from google.cloud import storage
from PIL import Image
from io import BytesIO
from utils.video_utils import create_sequence_image_for_segment

logger = logging.getLogger(__name__)

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def extract_timestamp(filename):
    """Extracts the timestamp from a keyframe filename."""
    match = re.search(r'_(\d+(?:\.\d+)?)\.jpg$', filename)
    if match:
        timestamp = float(match.group(1))
        logging.debug(f"Extracted timestamp {timestamp} from filename {filename}")
        return timestamp
    else:
        logging.warning(f"No timestamp found in filename {filename}")
        return None

# Initialize clients
if CLAUDE_API_KEY:
    anthropic_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
else:
    anthropic_client = None

# Default anthropic model; actual model may be overridden per request
anthropic_model_name = "claude-3-5-sonnet-20241022"

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None

# Initialize local Llama model if path is provided
if LLAMA_MODEL_PATH and os.path.exists(LLAMA_MODEL_PATH):
    try:
        llama_model = Llama(model_path=LLAMA_MODEL_PATH)
    except Exception as e:
        logging.error(f"Failed to load Llama model: {e}")
        llama_model = None
else:
    llama_model = None

def generate_captions_and_narrations(
    project_folder,
    caption_prompt_input,
    narration_prompt_input,
    video_duration,
    model_choice,
    style_instructions="",
    keyframes_per_segment=3
):
    """
    Generate captions and narrations using the sequence keyframe image for each segment.
    """
    logging.info("Generating segmented narrations for keyframes.")
    script_data = []
    api_responses = []

    words_per_second = 2.5
    max_words_total = int(video_duration * words_per_second)
    logging.info(f"Max words for video: {max_words_total}")

    # Get keyframe timestamps
    keyframes_folder = os.path.join(project_folder, "keyframes")
    keyframe_files = sorted(
        [f for f in os.listdir(keyframes_folder) if f.endswith('.jpg')],
        key=lambda x: extract_timestamp(x) or 0.0
    )

    if not keyframe_files:
        logging.error(f"No keyframe files found in: {keyframes_folder}")
        return script_data, api_responses

    # Create segments
    group_size = keyframes_per_segment
    step_size = group_size - 1
    segments = []
    total_frames = len(keyframe_files)
    idx = 0

    while idx <= total_frames - group_size:
        current_files = keyframe_files[idx:idx + group_size]
        frame_numbers = list(range(idx, idx + len(current_files)))

        # Extract timestamps from filenames
        timestamps = [extract_timestamp(f) for f in current_files]

        # Check if any timestamp extraction failed
        if None in timestamps or len(timestamps) != group_size:
            logging.warning(f"Skipping segment starting at index {idx} due to invalid timestamps.")
            idx += step_size
            continue

        start_time = timestamps[0]
        end_time = timestamps[-1]
        duration = end_time - start_time

        if duration <= 0:
            logging.warning(f"Invalid duration for segment starting at index {idx}. Skipping.")
            idx += step_size
            continue

        max_words = max(1, int(duration * words_per_second))

        # Create sequence image for the segment
        segment_image_path = create_sequence_image_for_segment(
            {"frame_numbers": frame_numbers, "start_time": start_time, "end_time": end_time},
            keyframes_folder,
            project_folder,
            keyframes_per_segment
        )

        segments.append({
            "frame_numbers": frame_numbers,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "max_words": max_words,
            "image_path": segment_image_path  # Store the path to the image
        })

        idx += step_size

    # Handle any remaining frames at the end
    if idx < total_frames:
        current_files = keyframe_files[idx:]
        frame_numbers = list(range(idx, idx + len(current_files)))
        timestamps = [extract_timestamp(f) for f in current_files]

        if None not in timestamps and len(timestamps) >= 2:
            start_time = timestamps[0]
            end_time = timestamps[-1]
            duration = end_time - start_time
            max_words = max(1, int(duration * words_per_second))

            # Create sequence image for the segment
            segment_image_path = create_sequence_image_for_segment(
                {"frame_numbers": frame_numbers, "start_time": start_time, "end_time": end_time},
                keyframes_folder,
                project_folder,
                keyframes_per_segment
            )

            segments.append({
                "frame_numbers": frame_numbers,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "max_words": max_words,
                "image_path": segment_image_path
            })

    if not segments:
        logging.error("No segments could be created from keyframes.")
        return script_data, api_responses

    # ------------------------------------------------
    # GPT models MULTI-IMAGE APPROACH (Modified for base64)
    # ------------------------------------------------
    if model_choice.lower().startswith("gpt"):
        openai_model_name = model_choice
        if not openai_client:
            logging.error("OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
            return script_data, api_responses

        # Enhance the prompt with user style instructions
        combined_prompt = (
            f"{caption_prompt_input}\n\n"
            "---------------------------------------------\n"
            "USER STYLE INSTRUCTIONS (if any):\n"
            f"{style_instructions}\n"
            "---------------------------------------------\n"
            "Incorporate any style or tone requests above, but keep all constraints.\n"
            "Now proceed with the segmented narration:\n"
        )

        messages_content = [
            {
                "type": "text",
                "text": combined_prompt.format(max_words_total=max_words_total)
            }
        ]

        # Add all segment image URLs and segment info to the messages
        for seg in segments:
            if seg['image_path']:
                try:
                    # Encode the image in base64 format
                    with open(seg['image_path'], "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

                    messages_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        }
                    })
                except Exception as e:
                    logging.error(f"Error encoding image for segment {seg['frame_numbers']}: {e}")
                    continue  # Skip to the next segment if encoding fails

                seg_info_text = (
                    f"Segment Info:\n"
                    f"Frame Numbers: {seg['frame_numbers']}\n"
                    f"Timestamp: {seg['start_time']:.2f}-{seg['end_time']:.2f}\n"
                    f"Duration: {seg['duration']:.2f} seconds\n"
                    f"Max Words Allowed: {seg['max_words']}\n"
                )

                messages_content.append({
                    "type": "text",
                    "text": seg_info_text
                })
            else:
                logging.warning(f"No image found for segment {seg['frame_numbers']}.")

        messages = [
            {
                "role": "user",
                "content": messages_content
            }
        ]

        # Call the selected GPT model a single time with all segment data
        narration_response = ""
        for attempt in range(3):
            try:
                response = openai_client.chat.completions.create(
                    model=openai_model_name,
                    messages=messages,
                    max_tokens=2000,
                    temperature=0.6
                )
                narration_response = response.choices[0].message.content.strip()
                if narration_response:
                    break
            except Exception as e:
                logging.error(f"Error generating narration with {openai_model_name}: {e}, attempt {attempt+1}/3")
                if attempt < 2:
                    time.sleep(2)

        if not narration_response:
            logging.error("Unable to generate narration after retries.")
            return script_data, api_responses

        # Parse the model's response
        segments_data = re.findall(
            r"<BEGIN SEGMENT>\s*Frame Numbers:\s*\[(.*?)\]\s*Timestamp:\s*([\d\.]+)-([\d\.]+)\s*Narration:\s*(.*?)<END SEGMENT>",
            narration_response,
            re.DOTALL
        )

        for seg_info in segments_data:
            frame_numbers = [int(num.strip()) for num in seg_info[0].split(',')]
            start_time = float(seg_info[1])
            end_time = float(seg_info[2])
            narration = seg_info[3].strip().replace('\n', ' ')

            # Find the segment that matches the frame numbers
            matching_segment = next((s for s in segments if s["frame_numbers"] == frame_numbers), None)

            if matching_segment:
                script_data.append({
                    "frame_numbers": frame_numbers,
                    "timestamp": f"{start_time:.2f}-{end_time:.2f}",
                    "narration": narration,
                    "OST": 2
                })

        # Save the entire response
        api_responses.append({
            "segments": segments,
            "narration": narration_response,
            "prompt": combined_prompt
        })

        logging.info(f"Generated script data with {openai_model_name} and style_instructions.")

    elif model_choice.lower().startswith("claude"):
        if "3-7" in model_choice:
            anthropic_model_name = "claude-3-7-sonnet-20241022"
        else:
            anthropic_model_name = "claude-3-5-sonnet-20241022"
        if not anthropic_client:
            logging.error("Claude API key not found. Please set CLAUDE_API_KEY in your environment.")
            return script_data, api_responses

        # Build messages for Claude
        script_data = []
        api_responses = []

        # Construct combined prompt
        combined_prompt = (
            f"{caption_prompt_input}\n\n"
            "---------------------------------------------\n"
            "USER STYLE INSTRUCTIONS (if any):\n"
            f"{style_instructions}\n"
            "---------------------------------------------\n"
            "Consider all images within these segments as a continuous sequence.\n"
            "For each segment, provide your response in this exact format:\n"
            "<BEGIN SEGMENT>\n"
            "Frame Numbers: [frame_numbers]\n"
            "Timestamp: start_time-end_time\n"
            "Narration: your_narration_here\n"
            "<END SEGMENT>\n"
        )

        messages_content = [{
            "type": "text",
            "text": combined_prompt
        }]

        # Add all segment images and info to the messages
        for seg in segments:
            if seg['image_path']:
                try:
                    # Load and encode image
                    with open(seg['image_path'], "rb") as image_file:
                        img_str = base64.b64encode(image_file.read()).decode("utf-8")
                    
                    # Add image
                    messages_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_str
                        }
                    })

                    # Add segment info
                    seg_info_text = (
                        f"Segment Info:\n"
                        f"Frame Numbers: {seg['frame_numbers']}\n"
                        f"Timestamp: {seg['start_time']:.2f}-{seg['end_time']:.2f}\n"
                        f"Duration: {seg['duration']:.2f} seconds\n"
                        f"Max Words Allowed: {seg['max_words']}\n"
                    )
                    messages_content.append({
                        "type": "text",
                        "text": seg_info_text
                    })
                except Exception as e:
                    logging.error(f"Error encoding image for segment {seg['frame_numbers']}: {e}")
                    continue  # Skip to the next segment if encoding fails
            else:
                logging.warning(f"No image found for segment {seg['frame_numbers']}.")

        messages = [{
            "role": "user",
            "content": messages_content
        }]

        # Call Claude a single time with all segment data
        narration_response = ""
        for attempt in range(3):
            try:
                response = anthropic_client.messages.create(
                    model=anthropic_model_name,
                    max_tokens=2000,
                    messages=messages,
                    temperature=0.7
                )
                if hasattr(response, 'content') and isinstance(response.content, list):
                    narration_response = response.content[0].text.strip()
                    if narration_response:
                        break
                else:
                    logging.error("Unexpected response format from Claude API")
            except Exception as e:
                logging.error(f"Error generating narration with Claude: {e}, attempt {attempt+1}/3")
                if attempt < 2:
                    time.sleep(2)

        if not narration_response:
            logging.error("Unable to generate narration after retries.")
            return script_data, api_responses

        # Parse the model's response
        segments_data = re.findall(
            r"<BEGIN SEGMENT>\s*Frame Numbers:\s*\[(.*?)\]\s*Timestamp:\s*([\d\.]+)-([\d\.]+)\s*Narration:\s*(.*?)<END SEGMENT>",
            narration_response,
            re.DOTALL
        )

        for seg_info in segments_data:
            try:
                frame_numbers = [int(num.strip()) for num in seg_info[0].split(',')]
                start_time = float(seg_info[1])
                end_time = float(seg_info[2])
                narration = seg_info[3].strip().replace('\n', ' ')

                # Find the segment that matches the frame numbers
                matching_segment = next((s for s in segments if s["frame_numbers"] == frame_numbers), None)

                if matching_segment:
                    script_data.append({
                        "frame_numbers": frame_numbers,
                        "timestamp": f"{start_time:.2f}-{end_time:.2f}",
                        "narration": narration,
                        "OST": 2
                    })
            except Exception as e:
                logging.error(f"Error processing segment data: {str(e)}")
                continue

        # Save the entire response
        api_responses.append({
            "segments": segments,
            "narration": narration_response,
            "prompt": combined_prompt
        })

        logging.info(f"Generated script data with {anthropic_model_name} and style_instructions.")

    elif model_choice == "Llama3":
        if not llama_model:
            logging.error("Local Llama model not loaded. Please set LLAMA_MODEL_PATH in your environment.")
            return script_data, api_responses

        combined_prompt = (
            f"{caption_prompt_input}\n\n"
            "---------------------------------------------\n"
            f"{style_instructions}\n"
            "---------------------------------------------\n"
            "For each segment described below provide your response in this format:\n"
            "<BEGIN SEGMENT>\n"
            "Frame Numbers: [frame_numbers]\n"
            "Timestamp: start_time-end_time\n"
            "Narration: your_narration_here\n"
            "<END SEGMENT>\n"
        )

        for seg in segments:
            seg_info = (
                f"Segment Info:\n"
                f"Frame Numbers: {seg['frame_numbers']}\n"
                f"Timestamp: {seg['start_time']:.2f}-{seg['end_time']:.2f}\n"
                f"Duration: {seg['duration']:.2f} seconds\n"
                f"Max Words Allowed: {seg['max_words']}\n"
            )
            combined_prompt += seg_info

        narration_response = ""
        for attempt in range(3):
            try:
                resp = llama_model.create_completion(
                    prompt=combined_prompt,
                    max_tokens=2000,
                    temperature=0.7,
                )
                narration_response = resp["choices"][0]["text"].strip()
                if narration_response:
                    break
            except Exception as e:
                logging.error(f"Error generating narration with Llama: {e}, attempt {attempt+1}/3")
                if attempt < 2:
                    time.sleep(2)

        if not narration_response:
            logging.error("Unable to generate narration with Llama after retries.")
            return script_data, api_responses

        segments_data = re.findall(
            r"<BEGIN SEGMENT>\s*Frame Numbers:\s*\[(.*?)\]\s*Timestamp:\s*([\d\.]+)-([\d\.]+)\s*Narration:\s*(.*?)<END SEGMENT>",
            narration_response,
            re.DOTALL,
        )

        for seg_info in segments_data:
            try:
                frame_numbers = [int(num.strip()) for num in seg_info[0].split(',')]
                start_time = float(seg_info[1])
                end_time = float(seg_info[2])
                narration = seg_info[3].strip().replace('\n', ' ')
                matching_segment = next((s for s in segments if s["frame_numbers"] == frame_numbers), None)
                if matching_segment:
                    script_data.append({
                        "frame_numbers": frame_numbers,
                        "timestamp": f"{start_time:.2f}-{end_time:.2f}",
                        "narration": narration,
                        "OST": 2
                    })
            except Exception as e:
                logging.error(f"Error processing segment data: {str(e)}")
                continue

        api_responses.append({
            "segments": segments,
            "narration": narration_response,
            "prompt": combined_prompt
        })

        logging.info("Generated script data with Llama3.")

    else:
        logging.error(f"Unsupported model choice: {model_choice}")
        return script_data, api_responses

    logging.info("Finished generating captions and narrations.")
    return script_data, api_responses
