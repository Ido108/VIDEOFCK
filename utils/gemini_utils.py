import logging
import os
from config import CLAUDE_API_KEY, OPENAI_API_KEY
import anthropic
from openai import OpenAI
import time
import datetime
import re
import base64
from google.cloud import storage
from PIL import Image
from io import BytesIO  # Import BytesIO

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

def upload_to_gcs(local_path, project_folder, bucket_name="eighth-block-311611.appspot.com"):
    """
    Uploads an image to Google Cloud Storage and returns the public URL.
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        project_name = os.path.basename(project_folder)
        blob_name = f"video-app/{project_name}/keyframes_sequence.jpg"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        image_url = blob.public_url
        logging.info(f"Uploaded image to {image_url}")
        return image_url, blob_name
    except Exception as e:
        logging.error(f"Error uploading to GCS: {e}")
        return None, None


def delete_from_gcs(blob_name, bucket_name="eighth-block-311611.appspot.com"):
    """Deletes a file from Google Cloud Storage."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
        logging.info(f"Deleted file {blob_name} from GCS bucket {bucket_name}")
        return True
    except Exception as e:
        logging.error(f"Error deleting file {blob_name} from GCS: {e}")
        return False


# Initialize clients
if CLAUDE_API_KEY:
    anthropic_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    anthropic_model_name = "claude-3-haiku-20240307"
else:
    anthropic_client = None
    anthropic_model_name = None

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None


def generate_captions_and_narrations(
    project_folder,  # Now takes project folder instead of keyframes folder
    caption_prompt_input,
    narration_prompt_input,
    video_duration,
    model_choice,
    style_instructions="",
    keyframes_per_segment=3  # New parameter with default value
):
    blob_name = None
    """
    Generate captions and narrations using the sequence keyframe image.
    """
    logging.info("Generating segmented narrations for keyframes.")
    script_data = []
    api_responses = []

    words_per_second = 2.5
    max_words_total = int(video_duration * words_per_second)
    logging.info(f"Max words for video: {max_words_total}")

    # Load the sequence image
    sequence_image_path = os.path.join(project_folder, "keyframes_sequence.jpg")
    try:
        sequence_image = Image.open(sequence_image_path)
    except Exception as e:
        logging.error(f"Error opening sequence image: {e}")
        return script_data, api_responses

    # Get keyframe timestamps from the sequence image filename
    keyframes_folder = os.path.join(project_folder, "keyframes")
    keyframe_files = sorted(
        [f for f in os.listdir(keyframes_folder) if f.endswith('.jpg')],
        key=lambda x: extract_timestamp(x) or 0.0
    )

    if not keyframe_files:
        logging.error(f"No keyframe files found in: {keyframes_folder}")
        return script_data, api_response

    # ------------------------------------------------
    # Adjusting Segment Creation to Use Multiple Keyframes
    # ------------------------------------------------
    group_size = keyframes_per_segment  # Use the parameter from user input
    step_size = group_size - 1  # Overlap by one frame to ensure continuity

    segments = []
    total_frames = len(keyframe_files)
    idx = 0

    while idx <= total_frames - group_size:
        current_files = keyframe_files[idx:idx + group_size]
        frame_count = len(current_files)
        frame_numbers = list(range(idx, idx + frame_count))

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
            continue  # Skip segments with non-positive duration

        max_words = max(1, int(duration * words_per_second))
        images_urls = []  # This will now contain only one URL, the sequence image URL
        #images_urls.append(sequence_image_url)

        segments.append({
            "frame_numbers": frame_numbers,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "max_words": max_words,
            "images": images_urls  # Only the sequence image URL
        })

        idx += step_size  # Overlap segments by one frame

    # Handle any remaining frames at the end
    if idx < total_frames:
        current_files = keyframe_files[idx:]
        frame_count = len(current_files)
        frame_numbers = list(range(idx, idx + frame_count))
        timestamps = [extract_timestamp(f) for f in current_files]

        if None not in timestamps and len(timestamps) >= 2:
            start_time = timestamps[0]
            end_time = timestamps[-1]
            duration = end_time - start_time
            max_words = max(1, int(duration * words_per_second))
            images_urls = []  # Only the sequence image URL

            segments.append({
                "frame_numbers": frame_numbers,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "max_words": max_words,
                "images": images_urls
            })

    if not segments:
        logging.error("No segments could be created from keyframes.")
        return script_data, api_responses

    # ------------------------------------------------
    # GPT-4o MULTI-IMAGE APPROACH
    # ------------------------------------------------
    if model_choice == "gpt":
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

        # Encode the sequence image to base64
        try:
           buffered = BytesIO()
           sequence_image.save(buffered, format="JPEG")
           img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        except Exception as e:
            logging.error(f"Error encoding sequence image to base64: {e}")
            return script_data, api_responses

        # Build messages for the GPT model
        messages_content = [
            {
                "type": "text",
                "text": combined_prompt.format(max_words_total=max_words_total)
            },
            {  # Add the image as base64
                "type": "image_url",
                 "image_url": {
                    "url": f"data:image/jpeg;base64,{img_str}",
                     "detail": "high",
                    }
            }
        ]

        for seg in segments:
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

        messages = [
            {
                "role": "user",
                "content": messages_content
            }
        ]

        # Call GPT-4o
        narration_response = ""
        for attempt in range(3):
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-2024-11-20",
                    messages=messages,
                    max_tokens=2000,
                    temperature=0.5
                )
                narration_response = response.choices[0].message.content.strip()
                if narration_response:
                    break
            except Exception as e:
                logging.error(f"Error generating narration with GPT-4o: {e}, attempt {attempt+1}/3")
                if attempt < 2:
                    time.sleep(2)

        if not narration_response:
            logging.error("Unable to generate narration after retries.")
            return script_data, api_responses

        # Parse the model’s response
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

            script_data.append({
                "frame_numbers": frame_numbers,
                "timestamp": f"{start_time:.2f}-{end_time:.2f}",
                "narration": narration,
                "OST": 2
            })

        if len(script_data) < len(segments):
            logging.warning("Some segments were not recognized in the GPT-4o output. Filling with blank narrations.")
            existing_frames = {tuple(s["frame_numbers"]) for s in script_data}
            for seg in segments:
                if tuple(seg["frame_numbers"]) not in existing_frames:
                    script_data.append({
                        "frame_numbers": seg["frame_numbers"],
                        "timestamp": f"{seg['start_time']:.2f}-{seg['end_time']:.2f}",
                        "narration": "",
                        "OST": 2
                    })

        # Sort the script data by the first frame number
        script_data.sort(key=lambda x: x["frame_numbers"][0])

        # Save the entire response
        api_responses.append({
            "segments": segments,
            "narration": narration_response,
            "prompt": combined_prompt
        })

        logging.info("Generated script data with GPT-4o and style_instructions.")
# ... (rest of the function up to the Claude section)

    # ------------------------------------------------
    # CLAUDE MULTI-IMAGE APPROACH
    # ------------------------------------------------
    elif model_choice == "Claude":
        if not anthropic_client:
            logging.error("Claude API key not found. Please set CLAUDE_API_KEY in your environment.")
            return script_data, api_responses

        # Build messages for Claude
        script_data = []
        api_responses = []

        for seg in segments:
            frame_numbers = seg['frame_numbers']
            start_time = seg['start_time']
            end_time = seg['end_time']
            duration = seg['duration']
            max_words = seg['max_words']

            # Merge style_instructions
            custom_prompt = (
                f"{caption_prompt_input}\n\n"
                "---------------------------------------------\n"
                "USER STYLE INSTRUCTIONS (if any):\n"
                f"{style_instructions}\n"
                "---------------------------------------------\n"
                "Consider all images within this segment as a continuous sequence.\n"
                f"Frame Numbers: {frame_numbers}\n"
                f"Timestamp: {start_time:.2f}-{end_time:.2f}\n"
                f"Duration: {duration:.2f} seconds\n"
                f"Max Words Allowed: {max_words}\n"
                "Now, generate the narration segment based on these images.\n"
            )

            # Encode the sequence image to base64 correctly
            try:
                buffered = BytesIO()
                sequence_image.save(buffered, format="JPEG")  # Save as JPEG to a buffer
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8") # Encode the buffer content
            except Exception as e:
                logging.error(f"Error encoding sequence image to base64: {e}")
                continue  # Skip this segment if encoding fails

            # Build Claude messages
            messages_content = []
            messages_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_str,
                },
            })

            messages_content.append({
                "type": "text",
                "text": custom_prompt,
            })

            messages = [
                {
                    "role": "user",
                    "content": messages_content,
                }
            ]

            # Call Claude
            narration_text = ""
            for attempt in range(3):
                try:
                    response = anthropic_client.messages.create(
                        model=anthropic_model_name,
                        max_tokens=1200,
                        messages=messages,
                        temperature=1
                    )
                    if isinstance(response.content, list):
                        narration_text = response.content[0].text.strip()
                    else:
                        narration_text = response.content.strip()

                    # Word-limits
                    words = narration_text.split()
                    if len(words) <= max_words:
                        current_words = len(words)
                    else:
                        narration_text = " ".join(words[:max_words])
                        current_words = max_words
                    break
                except Exception as e:
                    logging.error(f"Error generating narration for segment with frames {frame_numbers}: {e}, attempt {attempt+1}/3")
                    if attempt < 2:
                        time.sleep(2)

            if narration_text:
                script_data.append({
                    "frame_numbers": frame_numbers,
                    "timestamp": f"{start_time:.2f}-{end_time:.2f}",
                    "narration": narration_text,
                    "OST": 2
                })
                api_responses.append({
                    "frames": frame_numbers,
                    "timestamp": f"{start_time:.2f}-{end_time:.2f}",
                    "narration": narration_text
                })
            else:
                logging.error(f"No narration generated for segment with frames {frame_numbers}. Skipping.")

        logging.info("Finished generating scripts for keyframes (Claude).")

    else:
        logging.error(f"Unsupported model choice: {model_choice}")
        return script_data, api_responses

    logging.info("Finished generating captions and narrations.")
    return script_data, api_responses