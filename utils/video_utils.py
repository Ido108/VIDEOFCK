import cv2
import os
import numpy as np
import logging
from PIL import Image, ImageDraw, ImageFont
import re
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import AudioArrayClip
import subprocess
# For subtitles:
from moviepy.video.tools.subtitles import file_to_subtitles, SubtitlesClip
from moviepy import CompositeVideoClip, TextClip


def create_video_with_audio(video_path, audio_paths, timestamps, output_path, original_audio_volume=1.0):
    """
    Creates a video with narration segments timed according to timestamps,
    and includes the original video audio with adjustable volume.
    """
    logging.info(f"Creating video with narrations from {video_path}")

    video_clip = None
    try:
        video_clip = VideoFileClip(video_path)
        final_duration = video_clip.duration
        video_fps = video_clip.fps
        audio_fps = 44100
        final_samples = int(final_duration * audio_fps)
        final_audio = np.zeros((final_samples, 2), dtype=np.float32)

        # Handle original audio
        if video_clip.audio:
            original_audio = video_clip.audio.to_soundarray()
            original_audio = original_audio * original_audio_volume  # Adjust volume

            # Pad or trim original audio to match final video length
            if len(original_audio) > final_samples:
                original_audio = original_audio[:final_samples]
            elif len(original_audio) < final_samples:
                padding = np.zeros((final_samples - len(original_audio), 2))
                original_audio = np.concatenate((original_audio, padding))

            final_audio += original_audio  # Mix original audio

        for i, audio_path in enumerate(audio_paths):
            audio_clip = AudioFileClip(audio_path)
            start_time = timestamps[i]
            end_time = timestamps[i + 1] if i < len(timestamps) - 1 else video_clip.duration
            max_duration = end_time - start_time

            # Convert audio to array at 44100 fps
            audio_array = audio_clip.to_soundarray(fps=audio_fps)
            samples_needed = int(max_duration * audio_fps)
            if len(audio_array) > samples_needed:
                audio_array = audio_array[:samples_needed]

            start_idx = int(start_time * audio_fps)
            end_idx = start_idx + len(audio_array)

            # Ensure end_idx does not exceed final_samples
            end_idx = min(end_idx, final_samples)

            # Mix only the relevant portion of the generated audio
            final_audio[start_idx:end_idx, :] += audio_array[:end_idx - start_idx, :]
            audio_clip.close()

        # Create a composite audio clip
        audio_clip = AudioArrayClip(final_audio, fps=audio_fps)
        final_video = video_clip.with_audio(audio_clip)

        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            fps=video_fps
        )

        logging.info(f"Successfully created video with audio: {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Error creating video with audio: {str(e)}")
        return None
    finally:
        if video_clip:
            video_clip.close()

def make_srt_from_script_data(script_data, srt_output_path):
    """Generate an SRT subtitle file from the narration script data."""
    def to_srt_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    try:
        with open(srt_output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(script_data, start=1):
                time_str = segment["timestamp"]
                start_str, end_str = time_str.split('-')
                start_sec = float(start_str)
                end_sec = float(end_str)

                start_srt = to_srt_time(start_sec)
                end_srt = to_srt_time(end_sec)

                f.write(f"{i}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{segment['narration']}\n\n")

        logging.info(f"Created SRT subtitles at: {srt_output_path}")
        return srt_output_path
    except Exception as e:
        logging.error(f"Error creating SRT file: {e}")
        return None

def add_subtitles_to_video(input_video, srt_file, output_video, burn_in=False):
    """
    Burn subtitles into the video by re-encoding.
    This time using MoviePy's 'file_to_subtitles' + 'SubtitlesClip' approach
    instead of the FFmpeg subtitles filter.
    """

    try:
        if not burn_in:
            # If burn_in is False, just return the original video path
            logging.info("burn_in=False, so skipping subtitles.")
            return input_video

        logging.info("Burning subtitles with MoviePy's SubtitlesClip approach.")

        # 1) Load the main video
        base_clip = VideoFileClip(input_video)

        # 2) Convert SRT into a list of ((start, end), text) using file_to_subtitles
        subtitles_data = file_to_subtitles(srt_file)  # e.g. [((0, 5), "Hello"), ...]

        # 3) A generator function for each subtitle text (styling)
        def make_textclip(txt):
            # You can customize color, stroke, etc.
            return (TextClip(txt, font="Arial", color='white', stroke_color='black', stroke_width=2)
                    .on_color(size=(base_clip.w, None), color=(0,0,0,0))  # transparent background
                    .set_position(("center", "bottom")))

        # 4) Create a SubtitlesClip, which is a VideoClip of timed subtitles
        subs_clip = SubtitlesClip(subtitles_data, make_textclip=make_textclip)

        # 5) Composite the subtitles on top of the base video
        final_clip = base_clip if subs_clip is None else CompositeVideoClip([base_clip, subs_clip])

        # 6) Write the final clip to output, re-encoding with libx264
        final_clip.write_videofile(
            output_video,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp_sub_audio.m4a',
            remove_temp=True,
            fps=base_clip.fps
        )

        logging.info(f"Successfully burned subtitles with MoviePy: {output_video}")
        return output_video

    except Exception as e:
        logging.error(f"Error adding subtitles to video: {e}")
        return None
    
def extract_timestamp(filename):
    match = re.search(r'_(\d+\.\d+)\.jpg$', filename)
    return float(match.group(1)) if match else None

def extract_key_frames(video_path, project_folder, keyframe_threshold=20, num_keyframes=None, keyframes_per_segment=3):
    """
    Extract key frames from a video using either frame differencing (threshold)
    or a fixed number of keyframes.
    Ensures at least 3 seconds between keyframes when using the threshold method.
    Now also creates a combined keyframe sequence image.
    """
    logging.info(f"Extracting key frames from {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file at {video_path}")
        return [], []

    keyframes_folder = os.path.join(project_folder, "keyframes")
    os.makedirs(keyframes_folder, exist_ok=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if num_keyframes is not None:
        key_frames, keyframe_timestamps = extract_keyframes_by_number(
            cap, num_keyframes, keyframes_folder
        )
    else:
        key_frames, keyframe_timestamps = extract_keyframes_by_threshold(
            cap, keyframe_threshold, fps, keyframes_folder, total_frames
        )

    cap.release()

    # Create combined keyframe sequence image
    create_sequence_image(project_folder, keyframes_per_segment=keyframes_per_segment)

    logging.info("Finished key frame extraction")
    return key_frames, keyframe_timestamps

def create_sequence_image(project_folder, keyframes_per_segment):
    """
    Combines keyframes from a project into a single image with timestamps and frame numbers.

    Args:
        project_folder: Path to the project folder (e.g., "output/project_20250103_221206").
        keyframes_per_segment: The number of keyframes per segment and per row in the combined image.
    """
    keyframes_per_row = keyframes_per_segment
    keyframes_folder = os.path.join(project_folder, "keyframes")
    keyframe_files = sorted([f for f in os.listdir(keyframes_folder) if f.endswith('.jpg')],
                            key=lambda x: extract_timestamp(x) or 0.0)

    if not keyframe_files:
        print(f"No keyframe files found in: {keyframes_folder}")
        return

    # 1. Load the first keyframe to get dimensions and calculate target size
    first_keyframe = Image.open(os.path.join(keyframes_folder, keyframe_files[0]))
    original_keyframe_width, original_keyframe_height = first_keyframe.size
    first_keyframe.close()

    # Calculate the target width and height for each keyframe (adjust as needed)
    target_keyframe_width = 427
    target_keyframe_height = int(original_keyframe_height * (target_keyframe_width / original_keyframe_width))

    # 2. Determine Layout and Canvas Size
    num_rows = (len(keyframe_files) + keyframes_per_row - 1) // keyframes_per_row
    canvas_width = target_keyframe_width * keyframes_per_row
    canvas_height = target_keyframe_height * num_rows

    # 3. Create Canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height), "black")
    draw = ImageDraw.Draw(canvas)

    # Choose a font
    try:
        font = ImageFont.truetype("arial.ttf", 25)
    except IOError:
        print("Arial font not found, using default font.")
        font = ImageFont.load_default()

    # 4. & 5. Paste Keyframes, Add Timestamps and Frame Numbers
    for i, keyframe_file in enumerate(keyframe_files):
        keyframe_path = os.path.join(keyframes_folder, keyframe_file)
        keyframe = Image.open(keyframe_path)

        # Resize the keyframe
        keyframe = keyframe.resize((target_keyframe_width, target_keyframe_height), Image.LANCZOS)

        row = i // keyframes_per_row
        col = i % keyframes_per_row
        x = col * target_keyframe_width
        y = row * target_keyframe_height

        canvas.paste(keyframe, (x, y))

        # Extract timestamp
        timestamp = extract_timestamp(keyframe_file)
        if timestamp is not None:
            timestamp_text = f"{timestamp:.2f}"

            # Calculate text position for timestamp
            text_x = x + 5
            text_y = y + 5

            # Add a background for the text
            text_bbox = draw.textbbox((text_x, text_y), timestamp_text, font=font)
            draw.rectangle(text_bbox, fill="black")
            draw.text((text_x, text_y), timestamp_text, font=font, fill="white")

        # Add frame number
        frame_number_text = f"Keyframe - {i}"

        # Calculate text position for frame number
        frame_number_x = x + 5
        frame_number_y = y + 5 + 50

        # Add a background for the text
        frame_number_bbox = draw.textbbox((frame_number_x, frame_number_y), frame_number_text, font=font)
        draw.rectangle(frame_number_bbox, fill="black")
        draw.text((frame_number_x, frame_number_y), frame_number_text, font=font, fill="yellow")

        keyframe.close()

    # 6. Save the Combined Image
    output_image_path = os.path.join(project_folder, "keyframes_sequence.jpg")
    canvas.save(output_image_path, quality=85)
    print(f"Combined keyframe image saved to: {output_image_path}")

def create_sequence_image_for_segment(segment, keyframes_folder, output_folder, keyframes_per_segment):
    """
    Creates a sequence image for a single segment.

    Args:
        segment (dict): The segment data containing frame numbers, timestamps, etc.
        keyframes_folder (str): Path to the folder containing individual keyframe images.
        output_folder (str): Path to the folder where the sequence image will be saved.
        keyframes_per_segment (int): Number of keyframes per row in the sequence image.
    """
    keyframes_per_row = keyframes_per_segment

    frame_numbers = segment["frame_numbers"]
    keyframe_files = [f for f in os.listdir(keyframes_folder) if f.endswith('.jpg') and int(f.split('_')[1]) in frame_numbers]
    keyframe_files.sort(key=lambda x: extract_timestamp(x) or 0.0)

    if not keyframe_files:
        logging.warning(f"No keyframe files found for segment: {segment['frame_numbers']}")
        return

    # Load the first keyframe to get dimensions
    first_keyframe = Image.open(os.path.join(keyframes_folder, keyframe_files[0]))
    original_keyframe_width, original_keyframe_height = first_keyframe.size
    first_keyframe.close()

    # Calculate target size for keyframes in the sequence image
    target_keyframe_width = 427
    target_keyframe_height = int(original_keyframe_height * (target_keyframe_width / original_keyframe_width))

    # Determine layout and canvas size
    num_rows = (len(keyframe_files) + keyframes_per_row - 1) // keyframes_per_row
    canvas_width = target_keyframe_width * keyframes_per_row
    canvas_height = target_keyframe_height * num_rows

    # Create canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height), "black")
    draw = ImageDraw.Draw(canvas)

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 25)
    except IOError:
        font = ImageFont.load_default()

    # Paste keyframes and add timestamps/frame numbers
    for i, keyframe_file in enumerate(keyframe_files):
        keyframe_path = os.path.join(keyframes_folder, keyframe_file)
        keyframe = Image.open(keyframe_path)

        # Resize keyframe
        keyframe = keyframe.resize((target_keyframe_width, target_keyframe_height), Image.LANCZOS)

        row = i // keyframes_per_row
        col = i % keyframes_per_row
        x = col * target_keyframe_width
        y = row * target_keyframe_height

        canvas.paste(keyframe, (x, y))

        # Add timestamp
        timestamp = extract_timestamp(keyframe_file)
        if timestamp is not None:
            timestamp_text = f"{timestamp:.2f}"
            text_x = x + 5
            text_y = y + 5
            text_bbox = draw.textbbox((text_x, text_y), timestamp_text, font=font)
            draw.rectangle(text_bbox, fill="black")
            draw.text((text_x, text_y), timestamp_text, font=font, fill="white")

        # Add frame number
        frame_number_text = f"F. {frame_numbers[i]}"
        frame_number_x = x + 5
        frame_number_y = y + 5 + 30
        frame_number_bbox = draw.textbbox((frame_number_x, frame_number_y), frame_number_text, font=font)
        draw.rectangle(frame_number_bbox, fill="black")
        draw.text((frame_number_x, frame_number_y), frame_number_text, font=font, fill="yellow")

        keyframe.close()

    # Save the sequence image for the segment
    output_image_path = os.path.join(output_folder, f"segment_frames_{frame_numbers[0]}-{frame_numbers[-1]}.jpg")
    canvas.save(output_image_path, quality=85)
    logging.info(f"Sequence image for segment {frame_numbers} saved to: {output_image_path}")
    return output_image_path

def extract_keyframes_by_number(cap, num_keyframes, keyframes_folder):
    """Extracts a fixed number of keyframes, evenly spaced, including the last frame."""
    key_frames = []
    keyframe_timestamps = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = max(1, total_frames // num_keyframes)
    for i in range(num_keyframes):
        frame_index = i * frame_interval
        if frame_index >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            timestamp = frame_index / fps
            keyframe_filename = os.path.join(keyframes_folder, f"keyframe_{i}_{timestamp:.2f}.jpg")
            cv2.imwrite(keyframe_filename, frame)
            key_frames.append(frame)
            keyframe_timestamps.append(timestamp)
            logging.info(f"Saving key frame {i} at {timestamp:.2f}s")

    # Also ensure the last frame is included
    last_frame_index = total_frames - 1
    if not keyframe_timestamps or (last_frame_index / fps) > keyframe_timestamps[-1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_index)
        ret, frame = cap.read()
        if ret:
            timestamp = last_frame_index / fps
            keyframe_filename = os.path.join(keyframes_folder, f"keyframe_{i+1}_{timestamp:.2f}.jpg")
            cv2.imwrite(keyframe_filename, frame)
            key_frames.append(frame)
            keyframe_timestamps.append(timestamp)
            logging.info(f"Saving last key frame at {timestamp:.2f}s")

    return key_frames, keyframe_timestamps
def create_sequence_image_for_segment(segment, keyframes_folder, output_folder, keyframes_per_row=3):
    """
    Creates a sequence image for a single segment.

    Args:
        segment (dict): The segment data containing frame numbers, timestamps, etc.
        keyframes_folder (str): Path to the folder containing individual keyframe images.
        output_folder (str): Path to the folder where the sequence image will be saved.
        keyframes_per_row (int): Number of keyframes per row in the sequence image.
    """
    frame_numbers = segment["frame_numbers"]
    keyframe_files = [f for f in os.listdir(keyframes_folder) if f.endswith('.jpg') and int(f.split('_')[1]) in frame_numbers]
    keyframe_files.sort(key=lambda x: extract_timestamp(x) or 0.0)  # Sort by timestamp

    if not keyframe_files:
        logging.warning(f"No keyframe files found for segment: {segment['frame_numbers']}")
        return

    # Load the first keyframe to get dimensions
    first_keyframe = Image.open(os.path.join(keyframes_folder, keyframe_files[0]))
    original_keyframe_width, original_keyframe_height = first_keyframe.size
    first_keyframe.close()

    # Calculate target size for keyframes in the sequence image
    target_keyframe_width = 427  # Adjust as needed
    target_keyframe_height = int(original_keyframe_height * (target_keyframe_width / original_keyframe_width))

    # Determine layout and canvas size
    num_rows = (len(keyframe_files) + keyframes_per_row - 1) // keyframes_per_row
    canvas_width = target_keyframe_width * keyframes_per_row
    canvas_height = target_keyframe_height * num_rows

    # Create canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height), "black")
    draw = ImageDraw.Draw(canvas)

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 25)
    except IOError:
        font = ImageFont.load_default()

    # Paste keyframes and add timestamps/frame numbers
    for i, keyframe_file in enumerate(keyframe_files):
        keyframe_path = os.path.join(keyframes_folder, keyframe_file)
        keyframe = Image.open(keyframe_path)

        # Resize keyframe
        keyframe = keyframe.resize((target_keyframe_width, target_keyframe_height), Image.LANCZOS)

        row = i // keyframes_per_row
        col = i % keyframes_per_row
        x = col * target_keyframe_width
        y = row * target_keyframe_height

        canvas.paste(keyframe, (x, y))

        # Add timestamp
        timestamp = extract_timestamp(keyframe_file)
        if timestamp is not None:
            timestamp_text = f"{timestamp:.2f}"
            text_x = x + 5
            text_y = y + 5
            text_bbox = draw.textbbox((text_x, text_y), timestamp_text, font=font)
            draw.rectangle(text_bbox, fill="black")
            draw.text((text_x, text_y), timestamp_text, font=font, fill="white")

        # Add frame number
        frame_number_text = f"F. {frame_numbers[i]}"
        frame_number_x = x + 5
        frame_number_y = y + 5 + 30  # Below the timestamp
        frame_number_bbox = draw.textbbox((frame_number_x, frame_number_y), frame_number_text, font=font)
        draw.rectangle(frame_number_bbox, fill="black")
        draw.text((frame_number_x, frame_number_y), frame_number_text, font=font, fill="yellow")

        keyframe.close()

    # Save the sequence image for the segment
    output_image_path = os.path.join(output_folder, f"segment_frames_{frame_numbers[0]}-{frame_numbers[-1]}.jpg")
    canvas.save(output_image_path, quality=85)
    logging.info(f"Sequence image for segment {frame_numbers} saved to: {output_image_path}")
    return output_image_path

def extract_keyframes_by_number(cap, num_keyframes, keyframes_folder):
    """Extracts a fixed number of keyframes, evenly spaced, including the last frame."""
    key_frames = []
    keyframe_timestamps = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = max(1, total_frames // num_keyframes)
    for i in range(num_keyframes):
        frame_index = i * frame_interval
        if frame_index >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            timestamp = frame_index / fps
            keyframe_filename = os.path.join(keyframes_folder, f"keyframe_{i}_{timestamp:.2f}.jpg")
            cv2.imwrite(keyframe_filename, frame)
            key_frames.append(frame)
            keyframe_timestamps.append(timestamp)
            logging.info(f"Saving key frame {i} at {timestamp:.2f}s")

    # Also ensure the last frame is included
    last_frame_index = total_frames - 1
    if not keyframe_timestamps or (last_frame_index / fps) > keyframe_timestamps[-1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_index)
        ret, frame = cap.read()
        if ret:
            timestamp = last_frame_index / fps
            keyframe_filename = os.path.join(keyframes_folder, f"keyframe_{i+1}_{timestamp:.2f}.jpg")
            cv2.imwrite(keyframe_filename, frame)
            key_frames.append(frame)
            keyframe_timestamps.append(timestamp)
            logging.info(f"Saving last key frame at {timestamp:.2f}s")

    return key_frames, keyframe_timestamps

def extract_keyframes_by_threshold(cap, keyframe_threshold, fps, keyframes_folder, total_frames):
    """Extracts keyframes based on frame differencing with a minimum time interval."""
    prev_frame = None
    key_frames = []
    keyframe_timestamps = []
    frame_count = 0
    min_time_between_keyframes = 3.0  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count == 1:
            # Always include the first frame
            save_keyframe(frame, 0.00, keyframes_folder, key_frames, keyframe_timestamps)
            prev_frame = gray_frame
            continue

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray_frame)
            mean_diff = np.mean(diff)
            current_timestamp = frame_count / fps

            # Check threshold + 3 second separation
            if (not keyframe_timestamps or (current_timestamp - keyframe_timestamps[-1]) >= min_time_between_keyframes) and mean_diff > keyframe_threshold:
                save_keyframe(frame, current_timestamp, keyframes_folder, key_frames, keyframe_timestamps)

        prev_frame = gray_frame

    # Always include last frame if enough time has passed
    last_frame_timestamp = (total_frames - 1) / fps
    if not keyframe_timestamps or (last_frame_timestamp - keyframe_timestamps[-1]) >= min_time_between_keyframes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, last_frame = cap.read()
        if ret:
            save_keyframe(last_frame, last_frame_timestamp, keyframes_folder, key_frames, keyframe_timestamps)
        else:
            logging.error("Could not capture the last frame")

    return key_frames, keyframe_timestamps

def save_keyframe(frame, timestamp, keyframes_folder, key_frames, keyframe_timestamps):
    keyframe_index = len(key_frames)
    keyframe_filename = os.path.join(keyframes_folder, f"keyframe_{keyframe_index}_{timestamp:.3f}.jpg")
    cv2.imwrite(keyframe_filename, frame)
    key_frames.append(frame)
    keyframe_timestamps.append(timestamp)
    logging.info(f"Saving key frame {keyframe_index} at {timestamp:.3f}s")