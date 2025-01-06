# VideoFck (by LordHaziza): AI-Powered Video Narration App

**VideoFck (by LordHaziza)** is a Streamlit application that automatically generates narrations for your videos using AI. It extracts keyframes, creates captions, synthesizes speech, and merges the audio back into the video, with the option to add burned-in subtitles. This tool supports multiple AI models for captioning and narration, and offers flexibility in text-to-speech engines and voice selection.

## Features

*   **Automatic Keyframe Extraction:** Extracts keyframes from your video using either threshold-based or fixed-number methods.
*   **AI-Powered Captions & Narration:** Generates contextually relevant captions and narrations using large language models (LLMs).
*   **Text-to-Speech (TTS) Integration:** Supports Google Cloud TTS and ElevenLabs, allowing you to select voices and adjust speed.
*   **Subtitles:** Option to burn subtitles directly into the output video.
*   **Customizable Narration:** Includes prompts to adjust narration style, tone, and perspective.
*   **Flexible Model Choice:** Supports multiple AI models such as Claude and GPT for captioning and narration.
*  **Cross-Platform Setup:** Provides setup scripts for both Windows and macOS/Linux.

## Setup Instructions

Follow these steps to set up VideoFck (by LordHaziza) on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Ido108/VIDEOFCK.git # Replace with your repo url
    cd VIDEOFCK
    ```
2.  **Run the Setup Script:**

    *   **Windows:** Double-click the `setup.bat` file.
    *   **macOS/Linux:** Run `./setup.sh` in your terminal.

    The setup script will:
        * Create a virtual environment (`venv`)
        * Install required Python dependencies.
        * Prompt you for API keys and Google Application Credentials.
        * Create a `.env` file with the provided keys.
        * Create a `config.py` file with placeholders to read API keys from the .env file.
        * Create a `start.bat` (Windows) or `start.sh` (macOS/Linux) script to launch the app.

3.  **Run the application:**
    *   **Windows:** Double-click `start.bat`
    *   **macOS/Linux:** Run `./start.sh`

## Prerequisites

Before using VideoFck (by LordHaziza), ensure you have the following:

*   **Python 3.6+** installed on your system.
*   **`pip`** (Python's package installer).
*   **API Keys:**
    *   ElevenLabs API Key (if using ElevenLabs TTS).
    *   OpenAI API Key (if using GPT model).
    *   Claude API Key (if using Claude model).
    *   Gemini API Key.
    *  Langchain API key (optional - for tracing).
*   **Google Cloud SDK (Optional):** For Google Cloud TTS and Google Cloud Storage (GCS) you must install the Google Cloud SDK: [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install), and have a [Google Service Account Key](https://cloud.google.com/iam/docs/keys-create-delete) which its path must be inserted when prompted (Optional).
*   **ffmpeg**: make sure that ffmpeg is installed and accessible by the system.



## Using the Application

1.  **Upload Video:** Upload a video file using the file input component.
2.  **Keyframe Settings:** Choose your keyframe extraction method and settings.
    *  **Threshold**: extract keyframes based on the threshold parameter
    *  **Number**: extract a specific number of keyframes, evenly spaced in the video.
    *   Adjust `keyframes per segment` to control the number of keyframes that are given to the LLM for caption and narration generation.
3.  **Model Settings:**
    *   Choose between the **Claude** or **GPT** model for captions and narrations.
    *   Adjust playback speed and add style instructions.
4.  **Text-to-Speech (TTS):** Select a TTS Engine (Google Cloud TTS or ElevenLabs), a voice, and model if using ElevenLabs.
5.  **Burn Subtitles:** Check the 'Burn Subtitles' box if you wish to add burned subtitles to the output video.
6.  **Original Audio Volume**: Change the original audio volume if needed, this is useful if you wish to keep the original audio with a low volume along with the generated narrations.
7.  **Process Video:** Click the "Process Video" button to begin generating the output video.
8.  **View the results:** When the app finishes, it will present you the processed video in the player, and it will show the path for where the json files and the video where saved, for your convenience.

## File Structure