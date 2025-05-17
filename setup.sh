#!/bin/bash

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

echo "Setting up the environment for the video processing app..."

# Step 1: Check if python3 is installed
echo "Checking for python3..."
if ! command_exists python3; then
  echo "Error: python3 is not installed. Please install it before running this script."
  exit 1
fi

# Step 2: Check if pip is installed
echo "Checking for pip..."
if ! command_exists pip; then
  echo "Error: pip is not installed. Please install it before running this script."
  exit 1
fi

# Step 3: Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
  echo "Error: Failed to create virtual environment."
  exit 1
fi

# Step 4: Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
  echo "Error: Failed to activate virtual environment."
  exit 1
fi

# Step 5: Install python3 dependencies
echo "Installing required python3 packages..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
  echo "Error: Failed to install python3 packages. Please check the requirements.txt file and try again."
  exit 1
fi
echo "python3 packages installed successfully."

# Step 6: Create .env file
echo "Creating .env file..."
> .env  # Truncate the .env file if it exists or create a new one

# Prompt for API keys and other configuration
echo "Please enter your API keys and configurations. You can leave empty if you wish to manually configure later."

# Function to prompt and write to .env
prompt_and_write() {
  local var_name=$1
  local prompt_message=$2
  local user_input

  read -r -p "$prompt_message: " user_input
  # Escape double quotes and backslashes in user input
  user_input=$(printf '%s' "$user_input" | sed 's/\\/\\\\/g; s/"/\\"/g')
  echo "${var_name}=\"${user_input}\"" >> .env
}

# ELEVENLABS_API_KEY
prompt_and_write "ELEVENLABS_API_KEY" "Enter your ElevenLabs API key"

# OPENAI_API_KEY
prompt_and_write "OPENAI_API_KEY" "Enter your OpenAI API key"

# CLAUDE_API_KEY
prompt_and_write "CLAUDE_API_KEY" "Enter your Claude API key"

# GEMINI_API_KEY
prompt_and_write "GEMINI_API_KEY" "Enter your Gemini API key"

# GOOGLE_APPLICATION_CREDENTIALS (Optional, only prompted if gcloud SDK is installed)
if command_exists gcloud; then
  prompt_and_write "GOOGLE_APPLICATION_CREDENTIALS" "Enter the full path to your Google Application Credentials JSON file (optional)"
fi

echo "Created .env file and added keys."

# Step 7: Create config.py and fill it
echo "Creating and populating config.py..."
cat > config.py << EOF
# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# Retrieve API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

def verify_api_keys():
    """Verify that all necessary API keys are set."""
    keys_present = True
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        keys_present = False
    if not ELEVENLABS_API_KEY:
        print("Warning: ELEVENLABS_API_KEY not found. ElevenLabs TTS will not be available.")
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not found. OpenAI services will not be available.")
    if not CLAUDE_API_KEY:
        print("Warning: CLAUDE_API_KEY not found. Claude services will not be available.")
    return keys_present
EOF
echo "config.py file created and populated"

# Step 8: Create start script with application launch
echo "Creating start.sh script with application launch..."
cat > start.sh << EOF
#!/bin/bash
cd "\$(dirname "\$0")"
source venv/bin/activate
python3 app.py
EOF
chmod +x start.sh
echo "start.sh file created. You can now start the app using ./start.sh"

echo "Setup complete. You can now run the app with './start.sh'."