#!/bin/bash

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

echo "Setting up the environment for the video processing app..."

# Step 1: Check if Python is installed
echo "Checking for Python..."
if ! command_exists python3; then
  echo "Error: Python is not installed. Please install it before running this script."
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


# Step 5: Install Python dependencies
echo "Installing required Python packages..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
  echo "Error: Failed to install Python packages. Please check the requirements.txt file and try again."
  exit 1
fi
echo "Python packages installed successfully."

# Step 6: Create .env file
echo "Creating .env file..."
touch .env

# Prompt for API keys and other configuration
echo "Please enter your API keys and configurations. You can leave empty if you wish to manually configure later."

# ELEVENLABS_API_KEY
read -r -p "Enter your ElevenLabs API key: " elevenlabs_api_key
if [[ ! -z "$elevenlabs_api_key" ]]; then
    echo "ELEVENLABS_API_KEY=$elevenlabs_api_key" >> .env
fi

# OPENAI_API_KEY
read -r -p "Enter your OpenAI API key: " openai_api_key
if [[ ! -z "$openai_api_key" ]]; then
    echo "OPENAI_API_KEY=$openai_api_key" >> .env
fi

# CLAUDE_API_KEY
read -r -p "Enter your Claude API key: " claude_api_key
if [[ ! -z "$claude_api_key" ]]; then
   echo "CLAUDE_API_KEY=$claude_api_key" >> .env
fi

# GEMINI_API_KEY
read -r -p "Enter your Gemini API key: " gemini_api_key
if [[ ! -z "$gemini_api_key" ]]; then
    echo "GEMINI_API_KEY=$gemini_api_key" >> .env
fi

# GOOGLE_APPLICATION_CREDENTIALS (Optional, only prompted if gcloud SDK is installed)
if command_exists gcloud; then
  read -r -p "Enter the full path to your Google Application Credentials JSON file (optional): " google_app_cred
  if [[ ! -z "$google_app_cred" ]]; then
    echo "GOOGLE_APPLICATION_CREDENTIALS=\"$google_app_cred\"" >> .env
  fi
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