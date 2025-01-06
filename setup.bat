@echo off
setlocal

echo Setting up the environment for the video processing app...

:: Step 1: Check if Python is installed
echo Checking for Python...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed. Please install it before running this script.
    pause
    exit /b 1
)

:: Step 2: Check if pip is installed
echo Checking for pip...
where pip >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: pip is not installed. Please install it before running this script.
    pause
    exit /b 1
)

:: Step 3: Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Error: Failed to create virtual environment.
    pause
    exit /b 1
)

:: Step 4: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Error: Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Step 5: Install Python dependencies
echo Installing required Python packages...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install Python packages. Please check the requirements.txt file and try again.
    pause
    exit /b 1
)
echo Python packages installed successfully.

:: Step 6: Create .env file
echo Creating .env file...
type nul > .env

:: Prompt for API keys and other configuration
echo Please enter your API keys and configurations. You can leave empty if you wish to manually configure later.

:: ELEVENLABS_API_KEY
set /p elevenlabs_api_key="Enter your ElevenLabs API key (Link to retrieve: https://elevenlabs.io/app/settings/api-keys ): "
if not "%elevenlabs_api_key%"=="" (
    echo ELEVENLABS_API_KEY=%elevenlabs_api_key% >> .env
)

:: OPENAI_API_KEY
set /p openai_api_key="Enter your OpenAI API key ( Link to retrieve: https://platform.openai.com/api-keys ): "
if not "%openai_api_key%"=="" (
    echo OPENAI_API_KEY=%openai_api_key% >> .env
)

:: CLAUDE_API_KEY
set /p claude_api_key="Enter your Claude API key: (Link to retrieve: https://console.anthropic.com/settings/keys )"
if not "%claude_api_key%"=="" (
   echo CLAUDE_API_KEY=%claude_api_key% >> .env
)

:: GEMINI_API_KEY
set /p gemini_api_key="Enter your Gemini API key ( Link to retrieve: https://aistudio.google.com/app/apikey ): "
if not "%gemini_api_key%"=="" (
    echo GEMINI_API_KEY=%gemini_api_key% >> .env
)

:: GOOGLE_APPLICATION_CREDENTIALS (path to json) (Optional for Google TTS. help: https://cloud.google.com/text-to-speech/docs/authentication )
where gcloud >nul 2>&1
if %errorlevel% equ 0 (
    set /p google_app_cred="Enter the full path to your Google Application Credentials JSON file (optional): "
    if not "%google_app_cred%"=="" (
      echo GOOGLE_APPLICATION_CREDENTIALS="%google_app_cred%" >> .env
    )
)

echo Created .env file and added keys.

:: Step 7: Create config.py and fill it
echo Creating and populating config.py...
(
echo # config.py
echo.
echo import os
echo from dotenv import load_dotenv
echo.
echo load_dotenv()
echo.
echo # Retrieve API keys from environment variables
echo GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
echo ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
echo CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
echo OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
echo GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
echo.
echo def verify_api_keys():
echo     """Verify that all necessary API keys are set."""
echo     keys_present = True
echo     if not GEMINI_API_KEY:
echo         print("Error: GEMINI_API_KEY not found in environment variables.")
echo         keys_present = False
echo     if not ELEVENLABS_API_KEY:
echo         print("Warning: ELEVENLABS_API_KEY not found. ElevenLabs TTS will not be available.")
echo     return keys_present
) > config.py

echo config.py file created and populated.


:: Step 8: Create start.bat with application launch
echo Creating start.bat file with application launch...

(
echo @echo off
echo cd %~dp0
echo call venv\Scripts\activate
echo start "" python app.py
) > start.bat


echo Setup complete. A start.bat file with the start command is created.
echo
echo
echo
echo ===================================================================
echo ===================================================================
echo INSTALL FINISHED YOU LAZY A** B***! Provided by Haziza, the one and only.
echo ===================================================================
echo
echo *** You can now run the app with start.bat ***

pause
exit /b 0