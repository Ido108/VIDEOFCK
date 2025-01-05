@echo off
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

:: Step 3: Install Python dependencies
echo Installing required Python packages...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install Python packages. Please check the requirements.txt file and try again.
    pause
    exit /b 1
)
echo Python packages installed successfully.

:: Step 4: Create .env file
echo Creating .env file...
type nul > .env

:: Prompt for API keys and other configuration
echo Please enter your API keys and configurations. You can leave empty if you wish to manually configure later.

:: ELEVENLABS_API_KEY
set /p elevenlabs_api_key="Enter your ElevenLabs API key: "
if not "%elevenlabs_api_key%"=="" (
    echo ELEVENLABS_API_KEY=%elevenlabs_api_key% >> .env
)

:: OPENAI_API_KEY
set /p openai_api_key="Enter your OpenAI API key: "
if not "%openai_api_key%"=="" (
    echo OPENAI_API_KEY=%openai_api_key% >> .env
)

:: CLAUDE_API_KEY
set /p claude_api_key="Enter your Claude API key: "
if not "%claude_api_key%"=="" (
   echo CLAUDE_API_KEY=%claude_api_key% >> .env
)

:: GEMINI_API_KEY
set /p gemini_api_key="Enter your Gemini API key: "
if not "%gemini_api_key%"=="" (
    echo GEMINI_API_KEY=%gemini_api_key% >> .env
)


:: LANGCHAIN_TRACING_V2
set /p langchain_tracing_v2="Enable Langchain Tracing V2 (true/false): "
if not "%langchain_tracing_v2%"=="" (
  echo LANGCHAIN_TRACING_V2=%langchain_tracing_v2% >> .env
)


:: LANGCHAIN_ENDPOINT
set /p langchain_endpoint="Enter your Langchain endpoint: "
if not "%langchain_endpoint%"=="" (
   echo LANGCHAIN_ENDPOINT="%langchain_endpoint%" >> .env
)

:: LANGCHAIN_API_KEY
set /p langchain_api_key="Enter your Langchain API key: "
if not "%langchain_api_key%"=="" (
    echo LANGCHAIN_API_KEY="%langchain_api_key%" >> .env
)

:: LANGCHAIN_PROJECT
set /p langchain_project="Enter your Langchain project name: "
if not "%langchain_project%"=="" (
   echo LANGCHAIN_PROJECT="%langchain_project%" >> .env
)

:: GOOGLE_APPLICATION_CREDENTIALS (Optional, only prompted if gcloud SDK is installed)
where gcloud >nul 2>&1
if %errorlevel% equ 0 (
    set /p google_app_cred="Enter the full path to your Google Application Credentials JSON file (optional): "
    if not "%google_app_cred%"=="" (
      echo GOOGLE_APPLICATION_CREDENTIALS="%google_app_cred%" >> .env
    )
)

echo Created .env file and added keys.

:: Step 5: Create config.py and fill it
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


:: Step 6: Create start.bat with icon and application launch
echo Creating start.bat file with icon and application launch...

(
echo @echo off
echo cd %~dp0
echo start "" "python" app.py
) > start.bat

:: Add a registry tweak to add icon to batch file start.bat
echo Adding icon to batch file...
echo Windows Registry Editor Version 5.00 > temp.reg
echo. >> temp.reg
echo [HKEY_CLASSES_ROOT\batfile\DefaultIcon] >> temp.reg
echo @="\"%~dp0icon.ico\",0" >> temp.reg
echo.
reg import temp.reg
del temp.reg

echo Setup complete. A start.bat file with the icon and start command is created.
echo You can now run the app with start.bat.

pause
exit /b 0