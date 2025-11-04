# VideoFck Development Guide

## 🏗️ Architecture Overview

VideoFck is a Python-based AI video narration application built with:
- **Frontend**: Gradio web interface
- **Video Processing**: OpenCV, MoviePy
- **AI Models**: Anthropic Claude, OpenAI GPT-4o
- **TTS Engines**: Google Cloud TTS, ElevenLabs
- **Speech Recognition**: OpenAI Whisper

### Project Structure

```
VIDEOFCK/
├── app.py                      # Main Gradio application
├── config.py                   # Configuration management
├── constants.py                # Centralized constants
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── .github/
│   └── workflows/
│       └── ci.yml             # CI/CD pipeline
├── utils/
│   ├── video_utils.py         # Video processing functions
│   ├── gemini_utils.py        # AI narration generation
│   ├── tts_utils.py           # Text-to-speech synthesis
│   ├── asr_utils.py           # Speech recognition
│   ├── error_handling.py      # Retry logic & circuit breakers
│   ├── validation.py          # Input validation & security
│   └── logging_setup.py       # Logging configuration
├── tests/
│   ├── conftest.py            # Pytest fixtures
│   ├── test_video_utils.py
│   ├── test_gemini_utils.py
│   ├── test_tts_utils.py
│   └── test_asr_utils.py
└── prompts/                    # AI prompt templates
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- FFmpeg
- CUDA (optional, for GPU-accelerated processing)

### Installation

```bash
# Clone repository
git clone https://github.com/Ido108/VIDEOFCK.git
cd VIDEOFCK

# Install dependencies
pip install -r requirements.txt

# Set up API keys
./setup.sh  # Linux/Mac
# or
setup.bat   # Windows
```

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test file
pytest tests/test_video_utils.py -v

# Run tests matching pattern
pytest tests/ -k "test_extract" -v

# Run with markers
pytest tests/ -m unit  # Only unit tests
pytest tests/ -m "not slow"  # Skip slow tests
```

### Code Quality Checks

```bash
# Format code
black .

# Check formatting
black --check --diff .

# Lint with flake8
flake8 . --max-line-length=127

# Type checking
mypy utils/ --ignore-missing-imports

# Security scan
bandit -r utils/ app.py

# Code complexity
radon cc utils/ -a -nb
```

## 📝 Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following PEP 8 style guide
- Add docstrings to all functions
- Include type hints
- Update tests for new functionality

### 3. Run Tests

```bash
# Run tests
pytest tests/ -v

# Check coverage (must be >90%)
pytest tests/ --cov=. --cov-fail-under=90
```

### 4. Commit and Push

```bash
git add .
git commit -m "feat: Add your feature description"
git push origin feature/your-feature-name
```

### 5. Create Pull Request

- CI/CD pipeline will automatically run tests
- Ensure all checks pass before merging

## 🧪 Testing Guidelines

### Test Organization

Tests are organized by module in the `tests/` directory:

- `conftest.py` - Shared fixtures and configuration
- `test_*.py` - Test files matching source files

### Writing Tests

```python
import pytest
from utils.your_module import your_function

class TestYourFunction:
    """Tests for your_function."""

    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test basic use case."""
        result = your_function("input")
        assert result == "expected"

    @pytest.mark.unit
    def test_error_handling(self):
        """Test error conditions."""
        with pytest.raises(ValueError):
            your_function(None)
```

### Test Fixtures

Common fixtures available in `conftest.py`:

- `temp_dir` - Temporary directory for test files
- `mock_video_path` - Mock video file
- `mock_keyframes_folder` - Folder with test keyframes
- `mock_anthropic_client` - Mocked Claude API
- `mock_openai_client` - Mocked GPT API
- `mock_google_tts_client` - Mocked Google TTS
- `mock_elevenlabs_client` - Mocked ElevenLabs

### Test Markers

Use markers to categorize tests:

```python
@pytest.mark.unit       # Fast unit test
@pytest.mark.integration  # Integration test
@pytest.mark.slow       # Long-running test
@pytest.mark.api        # Requires real API calls (expensive)
```

Run specific markers:
```bash
pytest tests/ -m unit  # Only unit tests
pytest tests/ -m "not slow"  # Skip slow tests
```

## 🔧 Common Development Tasks

### Adding a New Feature

1. **Plan the feature**
   - Update architecture docs if needed
   - Design API/interfaces

2. **Implement the feature**
   - Write code in appropriate module
   - Add constants to `constants.py`
   - Use error handling decorators from `utils/error_handling.py`
   - Add input validation from `utils/validation.py`

3. **Write tests**
   - Create test file `tests/test_your_module.py`
   - Aim for 90%+ coverage
   - Include edge cases and error conditions

4. **Update documentation**
   - Add docstrings
   - Update README.md if user-facing
   - Update DEVELOPMENT.md for technical changes

### Adding API Error Handling

Use the retry decorators:

```python
from utils.error_handling import exponential_backoff_retry

@exponential_backoff_retry(max_retries=3, base_delay=2.0)
def call_external_api():
    response = requests.get("https://api.example.com")
    return response.json()
```

### Adding Input Validation

Use validation utilities:

```python
from utils.validation import (
    validate_video_file,
    sanitize_file_path,
    validate_api_key
)

def process_video(video_path: str):
    # Validate and sanitize input
    validate_video_file(video_path)
    safe_path = sanitize_file_path(video_path)

    # Process video...
```

## 🔒 Security Best Practices

1. **Never commit API keys** - Use environment variables
2. **Validate all user input** - Use `utils/validation.py`
3. **Sanitize file paths** - Prevent path traversal attacks
4. **Use constants** - Centralize magic numbers in `constants.py`
5. **Run security scans** - Use `bandit` before committing

## 🐛 Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Run Single Test with Output

```bash
pytest tests/test_video_utils.py::TestExtractKeyFrames::test_extract_key_frames_threshold_mode -v -s
```

### Use pdb Debugger

```python
import pdb; pdb.set_trace()
```

## 📊 Performance Optimization

### Profiling

```bash
# Profile code
python -m cProfile -o output.prof app.py

# View profile
python -m pstats output.prof
```

### Memory Profiling

```bash
pip install memory_profiler
python -m memory_profiler app.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write/update tests
5. Ensure all tests pass
6. Submit a pull request

### Commit Message Format

Follow conventional commits:

```
feat: Add new feature
fix: Fix bug
docs: Update documentation
test: Add tests
refactor: Refactor code
perf: Performance improvement
chore: Maintenance tasks
```

## 📞 Support

- **Issues**: https://github.com/Ido108/VIDEOFCK/issues
- **Discussions**: https://github.com/Ido108/VIDEOFCK/discussions

## 📜 License

See LICENSE file for details.
