# Testing Guide for VideoFck

## Overview

VideoFck uses `pytest` for testing with a target of **90%+ code coverage**. Tests are organized into unit tests, integration tests, and end-to-end tests.

## Quick Start

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Test Organization

### Directory Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_video_utils.py      # Video processing tests
├── test_gemini_utils.py     # AI narration tests
├── test_tts_utils.py        # Text-to-speech tests
├── test_asr_utils.py        # Speech recognition tests
└── __init__.py
```

### Test Markers

Tests are categorized using pytest markers:

```python
@pytest.mark.unit          # Fast unit tests (<1s)
@pytest.mark.integration   # Integration tests (1-10s)
@pytest.mark.slow          # Slow tests (>10s)
@pytest.mark.api           # Tests requiring real API calls (expensive)
```

Run specific categories:
```bash
pytest tests/ -m unit              # Only unit tests
pytest tests/ -m "not slow"        # Skip slow tests
pytest tests/ -m "unit or integration"  # Multiple markers
```

## Fixtures

Shared fixtures are defined in `conftest.py`:

### File System Fixtures

- **`temp_dir`**: Temporary directory that's cleaned up after test
- **`mock_video_path`**: Path to a generated test video file
- **`mock_keyframe_image`**: Path to a test keyframe image
- **`mock_keyframes_folder`**: Folder with multiple test keyframes
- **`mock_audio_path`**: Path to a test audio file

### API Client Fixtures

- **`mock_anthropic_client`**: Mocked Claude API client
- **`mock_openai_client`**: Mocked GPT-4 API client
- **`mock_google_tts_client`**: Mocked Google TTS client
- **`mock_elevenlabs_client`**: Mocked ElevenLabs TTS client

### Data Fixtures

- **`sample_script_data`**: Sample narration script data
- **`sample_segments`**: Sample video segments
- **`set_env_vars`**: Sets up test environment variables

### Example Usage

```python
def test_extract_keyframes(mock_video_path, temp_dir):
    """Test using fixtures."""
    keyframes, timestamps = extract_key_frames(
        mock_video_path,
        temp_dir,
        keyframe_threshold=20
    )
    assert len(keyframes) > 0
```

## Writing Tests

### Test Class Organization

```python
class TestFunctionName:
    """Tests for function_name."""

    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test the basic happy path."""
        result = function_name("input")
        assert result == "expected"

    @pytest.mark.unit
    def test_edge_case(self):
        """Test edge cases."""
        result = function_name("")
        assert result is None

    @pytest.mark.unit
    def test_error_handling(self):
        """Test error conditions."""
        with pytest.raises(ValueError):
            function_name(None)
```

### Mocking External Dependencies

#### Mocking API Calls

```python
from unittest.mock import patch, Mock

@patch('utils.gemini_utils.openai_client')
def test_api_call(mock_client):
    """Test with mocked API."""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="response"))]
    mock_client.chat.completions.create.return_value = mock_response

    result = generate_narration("prompt")
    assert result is not None
```

#### Mocking File Operations

```python
@patch('utils.video_utils.VideoFileClip')
def test_video_processing(mock_clip):
    """Test with mocked video clip."""
    mock_instance = Mock()
    mock_instance.duration = 10.0
    mock_clip.return_value = mock_instance

    result = process_video("fake_path.mp4")
    assert result is not None
```

### Testing Error Conditions

```python
def test_handles_missing_file():
    """Test error when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_video("/nonexistent/path.mp4")

def test_returns_none_on_error():
    """Test function returns None on error."""
    result = safe_function(invalid_input)
    assert result is None
```

### Parametrized Tests

Test multiple inputs efficiently:

```python
@pytest.mark.parametrize("input,expected", [
    ("keyframe_0_3.50.jpg", 3.50),
    ("keyframe_5_120.75.jpg", 120.75),
    ("invalid.jpg", None),
])
def test_extract_timestamp(input, expected):
    """Test timestamp extraction with various inputs."""
    assert extract_timestamp(input) == expected
```

## Coverage Requirements

### Minimum Coverage

- **Overall**: 90%
- **Per File**: 85%
- **Branch Coverage**: Enabled

### Checking Coverage

```bash
# Generate coverage report
pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

# Fail if coverage below threshold
pytest tests/ --cov=. --cov-fail-under=90
```

### Viewing Coverage Report

```bash
# HTML report (detailed, per-file)
open htmlcov/index.html

# Terminal report
pytest tests/ --cov=. --cov-report=term-missing

# XML report (for CI/CD)
pytest tests/ --cov=. --cov-report=xml
```

## Test Performance

### Running Tests Quickly

```bash
# Run in parallel (requires pytest-xdist)
pytest tests/ -n auto

# Run only fast tests
pytest tests/ -m "not slow"

# Stop on first failure
pytest tests/ -x

# Run last failed tests
pytest tests/ --lf
```

### Profiling Slow Tests

```bash
# Show slowest 10 tests
pytest tests/ --durations=10

# Show all test durations
pytest tests/ --durations=0
```

## Continuous Integration

Tests run automatically on:
- Every push to `main`, `develop`, or `claude/*` branches
- Every pull request

### CI Pipeline Stages

1. **Lint** - flake8, black formatting check
2. **Type Check** - mypy type checking
3. **Test** - pytest with coverage
4. **Security Scan** - bandit, safety
5. **Code Quality** - pylint, radon complexity

### Local CI Simulation

```bash
# Run full CI suite locally
flake8 .
black --check .
mypy utils/
pytest tests/ --cov=. --cov-fail-under=90
bandit -r utils/ app.py
pylint utils/
```

## Test Data Management

### Creating Test Videos

```python
import cv2
import numpy as np

def create_test_video(path, duration_seconds=3, fps=30):
    """Create a simple test video."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (640, 480))

    for i in range(int(duration_seconds * fps)):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = [255, 0, 0]  # Blue frame
        out.write(frame)

    out.release()
```

### Creating Test Images

```python
from PIL import Image

def create_test_image(path, color='blue'):
    """Create a test image."""
    img = Image.new('RGB', (640, 480), color=color)
    img.save(path)
```

## Common Testing Patterns

### Testing Async Operations

```python
import asyncio

@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await async_function()
    assert result is not None
```

### Testing with Temporary Files

```python
def test_with_temp_file(temp_dir):
    """Test using temporary directory."""
    test_file = os.path.join(temp_dir, "test.txt")

    # Create file
    with open(test_file, 'w') as f:
        f.write("test content")

    # Test function
    result = process_file(test_file)
    assert result is not None

    # temp_dir is automatically cleaned up
```

### Testing Logging

```python
import logging
from unittest.mock import patch

def test_logging(caplog):
    """Test logging output."""
    with caplog.at_level(logging.INFO):
        function_that_logs()

    assert "Expected log message" in caplog.text
```

### Testing Exceptions

```python
def test_raises_specific_exception():
    """Test specific exception is raised."""
    with pytest.raises(ValueError, match="expected error message"):
        function_that_raises()
```

## Debugging Tests

### Run Single Test

```bash
pytest tests/test_video_utils.py::TestExtractKeyFrames::test_basic -v
```

### Print Output

```bash
# Show print statements
pytest tests/test_video_utils.py -s

# Show logging output
pytest tests/test_video_utils.py --log-cli-level=DEBUG
```

### Use pdb Debugger

```python
def test_with_debugger():
    """Test with debugger breakpoint."""
    import pdb; pdb.set_trace()
    result = function_to_debug()
    assert result is not None
```

Or use pytest's built-in debugger:
```bash
pytest tests/test_video_utils.py --pdb
```

## Best Practices

1. **One Assert Per Test** (when possible)
   ```python
   def test_one_thing():
       assert function() == expected_value
   ```

2. **Descriptive Test Names**
   ```python
   def test_returns_none_when_file_missing():
       # Clear what's being tested
   ```

3. **Arrange-Act-Assert Pattern**
   ```python
   def test_pattern():
       # Arrange
       input_data = setup_test_data()

       # Act
       result = function(input_data)

       # Assert
       assert result == expected
   ```

4. **Use Fixtures for Setup**
   ```python
   @pytest.fixture
   def test_data():
       return {"key": "value"}

   def test_with_fixture(test_data):
       assert test_data["key"] == "value"
   ```

5. **Mock External Dependencies**
   - Don't make real API calls in tests
   - Don't depend on external services
   - Use fixtures for expensive operations

6. **Test Edge Cases**
   - Empty inputs
   - None values
   - Maximum values
   - Invalid inputs
   - Error conditions

## Troubleshooting

### Tests Failing Locally But Passing in CI

- Check Python version matches CI
- Check environment variables
- Check file paths (use `os.path.join`)
- Check for test order dependencies

### Flaky Tests

- Use `pytest-randomly` to detect order dependencies
- Avoid time-based assertions
- Use proper mocking
- Reset state between tests

### Low Coverage

```bash
# Find uncovered lines
pytest tests/ --cov=. --cov-report=term-missing

# Generate HTML report for detailed view
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

Add tests for:
- Error handling branches
- Edge cases
- Helper functions
- Configuration logic

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [Python Testing with pytest](https://pythontest.com/pytest-book/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
