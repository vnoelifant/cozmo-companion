"""
import pytest

@pytest.fixture(autouse=True)
def mock_speech_settings(mocker):
    mocker.patch('marvin.settings.openai.audio.speech.voice', 'echo')



import pytest
from decouple import config


@pytest.fixture(autouse=True, scope="session")
def override_voice_variable(monkeypatch):
    original_voice = config("VOICE")
    monkeypatch.setenv("VOICE", "echo")  # Set to a valid value for `marvin`
    yield
    monkeypatch.setenv("VOICE", original_voice)  # Reset after tests

"""
import sys
from pathlib import Path

# Add the project root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))
