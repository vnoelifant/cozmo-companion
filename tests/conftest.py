import pytest
import os
from decouple import config
from mockito import when
from cozmo_companion.assistant import VoiceAssistant


@pytest.fixture(scope="session")
def setup_ibm_env():
    # Set real IBM API keys and URLs for the test session
    os.environ["IAM_APIKEY_STT"] = config("IAM_APIKEY_STT")
    os.environ["URL_STT"] = config("URL_STT")
    os.environ["IAM_APIKEY_TTS"] = config("IAM_APIKEY_TTS")
    os.environ["URL_TTS"] = config("URL_TTS")


@pytest.fixture(scope="session")
def setup_marvin_env():
    # Set real Marvin OpenAI API Key and chat completions model for the test session
    os.environ["MARVIN_OPENAI_API_KEY"] = config("MARVIN_OPENAI_API_KEY")
    os.environ["MARVIN_CHAT_COMPLETIONS_MODEL"] = config(
        "MARVIN_CHAT_COMPLETIONS_MODEL"
    )


@pytest.fixture
def marvin_assistant(setup_marvin_env):
    """Fixture to provide a Voice Assistant with Marvin configured."""
    assistant = VoiceAssistant()
    assistant._configure_marvin_settings()  # Load Marvin settings
    return assistant


@pytest.fixture
def basic_assistant():
    """Fixture that provides a basic instance of VoiceAssistant without any prior configuration."""
    return VoiceAssistant()


@pytest.fixture(scope="session")
def ibm_assistant(setup_ibm_env):
    """
    Pytest fixture that sets up the environment for IBM services and returns a new instance of VoiceAssistant.

    Args:
        setup_ibm_env (fixture): A setup fixture that configures the necessary IBM environment variables.

    Returns:
        VoiceAssistant: An instance of VoiceAssistant configured for IBM service testing.
    """
    # Ensures environment variables are set and returns a new instance of VoiceAssistant
    return VoiceAssistant()


@pytest.fixture
def voice_assistant_with_mocked_io(setup_marvin_env):
    """
    Provides a VoiceAssistant instance with mocked I/O operations to simulate user interactions.
    This fixture is useful for tests that require simulating speech-to-text and text-to-speech without actual I/O.
    """
    assistant = VoiceAssistant()
    # assistant.last_sentiment = Sentiment.NEUTRAL  # Initialize last sentiment to NEUTRAL
    when(assistant)._speak(...)  # Mock the speak method to simulate interaction
    when(assistant)._listen().thenReturn("I feel sad", "Tell me a joke")
    return assistant
