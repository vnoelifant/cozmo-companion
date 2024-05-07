import os
import pytest
from cozmo_companion.assistant import VoiceAssistant
import marvin


@pytest.mark.integration
def test_configure_marvin_settings(setup_marvin_env):
    """
    Test the _configure_marvin_settings method of the VoiceAssistant class to ensure
    it correctly sets up the Marvin settings based on environment variables.

    This integration test checks that:
    - The OpenAI API key is correctly set in the Marvin settings.
    - The chat completions model in Marvin settings matches the expected model
      specified in the environment variables.

    Args:
        setup_marvin_env (fixture): A pytest fixture that sets up the Marvin environment
                                    necessary for the test.
    """
    # Create an instance of VoiceAssistant
    assistant = VoiceAssistant()

    # Configure Marvin settings using the method under test
    assistant._configure_marvin_settings()

    # Assert that the Marvin OpenAI API key is properly set (not None or empty)
    assert (
        marvin.settings.openai.api_key
    ), "Marvin OpenAI API Key is not set or is empty"

    # Assert that the Marvin chat completions model is set to the expected value from the environment
    assert (
        marvin.settings.openai.chat.completions.model
        == os.environ["MARVIN_CHAT_COMPLETIONS_MODEL"]
    ), "Failed to set Marvin chat completions model correctly according to the environment variable"
