import os

import marvin
import pytest

from cozmo_companion.assistant import VoiceAssistant


@pytest.mark.integration
@pytest.mark.usefixtures("setup_marvin_env")
def test_configure_marvin_settings():
    assistant = VoiceAssistant()
    assistant._configure_marvin_settings()
    assert (
        marvin.settings.openai.api_key
    ), "Marvin OpenAI API Key is not set or is empty"
    assert (
        marvin.settings.openai.chat.completions.model
        == os.environ["MARVIN_CHAT_COMPLETIONS_MODEL"]
    ), "Failed to set Marvin chat completions model"
