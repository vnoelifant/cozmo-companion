import pytest
import os
from decouple import config


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
