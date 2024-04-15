import pytest
import os
from decouple import config


@pytest.fixture(scope="session")
def setup_ibm_env():
    # Set real API keys and URLs for the test session
    os.environ["IAM_APIKEY_STT"] = config("IAM_APIKEY_STT")
    os.environ["URL_STT"] = config("URL_STT")
    os.environ["IAM_APIKEY_TTS"] = config("IAM_APIKEY_TTS")
    os.environ["URL_TTS"] = config("URL_TTS")
