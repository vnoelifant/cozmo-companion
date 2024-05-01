import os

import pytest

from cozmo_companion.assistant import VoiceAssistant


@pytest.fixture(scope="session")
def assistant(setup_ibm_env):
    # Ensures environment variables are set and returns a new instance of VoiceAssistant
    return VoiceAssistant()


@pytest.mark.integration
class TestIBMServiceInitialization:
    @pytest.mark.parametrize(
        "api_key_env, url_env, service_name",
        [
            ("IAM_APIKEY_STT", "URL_STT", "Speech to Text"),
            ("IAM_APIKEY_TTS", "URL_TTS", "Text to Speech"),
        ],
    )
    def test_service_initialization(
        self, assistant, setup_ibm_env, api_key_env, url_env, service_name
    ):
        service = assistant._initialize_ibm_service(
            os.environ[api_key_env], os.environ[url_env]
        )
        assert service is not None, f"Failed to initialize {service_name}"
        assert (
            service.service_url == os.environ[url_env]
        ), f"{service_name} URL is incorrect"

    @pytest.mark.parametrize(
        "api_key_env, service_name",
        [("IAM_APIKEY_STT", "Speech to Text"), ("IAM_APIKEY_TTS", "Text to Speech")],
    )
    def test_invalid_url_handling(
        self, assistant, setup_ibm_env, api_key_env, service_name
    ):
        invalid_url = "https://api.ibm.com/incorrect-path"
        with pytest.raises(ValueError) as exc_info:
            assistant._initialize_ibm_service(os.environ[api_key_env], invalid_url)
        assert "Invalid service URL" in str(
            exc_info.value
        ), f"{service_name} should not initialize with incorrect URL keywords"
