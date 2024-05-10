import os
import pytest
from cozmo_companion.assistant import VoiceAssistant


@pytest.fixture(scope="session")
def assistant(setup_ibm_env):
    """
    Pytest fixture that sets up the environment for IBM services and returns a new instance of VoiceAssistant.

    Args:
        setup_ibm_env (fixture): A setup fixture that configures the necessary IBM environment variables.

    Returns:
        VoiceAssistant: An instance of VoiceAssistant configured for IBM service testing.
    """
    # Ensures environment variables are set and returns a new instance of VoiceAssistant
    return VoiceAssistant()


@pytest.mark.integration
class TestIBMServiceInitialization:
    """
    A test suite for verifying the initialization of IBM services within the VoiceAssistant class.
    """

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
        """
        Test the initialization of IBM services by ensuring that the service objects are created correctly with valid API keys and URLs.

        Args:
            assistant (VoiceAssistant): The VoiceAssistant instance provided by the fixture.
            setup_ibm_env (fixture): A fixture to setup IBM environment variables.
            api_key_env (str): The environment variable name for the IBM service's API key.
            url_env (str): The environment variable name for the IBM service's URL.
            service_name (str): The name of the service being tested (e.g., "Speech to Text").
        """
        # Attempt to initialize the IBM service using the provided API key and URL environment variables
        service = assistant._initialize_ibm_service(
            os.environ[api_key_env], os.environ[url_env]
        )

        # Assert the service is successfully initialized and not None
        assert service is not None, f"Failed to initialize {service_name}"
        # Assert the service URL matches the environment configuration
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
        """
        Test the handling of invalid URLs during the initialization of IBM services to ensure proper error handling and validation.

        Args:
            assistant (VoiceAssistant): The VoiceAssistant instance provided by the fixture.
            setup_ibm_env (fixture): A fixture to setup IBM environment variables.
            api_key_env (str): The environment variable name for the IBM service's API key.
            service_name (str): The name of the service being tested for URL validation.
        """
        # Define an invalid URL for testing error handling
        invalid_url = "https://api.ibm.com/incorrect-path"

        # Attempt to initialize the IBM service with an invalid URL and expect a ValueError
        with pytest.raises(ValueError) as exc_info:
            assistant._initialize_ibm_service(os.environ[api_key_env], invalid_url)

        # Assert that the expected error message is raised
        assert "Invalid service URL" in str(
            exc_info.value
        ), f"{service_name} should not initialize with incorrect URL keywords"
