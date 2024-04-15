import pytest
import os
from cozmo_companion.assistant import (
    VoiceAssistant,
    is_feedback_inquiry_present,
    get_feedback_inquiry,
)


@pytest.mark.usefixtures("setup_ibm_env")
def test_initialize_ibm_service():
    assistant = VoiceAssistant()
    stt_service = assistant._initialize_ibm_service(
        os.environ["IAM_APIKEY_STT"], os.environ["URL_STT"]
    )
    tts_service = assistant._initialize_ibm_service(
        os.environ["IAM_APIKEY_TTS"], os.environ["URL_TTS"]
    )

    # Assert that the services are correctly initialized
    assert stt_service is not None, "Failed to initialize Speech to Text Service"
    assert tts_service is not None, "Failed to initialize Text to Speech Service"

    # Verify that the correct URLs are set on the service instances
    assert (
        stt_service.service_url == os.environ["URL_STT"]
    ), "STT Service URL is incorrect"
    assert (
        tts_service.service_url == os.environ["URL_TTS"]
    ), "TTS Service URL is incorrect"


@pytest.mark.parametrize(
    "bot_text, expected",
    [
        ("North America is a continent. Did this answer help you?", True),
        ("South America is a continent. Did this answer surprise you?", True),
        ("Soccer is the most popular sports in Germany.", False),
    ],
)
def test_is_feedback_inquiry_present(bot_text, expected):
    """
    Test detection of feedback inquiries within the bot's response text.
    """
    feedback_present = is_feedback_inquiry_present(bot_text)
    assert (
        feedback_present == expected
    ), "Feedback inquiry presence should match the expected outcome."


@pytest.mark.parametrize(
    "user_request_type, user_sentiment, expected",
    [
        ("joke", "positive", "Did that joke make you smile?"),
        ("joke", "negative", "Did that joke help cheer you up a bit?"),
        ("joke", "neutral", "What did you think of that joke?"),
    ],
)
def test_get_feedback_inquiry(user_request_type, user_sentiment, expected):
    """
    Test generation of feedback inquiries based on user request type and sentiment.
    """
    feedback_inquiry = get_feedback_inquiry(user_request_type, user_sentiment)
    assert (
        feedback_inquiry == expected
    ), "Generated feedback inquiry should match the expected content."
