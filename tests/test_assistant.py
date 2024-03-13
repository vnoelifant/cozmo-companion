import pytest
import marvin

from src.cozmo_companion.assistant import (
    is_feedback_inquiry_present,
    get_feedback_inquiry,
    VoiceAssistant,
)


def test_configure_services(monkeypatch):
    # Set up the monkeypatch for the environment variables
    monkeypatch.setenv("MARVIN_OPENAI_API_KEY", "fake_api_key")
    monkeypatch.setenv("MARVIN_CHAT_COMPLETIONS_MODEL", "fake_model")

    # Create an instance of your class
    assistant = VoiceAssistant()

    # Invoke the method that configures services
    assistant._configure_services()

    # Assert that marvin settings were correctly assigned from environment variables
    assert marvin.settings.openai.api_key.get_secret_value() == "fake_api_key"
    assert marvin.settings.openai.chat.completions.model == "fake_model"


@pytest.mark.parametrize(
    "bot_text, expected",
    [
        pytest.param(
            "North America is a continent. Did this answer help you?",
            True,
            id="Question 1",
        ),
        pytest.param(
            "South America is a continent. Did this answer surprise you?",
            True,
            id="Question 2",
        ),
        pytest.param(
            "Soccer is the most popular sports in Germany.", False, id="No question"
        ),
    ],
)
def test_is_feedback_inquiry_present(bot_text, expected):
    feedback_present = is_feedback_inquiry_present(bot_text)

    assert feedback_present == expected


@pytest.mark.parametrize(
    "user_request_type, user_sentiment, expected",
    [
        pytest.param(
            "joke", "positive", "Did that joke make you smile?", id="Positive"
        ),
        pytest.param(
            "joke", "negative", "Did that joke help cheer you up a bit?", id="Negative"
        ),
        pytest.param(
            "joke", "neutral", "What did you think of that joke?", id="Neutral"
        ),
    ],
)
def test_get_feedback_inquiry(user_request_type, user_sentiment, expected):
    feedback_inquiry = get_feedback_inquiry(user_request_type, user_sentiment)

    assert feedback_inquiry == expected
