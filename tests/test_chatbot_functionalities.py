import pytest

from cozmo_companion.assistant import (
    VoiceAssistant,
    get_feedback_inquiry,
    is_feedback_inquiry_present,
)


@pytest.fixture
def assistant():
    return VoiceAssistant()


@pytest.mark.parametrize(
    "user_input, expected",
    [
        ("Can you tell me a joke?", "joke"),
        ("Can you show me a picture?", "picture"),
        ("I like talking to you", "general"),
    ],
)
def test_request_categorization(assistant, user_input, expected):
    """
    Test the categorization of user requests based on the input text.
    """
    request_type = assistant.categorize_user_request(user_input)
    assert (
        request_type == expected
    ), "Request categorization should match the expected outcome."


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
