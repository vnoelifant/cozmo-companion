import pytest

from cozmo_companion.assistant import (
    VoiceAssistant,
    Sentiment,
    get_feedback_inquiry,
    is_feedback_inquiry_present,
)


@pytest.fixture
def assistant():
    return VoiceAssistant()


@pytest.fixture
def configured_assistant():
    """Fixture to provide a Voice Assistant with Marvin configured."""
    assistant = VoiceAssistant()
    assistant._configure_marvin_settings()  # Load Marvin settings
    return assistant


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


@pytest.mark.integration
@pytest.mark.parametrize(
    "user_input, expected",
    [
        ("I feel amazing!", Sentiment.POSITIVE),
        ("I feel horrible!", Sentiment.NEGATIVE),
        ("I feel okay.", Sentiment.NEUTRAL),
    ],
)
def test_detect_sentiment(configured_assistant, user_input, expected):
    """
    Integration test to verify sentiment detection of user input.
    """
    sentiment = configured_assistant.detect_sentiment(user_input)
    assert (
        sentiment == expected
    ), f"Detected sentiment does not match expected. Expected {expected}, got {sentiment}"


@pytest.mark.parametrize(
    "bot_text, expected",
    [
        ("North America is a continent. Did this answer help you?", True),
        ("South America is a continent. Did this answer surprise you?", True),
        ("Soccer is the most popular sports in Germany.", False),
    ],
)
def test_is_feedback_inquiry_present(configured_assistant, bot_text, expected):
    """
    Test detection of feedback inquiries within the bot's response text.
    """
    feedback_present = is_feedback_inquiry_present(bot_text)
    assert (
        feedback_present == expected
    ), "Feedback inquiry presence should match the expected outcome."


@pytest.mark.parametrize(
    "user_request_type, user_sentiment, expected_output",
    [
        # Test for recognized types ("joke", "picture", "motivational_quote")
        ("joke", Sentiment.POSITIVE, "Did that joke make you smile?"),
        ("joke", Sentiment.NEGATIVE, "Did that joke help cheer you up a bit?"),
        ("joke", Sentiment.NEUTRAL, "What did you think of that joke?"),
        (
            "joke",
            None,
            "How did you like that joke?",
        ),  # Default sentiment response for jokes
        ("picture", Sentiment.POSITIVE, "Did that picture make you smile?"),
        ("picture", Sentiment.NEGATIVE, "Did that picture help cheer you up a bit?"),
        ("picture", Sentiment.NEUTRAL, "What did you think of that picture?"),
        (
            "picture",
            None,
            "How did you like that picture?",
        ),  # Default sentiment response for pictures
        ("motivational_quote", Sentiment.POSITIVE, "Did that quote make you smile?"),
        ("motivational_quote", Sentiment.NEGATIVE, "Did that quote uplift you?"),
        ("motivational_quote", Sentiment.NEUTRAL, "What did you think of that quote?"),
        (
            "motivational_quote",
            None,
            "How did you like that quote?",
        ),  # Default sentiment response for quotes
        # Default request type response for an unrecognized type, testing all sentiments
        ("unknown_type", Sentiment.POSITIVE, "Was this information helpful to you?"),
        ("unknown_type", Sentiment.NEGATIVE, "Was this information helpful to you?"),
        ("unknown_type", Sentiment.NEUTRAL, "Was this information helpful to you?"),
        (
            "unknown_type",
            None,
            "Was this information helpful to you?",
        ),  # Default sentiment response for unknown types
    ],
)
def test_get_feedback_inquiry(user_request_type, user_sentiment, expected_output):
    """
    Test that the get_feedback_inquiry function returns the correct feedback inquiry
    based on the user's request type and sentiment.
    """
    result = get_feedback_inquiry(user_request_type, user_sentiment)
    assert (
        result == expected_output
    ), f"Expected '{expected_output}', but got '{result}'"
