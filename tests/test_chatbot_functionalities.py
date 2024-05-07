import pytest
from mockito import when, unstub, verify

from cozmo_companion.assistant import (
    VoiceAssistant,
    Sentiment,
    get_feedback_inquiry,
    is_feedback_inquiry_present,
)


@pytest.fixture
def assistant():
    """Fixture to provide a basic Voice Assistant"""
    return VoiceAssistant()


@pytest.fixture
def configured_assistant():
    """Fixture to provide a Voice Assistant with Marvin configured."""
    assistant = VoiceAssistant()
    assistant._configure_marvin_settings()  # Load Marvin settings
    return assistant


@pytest.fixture
def voice_assistant_with_mocked_io(setup_marvin_env):
    """
    Provides a VoiceAssistant with mocked I/O operations.
    Useful for tests simulating user interaction without real IO.
    """
    assistant = VoiceAssistant()
    assistant.last_sentiment = Sentiment.NEUTRAL  # Initialize last sentiment to NEUTRAL
    when(assistant)._speak(...)  # Mock the speak method to simulate interaction
    when(assistant)._listen().thenReturn("I feel sad", "Tell me a joke")
    return assistant


# Test class for basic functionalities
class TestBasicFunctionality:
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
            (
                "picture",
                Sentiment.NEGATIVE,
                "Did that picture help cheer you up a bit?",
            ),
            ("picture", Sentiment.NEUTRAL, "What did you think of that picture?"),
            (
                "picture",
                None,
                "How did you like that picture?",
            ),  # Default sentiment response for pictures
            (
                "motivational_quote",
                Sentiment.POSITIVE,
                "Did that quote make you smile?",
            ),
            ("motivational_quote", Sentiment.NEGATIVE, "Did that quote uplift you?"),
            (
                "motivational_quote",
                Sentiment.NEUTRAL,
                "What did you think of that quote?",
            ),
            (
                "motivational_quote",
                None,
                "How did you like that quote?",
            ),  # Default sentiment response for quotes
            # Default request type response for an unrecognized type, testing all sentiments
            (
                "unknown_type",
                Sentiment.POSITIVE,
                "Was this information helpful to you?",
            ),
            (
                "unknown_type",
                Sentiment.NEGATIVE,
                "Was this information helpful to you?",
            ),
            ("unknown_type", Sentiment.NEUTRAL, "Was this information helpful to you?"),
            (
                "unknown_type",
                None,
                "Was this information helpful to you?",
            ),  # Default sentiment response for unknown types
        ],
    )
    def test_get_feedback_inquiry(
        self, user_request_type, user_sentiment, expected_output
    ):
        """
        Test that the get_feedback_inquiry function returns the correct feedback inquiry
        based on the user's request type and sentiment.
        """
        result = get_feedback_inquiry(user_request_type, user_sentiment)
        assert (
            result == expected_output
        ), f"Expected '{expected_output}', but got '{result}'"

    @pytest.mark.parametrize(
        "user_input, expected",
        [
            ("Can you tell me a joke?", "joke"),
            ("Can you show me a picture?", "picture"),
            ("I like talking to you", "general"),
        ],
    )
    def test_request_categorization(self, assistant, user_input, expected):
        """
        Test the categorization of user requests based on the input text.
        """
        request_type = assistant.categorize_user_request(user_input)
        assert (
            request_type == expected
        ), "Request categorization should match the expected outcome."


# Test class for integration scenarios
@pytest.mark.integration
class TestIntegrationScenarios:
    @pytest.mark.parametrize(
        "user_input, expected",
        [
            ("I feel amazing!", Sentiment.POSITIVE),
            ("I feel horrible!", Sentiment.NEGATIVE),
            ("I feel okay.", Sentiment.NEUTRAL),
        ],
    )
    def test_detect_sentiment(self, configured_assistant, user_input, expected):
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
    def test_is_feedback_inquiry_present(
        self, configured_assistant, bot_text, expected
    ):
        """
        Test detection of feedback inquiries within the bot's response text.
        """
        feedback_present = is_feedback_inquiry_present(bot_text)
        assert (
            feedback_present == expected
        ), "Feedback inquiry presence should match the expected outcome."

    def test_sentiment_and_response_flow(self, voice_assistant_with_mocked_io):
        """
        Tests the Voice Assistant's response logic using actual implementations of sentiment detection,
        feedback inquiry generation, and feedback presence checking,
        with mocked I/O (text-to-speech and speech-to-text) to simulate user interaction.
        """
        # First interaction where the user expresses sadness
        user_input1 = (
            voice_assistant_with_mocked_io._listen()
        )  # Should return "I feel sad"
        sentiment1 = voice_assistant_with_mocked_io.detect_sentiment(user_input1)
        assert (
            sentiment1 == Sentiment.NEGATIVE
        ), "Sentiment detection failed; expected NEGATIVE."
        voice_assistant_with_mocked_io.last_sentiment = sentiment1

        response1 = voice_assistant_with_mocked_io._generate_response(user_input1)
        voice_assistant_with_mocked_io._speak(response1)

        # Second interaction where the user asks for a joke
        user_input2 = (
            voice_assistant_with_mocked_io._listen()
        )  # Should return "Tell me a joke"
        user_request_type = voice_assistant_with_mocked_io.categorize_user_request(
            user_input2
        )
        assert (
            user_request_type == "joke"
        ), "Request categorization failed; expected 'joke'."

        response2 = voice_assistant_with_mocked_io._generate_response(user_input2)
        voice_assistant_with_mocked_io._speak(response2)

        # Validate that feedback inquiry is included correctly based on the sentiment
        feedback_inquiry = get_feedback_inquiry(
            user_request_type, voice_assistant_with_mocked_io.last_sentiment
        )
        feedback_present = is_feedback_inquiry_present(response2)
        assert (
            not feedback_present
        ), "Feedback should not have been present initially in the generated response."
        assert (
            feedback_inquiry in response2
        ), "Feedback inquiry should be included in the response based on the previous NEGATIVE sentiment."

        # Ensure that the speak method was invoked with the expected responses
        verify(voice_assistant_with_mocked_io)._speak(response1)
        verify(voice_assistant_with_mocked_io)._speak(response2)
        # Ensure that the listen method was invoked correctly
        verify(voice_assistant_with_mocked_io, times=2)._listen()

        # Clean up mocks
        unstub()