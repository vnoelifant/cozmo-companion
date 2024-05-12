import pytest
from mockito import when

from cozmo_companion.assistant import (
    VoiceAssistant,
    Sentiment,
    RequestType,
    get_feedback_inquiry,
    is_feedback_inquiry_present,
)


@pytest.fixture
def assistant():
    """Fixture that provides a basic instance of VoiceAssistant without any prior configuration."""
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
    Provides a VoiceAssistant instance with mocked I/O operations to simulate user interactions.
    This fixture is useful for tests that require simulating speech-to-text and text-to-speech without actual I/O.
    """
    assistant = VoiceAssistant()
    assistant.last_sentiment = Sentiment.NEUTRAL  # Initialize last sentiment to NEUTRAL
    when(assistant)._speak(...)  # Mock the speak method to simulate interaction
    when(assistant)._listen().thenReturn("I feel sad", "Tell me a joke")
    return assistant


# Test class for basic functionalities
class TestBasicFunctionality:
    """
    A test suite for verifying the basic functionalities of the VoiceAssistant,
    focusing on user request categorization and expected bot feedback inquiry response based on user sentiment and request type.
    """

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


# Test class for integration scenarios
@pytest.mark.integration
class TestIntegrationScenarios:
    """
    A test suite for evaluating the integrated behavior of the VoiceAssistant, focusing on its ability to
    dynamically and contextually interact with users. This includes accurately detecting sentiments,
    incorporating relevant feedback based on ongoing interactions, and adapting responses according to the
    conversational context and user's emotional state.

    These tests ensure that the VoiceAssistant can:
    - Accurately detect and interpret the sentiment expressed in user inputs.
    - Identify whether a feedback inquiry is appropriately included in the assistant's responses.
    - Manage a sequence of interactions effectively, applying correct sentiment analysis and
      generating contextually appropriate feedback inquiries, simulating realistic user interactions.

    This suite is crucial for validating the integration of sentiment analysis, feedback mechanisms,
    and overall conversational logic in scenarios that mimic real-world usage, ensuring the assistant's
    reliability and accuracy in user interactions.
    """

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
        Tests verification of sentiment detection based on user input.
        """
        sentiment = configured_assistant.detect_sentiment(user_input)
        assert (
            sentiment == expected
        ), f"Detected sentiment does not match expected. Expected {expected}, got {sentiment}"

    @pytest.mark.parametrize(
        "user_input, expected",
        [
            ("Can you tell me a joke?", RequestType.JOKE),
            ("Can you show me a picture?", RequestType.PICTURE),
            ("Can you tell me a motivational quote", RequestType.MOTIVATIONAL_QUOTE),
        ],
    )
    def test_classify_user_request(self, configured_assistant, user_input, expected):
        """
        Tests verification of user request classification based on user input.
        """
        user_input_lower = user_input.lower()
        user_request = configured_assistant.classify_user_request(user_input_lower)
        assert (
            user_request == expected
        ), f"User request type does not match expected. Expected {expected}, got {user_request}"

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

    @pytest.mark.asyncio
    async def test_user_interaction_flow_with_sentiment_context(
        self, configured_assistant
    ):
        """Test the interaction flow where user sentiment influences subsequent responses."""
        # Simulate user expressing sadness
        user_input_sad = "I am feeling very sad today."
        sentiment_sad = configured_assistant.detect_sentiment(user_input_sad)
        response_sad = await configured_assistant._generate_response(user_input_sad)

        # Ensure the sentiment is detected correctly and response is generated
        assert (
            sentiment_sad == Sentiment.NEGATIVE
        ), "The detected sentiment should be NEGATIVE."
        assert (
            response_sad is not None
        ), "The assistant should provide a response to sadness."

        # Ensure sentiment is stored for subsequent use
        assert (
            configured_assistant.last_sentiment == Sentiment.NEGATIVE
        ), "The assistant should store the last sentiment."

        # Simulate user asking for a joke after expressing sadness
        user_input_joke = "Can you tell me a joke?"
        user_request_type = configured_assistant.categorize_user_request(
            user_input_joke
        )
        response_joke = await configured_assistant._generate_response(user_input_joke)

        # Ensure the joke request is categorized correctly
        assert (
            user_request_type == "joke"
        ), "The user request should be categorized as 'joke'."

        # Verify the joke response is influenced by the previous sad sentiment
        feedback_inquiry = "Did that joke help cheer you up a bit?"
        assert (
            feedback_inquiry in response_joke
        ), "The response to the joke should be influenced by the previous sad sentiment."

        # Validate the full conversation for accuracy
        expected_history = [
            {"role": "user", "content": user_input_sad},
            {"role": "gpt", "content": response_sad},
            {"role": "user", "content": user_input_joke},
            {"role": "gpt", "content": response_joke},
        ]
        assert (
            configured_assistant.conversation_history == expected_history
        ), "The conversation history should correctly reflect the interaction sequence."
