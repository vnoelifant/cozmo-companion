import pytest

from cozmo_companion.assistant import Sentiment, check_exit_command


# Test class for basic unit test functionalities
@pytest.mark.unit
class TestBasicFunctionality:
    """
    A test suite for verifying the basic functionalities of the VoiceAssistant,
    focusing on user request categorization and expected bot feedback inquiry response based on user sentiment and request type.
    """

    @pytest.mark.parametrize(
        "user_input, expected",
        [
            ("exit", True),
            ("stop", True),
            ("goodbye", True),
            ("hello", False),
        ],
    )
    def test_check_exit_command(self, marvin_assistant, user_input, expected):
        """
        Test the check_exit_command function to verify if the user input signals an exit command.
        """
        is_exit_command = check_exit_command(user_input)
        assert (
            is_exit_command == expected
        ), f"Expected {expected}, but got {is_exit_command}"

    @pytest.mark.parametrize(
        "user_input, expected",
        [
            ("I feel amazing!", Sentiment.POSITIVE),
            ("I feel horrible!", Sentiment.NEGATIVE),
            ("I feel okay.", Sentiment.NEUTRAL),
        ],
    )
    def test_detect_sentiment(self, marvin_assistant, user_input, expected):
        """
        Tests verification of sentiment detection based on user input.
        """
        sentiment = marvin_assistant.detect_sentiment(user_input)
        assert (
            sentiment == expected
        ), f"Detected sentiment does not match expected. Expected {expected}, got {sentiment}"
