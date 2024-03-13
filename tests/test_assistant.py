import pytest
import marvin
from datetime import datetime
# from unittest.mock import patch

from src.cozmo_companion.assistant import (
    is_feedback_inquiry_present,
    get_feedback_inquiry,
    Sentiment,
    VoiceAssistant,
    SpeechToTextV1,
    TextToSpeechV1,
)


def test_ibm_services_initialization(monkeypatch):
    monkeypatch.setenv("IAM_APIKEY_STT", "fake_stt_api_key")
    monkeypatch.setenv("URL_STT", "fake_stt_url")
    monkeypatch.setenv("IAM_APIKEY_TTS", "fake_tts_api_key")
    monkeypatch.setenv("URL_TTS", "fake_tts_url")

    assistant = VoiceAssistant()

    assert isinstance(assistant.SPEECH_TO_TEXT, SpeechToTextV1)
    assert isinstance(assistant.TEXT_TO_SPEECH, TextToSpeechV1)


def test_marvin_services_initialization(monkeypatch):
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


def test_wav_file_creation(mocker, tmp_path):
    # Generate the expected timestamp and filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    expected_filename = f"user_{timestamp}.wav"
    expected_path = tmp_path / "wav_output" / expected_filename

    # Use mocker to patch os.getcwd to return the path of tmp_path
    mocker.patch("os.getcwd", return_value=str(tmp_path))

    # Create the WAV file using the patched current working directory
    actual_path = VoiceAssistant._create_wav_file(prefix="user")

    # Assert that file was created correctly
    assert actual_path == str(expected_path)
    assert expected_path.parent.exists()


def test_conversation_history_update(mocker):
    mocker.patch("assistant.VoiceAssistant._listen", return_value="Hello")
    mocker.patch(
        "assistant.VoiceAssistant._generate_response", return_value="Hi there!"
    )

    assistant = VoiceAssistant()
    assistant.start_session()

    assert assistant.conversation_history[0]["role"] == "user"
    assert assistant.conversation_history[0]["content"] == "Hello"
    assert assistant.conversation_history[1]["role"] == "gpt"
    assert assistant.conversation_history[1]["content"] == "Hi there!"


@pytest.mark.parametrize(
    "request_type, expected_response",
    [
        ("joke", "Requesting an uplifting joke."),
        ("motivational_quote", "Requesting an inspiring quote."),
        ("other", "Requesting an uplifting response."),
    ],
)
def test_response_generation_with_negative_sentiment(
    mocker, request_type, expected_response
):
    mocker.patch(
        "assistant.VoiceAssistant.detect_sentiment", return_value=Sentiment.NEGATIVE
    )
    mocker.patch(
        "assistant.VoiceAssistant.categorize_user_request", return_value=request_type
    )

    assistant = VoiceAssistant()
    response = assistant._generate_response("Tell me a joke.")

    assert expected_response in response


def test_start_session(mocker):
    # Simulate scenario where assistant hears "Hello", generates a response "Hi there!",
    # and then hears "exit" to end the session
    mocker.patch("assistant.VoiceAssistant._listen", side_effect=["Hello", "exit"])
    mocker.patch(
        "assistant.VoiceAssistant._generate_response", return_value="Hi there!"
    )
    mocker.patch("assistant.VoiceAssistant._speak")

    assistant = VoiceAssistant()
    assistant.start_session()

    # Assert _speak was called with the greeting and goodbye messages
    assistant._speak.assert_has_calls(
        [
            mocker.call("Hello! Chat with GPT and I will speak its responses!"),
            mocker.call("Goodbye!"),
        ]
    )


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
