import pytest
import marvin
import src.cozmo_companion.assistant as assistant
from datetime import datetime
from ibm_watson import SpeechToTextV1, TextToSpeechV1
from src.cozmo_companion.assistant import (
    is_feedback_inquiry_present,
    get_feedback_inquiry,
    Sentiment,
    VoiceAssistant,
    CONTENT_TYPE,
    WORD_ALTERNATIVE_THRESHOLDS,
    KEYWORDS,
    KEYWORDS_THRESHOLD,
    VOICE,
    AUDIO_FORMAT,
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


def test_listen_records_and_transcribes_speech(mocker, tmp_path):
    # Mock the creation of a WAV file to a known location
    audio_file_path = tmp_path / "test_audio.wav"
    mocker.patch(
        "src.cozmo_companion.assistant.VoiceAssistant._create_wav_file",
        return_value=str(audio_file_path),
    )

    # Mock Recorder's record method to simulate successful recording
    mock_record = mocker.patch(
        "src.cozmo_companion.recorder.Recorder.record", return_value=None
    )

    # Mock the file open operation to simulate the presence of recorded audio
    mocker.patch("builtins.open", mocker.mock_open(read_data=b"fake audio data"))

    # Prepare a mock speech-to-text response
    transcription_response = {
        "results": [{"alternatives": [{"transcript": "hello world"}]}]
    }

    # Mock SpeechToTextV1's recognize method to return the predefined transcription response
    mock_recognize = mocker.patch.object(
        SpeechToTextV1,
        "recognize",
        return_value=mocker.Mock(get_result=lambda: transcription_response),
    )

    assistant = VoiceAssistant()
    transcribed_text = assistant._listen()

    # Assertions
    assert transcribed_text == "hello world"
    mock_record.assert_called_once()
    mock_recognize.assert_called_once()

    # Assert that the recognize method was called with expected arguments
    mock_recognize.assert_called_once_with(
        audio=mocker.ANY,  # Indicates we're not specifying exactly what the audio file object is
        content_type=CONTENT_TYPE,
        word_alternatives_threshold=WORD_ALTERNATIVE_THRESHOLDS,
        keywords=KEYWORDS,
        keywords_threshold=KEYWORDS_THRESHOLD,
    )


def test_text_to_speech_conversion(mocker, tmp_path):
    # Use mocker.Mock() to specify the return structure of the mock
    synthesis_mock = mocker.Mock(get_result=mocker.Mock(content=b"audio data"))
    mock_synthesize = mocker.patch.object(
        TextToSpeechV1, "synthesize", return_value=synthesis_mock
    )

    # Define the expected file path
    bot_speech_file = tmp_path / "bot_speech.wav"
    mocker.patch(
        "src.cozmo_companion.assistant.VoiceAssistant._create_wav_file",
        return_value=str(bot_speech_file),
    )

    # Mock the play function to prevent actual playback during tests
    mocker.patch("pydub.playback.play")

    test_text = "Test speech"
    assistant = VoiceAssistant()
    assistant._speak(test_text)

    # Assert that the synthesized audio file exists and contains the expected content
    assert bot_speech_file.exists()
    assert bot_speech_file.read_bytes() == b"audio data"

    # Assert that synthesize was called with expected arguments
    mock_synthesize.assert_called_once_with(
        text=test_text, voice=VOICE, accept=AUDIO_FORMAT
    )


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
    "input_text, expected_sentiment",
    [
        ("I am happy", Sentiment.POSITIVE),
        ("I am sad", Sentiment.NEGATIVE),
        ("I am okay", Sentiment.NEUTRAL),
    ],
)
def test_detect_sentiment(input_text, expected_sentiment):
    assistant = VoiceAssistant()
    sentiment = assistant.detect_sentiment(input_text)
    assert sentiment == expected_sentiment


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
    response = assistant._generate_response("I feel horrible. Can you tell me a joke?")

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


def test_exit_condition_handling(mocker):
    # Mock the _listen method to simulate receiving "exit"
    mocker.patch("assistant.VoiceAssistant._listen", side_effect=["exit"])
    # Assuming _generate_response would generate a "Goodbye!" response upon "exit"
    mock_generate_response = mocker.patch(
        "assistant.VoiceAssistant._generate_response", return_value="Goodbye!"
    )
    # Mock the _speak method to avoid actual speech synthesis during the test
    mock_speak = mocker.patch("assistant.VoiceAssistant._speak")

    assistant = VoiceAssistant()
    assistant.start_session()  # Trigger the session, which should handle the "exit" condition

    # Verify that _generate_response was called with "exit", leading to a "Goodbye!" response
    mock_generate_response.assert_called_once_with("exit")
    # Verify that _speak was called with the "Goodbye!" response
    mock_speak.assert_called_once_with("Goodbye!")


def test_error_handling_in_generate_response(mocker):
    mocker.patch("assistant.VoiceAssistant._listen", return_value="Some input")
    mocker.patch(
        "assistant.VoiceAssistant.detect_sentiment",
        side_effect=assistant.ApiException("API error"),
    )
    mock_speak = mocker.patch("assistant.VoiceAssistant._speak")

    assistant_obj = VoiceAssistant()
    assistant_obj.start_session()

    mock_speak.assert_called_with("I'm sorry, I couldn't process that.")


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
