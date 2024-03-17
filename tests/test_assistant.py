import pytest
import marvin
from datetime import datetime
from ibm_watson import SpeechToTextV1, TextToSpeechV1, ApiException
from src.cozmo_companion.assistant import (
    is_feedback_inquiry_present,
    get_feedback_inquiry,
    Sentiment,
    VoiceAssistant,
    VOICE,
    AUDIO_FORMAT,
)


def test_configure_services(mocker):
    # Mock config to return specific values for given calls
    mocker.patch(
        "decouple.config",
        side_effect=lambda var_name, default=None: {
            "IAM_APIKEY_STT": "fake_stt_api_key",
            "URL_STT": "fake_stt_url",
            "IAM_APIKEY_TTS": "fake_tts_api_key",
            "URL_TTS": "fake_tts_url",
            "MARVIN_OPENAI_API_KEY": "fake_api_key",
            "MARVIN_CHAT_COMPLETIONS_MODEL": "fake_model",
        }.get(var_name, default),
    )

    # Mock IAMAuthenticator
    mock_authenticator = mocker.patch(
        "ibm_cloud_sdk_core.authenticators.IAMAuthenticator"
    )
    # Mock SpeechToTextV1 and TextToSpeechV1
    mock_stt = mocker.patch("ibm_watson.SpeechToTextV1")
    mock_tts = mocker.patch("ibm_watson.TextToSpeechV1")

    # Instantiate your assistant and configure services
    assistant = VoiceAssistant()
    assistant._configure_services()

    # Assert that IAMAuthenticator was called with the correct API keys
    mock_authenticator.assert_any_call("fake_stt_api_key")
    mock_authenticator.assert_any_call("fake_tts_api_key")

    # Assert if SpeechToTextV1 and TextToSpeechV1 services were called with the mock authenticator
    mock_stt.assert_called_once()
    mock_tts.assert_called_once()

    # Verify that Marvin settings were set correctly
    assert marvin.settings.openai.api_key == "fake_openai_api_key"
    assert marvin.settings.openai.chat.completions.model == "fake_model"


def test_listen_records_and_transcribes_speech(mocker, tmp_path):
    """
    Test that the _listen method records speech and correctly transcribes it using
    IBM's Speech to Text service.
    """
    # Mock WAV file creation and audio recording
    audio_file_path = tmp_path / "test_audio.wav"
    mocker.patch(
        "src.cozmo_companion.assistant.VoiceAssistant._create_wav_file",
        return_value=str(audio_file_path),
    )
    mocker.patch("src.cozmo_companion.recorder.Recorder.record", return_value=None)
    mocker.patch("builtins.open", mocker.mock_open(read_data=b"fake audio data"))

    # Prepare and mock transcription response
    transcription_response = {
        "results": [{"alternatives": [{"transcript": "hello world"}]}]
    }
    mocker.patch.object(
        SpeechToTextV1,
        "recognize",
        return_value=mocker.Mock(get_result=lambda: transcription_response),
    )

    # Test transcription
    assistant = VoiceAssistant()
    transcribed_text = assistant._listen()
    assert (
        transcribed_text == "hello world"
    ), "Transcribed text should match the mock response."


def test_text_to_speech_conversion(mocker, tmp_path):
    """
    Test the _speak method's ability to convert text to speech and save the audio
    to a WAV file using IBM's Text to Speech service.
    """
    # Mock the TextToSpeechV1 synthesize method
    synthesis_mock = mocker.Mock(get_result=mocker.Mock(content=b"audio data"))
    mock_synthesize = mocker.patch.object(
        TextToSpeechV1, "synthesize", return_value=synthesis_mock
    )

    # Mock WAV file creation
    bot_speech_file = tmp_path / "bot_speech.wav"
    mocker.patch(
        "src.cozmo_companion.assistant.VoiceAssistant._create_wav_file",
        return_value=str(bot_speech_file),
    )
    mocker.patch("pydub.playback.play")

    # Convert text to speech and assert file creation and content
    assistant = VoiceAssistant()
    assistant._speak("Test speech")
    assert bot_speech_file.exists(), "The bot speech file should exist."
    assert (
        bot_speech_file.read_bytes() == b"audio data"
    ), "The content of the bot speech file should match the mocked audio data."

    # Assert synthesize was called with expected arguments
    mock_synthesize.assert_called_once_with(
        text="Test speech", voice=VOICE, accept=AUDIO_FORMAT
    )


def test_wav_file_creation(mocker, tmp_path):
    """
    Test the creation of a WAV file with the expected naming convention and directory.
    """
    # Generate expected filename and path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    expected_filename = f"user_{timestamp}.wav"
    expected_path = tmp_path / "wav_output" / expected_filename

    # Mock os.getcwd() and test WAV file creation
    mocker.patch("os.getcwd", return_value=str(tmp_path))
    actual_path = VoiceAssistant._create_wav_file(prefix="user")
    assert actual_path == str(
        expected_path
    ), "The actual path should match the expected path."
    assert (
        expected_path.parent.exists()
    ), "The parent directory of the WAV file should exist."


def test_conversation_history_update(mocker):
    """
    Test that conversation history is updated correctly during a session.
    """
    # Mock the _listen and _generate_response methods
    mocker.patch(
        "src.cozmo_companion.assistant.VoiceAssistant._listen", return_value="Hello"
    )
    mocker.patch(
        "src.cozmo_companion.assistant.VoiceAssistant._generate_response",
        return_value="Hi there!",
    )

    # Start session and assert conversation history
    assistant = VoiceAssistant()
    assistant.start_session()
    assert (
        assistant.conversation_history[0]["role"] == "user"
    ), "The first entry in conversation history should be from the user."
    assert (
        assistant.conversation_history[0]["content"] == "Hello"
    ), "The first entry's content should match the mocked input."
    assert (
        assistant.conversation_history[1]["role"] == "gpt"
    ), "The second entry in conversation history should be from GPT."
    assert (
        assistant.conversation_history[1]["content"] == "Hi there!"
    ), "The second entry's content should match the mocked response."


@pytest.mark.parametrize(
    "input_text, expected_sentiment",
    [
        ("I am happy", Sentiment.POSITIVE),
        ("I am sad", Sentiment.NEGATIVE),
        ("I am okay", Sentiment.NEUTRAL),
    ],
)
def test_detect_sentiment(input_text, expected_sentiment):
    """
    Test sentiment detection based on input text.
    """
    assistant = VoiceAssistant()
    sentiment = assistant.detect_sentiment(input_text)
    assert (
        sentiment == expected_sentiment
    ), "Detected sentiment should match the expected sentiment."


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
    """
    Test response generation based on negative sentiment and different request types.
    """
    # Mock sentiment detection and request categorization
    mocker.patch(
        "src.cozmo_companion.assistant.VoiceAssistant.detect_sentiment",
        return_value=Sentiment.NEGATIVE,
    )
    mocker.patch(
        "src.cozmo_companion.assistant.VoiceAssistant.categorize_user_request",
        return_value=request_type,
    )

    # Generate response and assert content
    assistant = VoiceAssistant()
    response = assistant._generate_response("I feel horrible. Can you tell me a joke?")
    assert (
        expected_response in response
    ), "Generated response should include the expected content based on negative sentiment and request type."


def test_start_session(mocker):
    """
    Test the overall flow of starting a session, including greeting, handling input, and exiting.
    """
    # Mock user input and responses
    mocker.patch(
        "src.cozmo_companion.assistant.VoiceAssistant._listen",
        side_effect=["Hello", "exit"],
    )
    mocker.patch(
        "src.cozmo_companion.assistant.VoiceAssistant._generate_response",
        return_value="Hi there!",
    )
    mock_speak = mocker.patch("VoiceAssistant._speak")

    # Start session and assert _speak calls
    assistant = VoiceAssistant()
    assistant.start_session()
    mock_speak.assert_has_calls(
        [
            mocker.call("Hello! Chat with GPT and I will speak its responses!"),
            mocker.call("Goodbye!"),
        ],
        "The assistant should greet at the beginning and say goodbye at the end.",
    )


def test_exit_condition_handling(mocker):
    """
    Test handling of the "exit" condition to end a session.
    """
    # Mock the _listen method to simulate "exit" input
    mocker.patch(
        "src.cozmo_companion.assistant.VoiceAssistant._listen", side_effect=["exit"]
    )
    mock_generate_response = mocker.patch(
        "src.cozmo_companion.assistant.VoiceAssistant._generate_response",
        return_value="Goodbye!",
    )
    mock_speak = mocker.patch("src.cozmo_companion.assistant.VoiceAssistant._speak")

    # Trigger session and assert response to "exit"
    assistant = VoiceAssistant()
    assistant.start_session()
    (
        mock_generate_response.assert_called_once_with("exit"),
        "The _generate_response method should be called with 'exit'.",
    )
    (
        mock_speak.assert_called_once_with("Goodbye!"),
        "The assistant should respond with 'Goodbye!' upon exit.",
    )


def test_error_handling_in_generate_response(mocker):
    """
    Test the error handling within the _generate_response method when an exception occurs.
    """
    # Mock _listen to return input and simulate an ApiException during sentiment detection
    mocker.patch(
        "src.cozmo_companion.assistant.VoiceAssistant._listen",
        return_value="Some input",
    )
    mocker.patch(
        "src.cozmo_companion.assistant.VoiceAssistant.detect_sentiment",
        side_effect=ApiException("API error"),
    )
    mock_speak = mocker.patch("src.cozmo_companion.assistant.VoiceAssistant._speak")

    # Trigger session and assert error handling
    assistant_obj = VoiceAssistant()
    assistant_obj.start_session()
    (
        mock_speak.assert_called_with("I'm sorry, I couldn't process that."),
        "The assistant should respond with an apology if an error occurs.",
    )


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
