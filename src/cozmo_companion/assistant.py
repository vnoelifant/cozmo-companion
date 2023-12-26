import logging
import os
from datetime import datetime
from enum import Enum

import marvin
from decouple import config
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import ApiException, SpeechToTextV1, TextToSpeechV1
from marvin import AIApplication, ai_classifier, ai_fn
from marvin.tools import tool
from pydub import AudioSegment
from pydub.playback import play

from .recorder import Recorder

logging.basicConfig(level=logging.INFO)

# Watson Speech to Text Configuration
CONTENT_TYPE = config("CONTENT_TYPE", default="audio/wav")
WORD_ALTERNATIVE_THRESHOLDS = config(
    "WORD_ALTERNATIVE_THRESHOLDS", default=0.9, cast=float
)
KEYWORDS = config("KEYWORDS", default="hey,hi,watson,friend,meet").split(",")
KEYWORDS_THRESHOLD = config("KEYWORDS_THRESHOLD", default=0.5, cast=float)
MAX_TOKENS = config("MAX_TOKENS", default=1000, cast=int)
TEMPERATURE = config("TEMPERATURE", default=1.2, cast=float)

# Watson Text to Speech Configuration
AUDIO_FORMAT = config("AUDIO_FORMAT", default="audio/wav")
VOICE = config("VOICE", default="en-US_AllisonV3Voice")


@ai_fn
def is_tokens_in_gpt_response(bot_text: str, bot_tokens: list[str]) -> bool:
    """
    Checks if any tokens exist already in bot response message

    Args:
        bot_text: The text provided by the bot.
        bot_tokens: Tokens to check in the bot's response.
    Returns:
        bool: True if any tokens exist in bot message and false otherwise
    """
    return any(token in bot_text for token in bot_tokens)


@tool
def get_feedback_inquiry(user_text: str, user_tokens: list[str]) -> None | str:
    """
    Checks if the user's text contains specific feedback inquiry triggers and
    returns an appropriate inquiry string from the bot based on the content.

    Args:
        user_text: The text provided by the user.
        user_tokens: Tokens to check in user text

    Returns:
        None | str: The feedback inquiry string or None if no match is found.
    """
    for user_token in user_tokens:
        if user_token in user_text.lower():
            return " Did this help put a smile to your face?"
    return None


# Enum class for sentiment analysis
@ai_classifier
class Sentiment(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class VoiceAssistant:

    """
    A class to represent a voice assistant capable of
    listening to voice input, generating a GPT response,
    and speaking the GPT response.
    """

    def __init__(self):
        """Initialize the VoiceAssistant and its services."""
        self._configure_services()
        # Setting up the chatbot with description and tools
        self.chatbot = AIApplication(
            description=(
                "A friendly, supportive chatbot."
                "It always provides an empathetic response when it detects"
                "negative sentiment."
            ),
            tools=[get_feedback_inquiry],
        )
        # Initializing conversation history to store user and bot interactions
        self.conversation_history = []

    def _configure_services(self):
        """Configure and initialize external services (IBM, Marvin, etc.)."""
        # Initialize IBM services for speech-to-text and text-to-speech
        self.SPEECH_TO_TEXT = self._initialize_ibm_service(
            config("IAM_APIKEY_STT"), config("URL_STT")
        )
        self.TEXT_TO_SPEECH = self._initialize_ibm_service(
            config("IAM_APIKEY_TTS"), config("URL_TTS")
        )
        # Setting up Marvin settings
        marvin.settings.openai.api_key = config("MARVIN_OPENAI_API_KEY")
        marvin.settings.llm_model = config("MARVIN_LLM_MODEL")

    def _initialize_ibm_service(self, api_key, url):
        """
        Helper method to initialize IBM services.
        :param api_key: The API key for the service.
        :param url: The URL for the service.
        :return: Initialized IBM service.
        """
        # Creating an IAM authenticator using the provided API key
        authenticator = IAMAuthenticator(api_key)
        service = None
        # Determine the type of service based on the URL and initialize it
        if "speech-to-text" in url:
            service = SpeechToTextV1(authenticator=authenticator)
        elif "text-to-speech" in url:
            service = TextToSpeechV1(authenticator=authenticator)
        else:
            raise ValueError(
                f"Invalid service URL: {url}. Expected 'speech-to-text' "
                f"or 'text-to-speech' in the URL."
            )
        service.set_service_url(url)
        return service

    @staticmethod
    def _create_wav_file(prefix=""):
        """Create a WAV file in the designated directory based on timestamp."""
        # Check if the directory exists, if not create it
        if not os.path.exists("wav_output"):
            os.makedirs("wav_output")

        # Generate filename based on current timestamp and provided prefix
        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{curr_time}.wav"
        speech_file = os.path.join(os.getcwd(), "wav_output", filename)

        return speech_file

    def _listen(self):
        """Record audio and transcribe the recorded speech."""
        # Create a WAV file to store the user's speech
        user_speech_file = VoiceAssistant._create_wav_file(prefix="user")

        logging.info("Starting recording process")
        # Initialize the recorder
        recorder = Recorder(user_speech_file)

        logging.info("Please say something to the microphone\n")
        # Start recording
        recorder.record()

        logging.info("Transcribing audio....\n")
        # Transcribe the recorded audio using IBM's Speech-to-Text service
        try:
            with open((user_speech_file), "rb") as audio:
                speech_result = self.SPEECH_TO_TEXT.recognize(
                    audio=audio,
                    content_type=CONTENT_TYPE,
                    word_alternatives_threshold=WORD_ALTERNATIVE_THRESHOLDS,
                    keywords=KEYWORDS,
                    keywords_threshold=KEYWORDS_THRESHOLD,
                ).get_result()
                # Check if there are any results in the transcription
                if speech_result["results"]:
                    # Extract the transcribed text from the result
                    result_alternative = speech_result["results"][0]["alternatives"][0]
                    user_speech_text = result_alternative["transcript"]

                    return user_speech_text
                else:
                    logging.info("No speech detected. Please try again.")
                    return None

        # Handle exceptions from the IBM service
        except ApiException as ex:
            logging.error(f"Method failed with status code {ex.code}: " f"{ex.message}")

    def construct_gpt_prompt(self, text):
        """Construct the GPT-3 prompt based on the user's sentiment."""
        # Detect the sentiment of the user's text
        detected_sentiment = Sentiment(text)
        # If the sentiment is positive, return the user's text
        if detected_sentiment == Sentiment.POSITIVE:
            logging.info("Positive Sentiment Detected...")
            return text
        else:
            # Return text appended with request for an empathetic response
            logging.info("Negative Sentiment Detected...")
            return text + " Requesting an empathetic response."

    def _generate_response(self, text):
        """Generate a GPT response to the user's text input."""
        try:
            # Construct the GPT prompt based on the input text
            gpt_prompt = self.construct_gpt_prompt(text)

            # Update the conversation history with the user's input
            self.conversation_history.append({"role": "user", "content": gpt_prompt})

            # Get the chatbot's response content
            gpt_response = self.chatbot(gpt_prompt).content

            # Generate list of tokens to trigger a feedback inquiry question from GPT
            feedback_inquiry_user_tokens = ["joke", "motivational quote"]
            feedback_inquiry = get_feedback_inquiry(
                gpt_prompt, feedback_inquiry_user_tokens
            )

            # Append specific feedback inquiry to gpt response
            feedback_inquiry_bot_tokens = ["help", "?"]
            if feedback_inquiry and not is_tokens_in_gpt_response(
                gpt_response, feedback_inquiry_bot_tokens
            ):
                gpt_response += feedback_inquiry

            # Update the conversation history with the chatbot's response
            self.conversation_history.append({"role": "gpt", "content": gpt_response})

            # logging.info the conversation log
            logging.info(f"Conversation Log: {self.conversation_history}")

            return gpt_response
        except Exception as e:
            logging.error(f"Error getting GPT completion: {e}", exc_info=True)
            return "I'm sorry, I couldn't process that."

    def _speak(self, text):
        """Convert text input to speech."""
        # Create a WAV file to store the bot's speech
        bot_speech_file = VoiceAssistant._create_wav_file(prefix="bot")
        # Convert the text to speech using IBM's Text-to-Speech service
        try:
            with open(bot_speech_file, "wb") as audio_out:
                audio_out.write(
                    self.TEXT_TO_SPEECH.synthesize(
                        text,
                        voice=VOICE,
                        accept=AUDIO_FORMAT,
                    )
                    .get_result()
                    .content
                )
            # Play the generated speech
            bot_speech_response = AudioSegment.from_wav(bot_speech_file)
            play(bot_speech_response)
        except ApiException as ex:
            # Handle exceptions from the IBM service
            logging.error(
                "Method failed with status code " + str(ex.code) + ": " + ex.message
            )

    def start_session(self):
        """Handle the conversation with the user."""
        # Start the session by speaking a greeting
        self._speak("Hello! Chat with GPT and I will speak its responses!")
        while True:
            # Listen to the user's speech and transcribe it
            user_speech_text = self._listen()
            logging.info(f"User Speech Text: {user_speech_text} \n")

            # Check if user_speech_text is not None
            if user_speech_text:
                # Exit the loop if the user says "exit"
                if "exit" in user_speech_text.strip().lower():
                    self._speak("Goodbye!")
                    break
                # Generate a GPT response for the user's input
                gpt_response_msg = self._generate_response(user_speech_text)
                logging.info(f"GPT Response Message: {gpt_response_msg} \n")
                # Speak the GPT response
                self._speak(gpt_response_msg)
            else:
                # If user_speech_text is None, handle the case appropriately
                logging.info("No valid input received. Please try speaking again.")
                self._speak("I didn't catch that, could you please repeat?")
