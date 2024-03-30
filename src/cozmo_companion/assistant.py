import logging
import os
from datetime import datetime
from enum import Enum

import marvin
from decouple import config
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import ApiException, SpeechToTextV1, TextToSpeechV1
from marvin.beta import Application
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
VOICE = config("VOICE", default="en-US_AllisonV3Voice")
# Watson Text to Speech Configuration
AUDIO_FORMAT = config("AUDIO_FORMAT", default="audio/wav")


# Dialogue Constants
DEFAULT_SENTIMENT_RESPONSE = "default_sentiment_response"
DEFAULT_REQUEST_TYPE_RESPONSE = "default_request_type_response"


@marvin.fn
def is_feedback_inquiry_present(bot_text: str) -> bool:
    """
    Analyzes the bot's response {{ bot_text }} to determine if it contains a feedback inquiry.

    The function leverages natural language understanding to interpret the text and
    identify feedback-related phrases, taking into account the context and subtleties
    of the conversation. It examines the text of a chatbot's response to detect if
    it includes phrases or questions that are seeking feedback from the user. Common
    examples of feedback inquiries might include direct questions like "Did that help
    you?" or "Do you want to know more?" as well as subtle cues such as the presence
    of a question mark (?) at the end of a statement.

    Args:
        bot_text: The text of the response provided by the chatbot. This can range
        from answers and statements to jokes or informational content.

    Returns:
        bool: Indicates whether the bot's response includes a feedback inquiry.
            - True: If phrases or questions seeking feedback are found, indicating
                an active attempt by the bot to engage with the user or confirm
                understanding.
            - False: If no feedback-seeking phrases or questions are detected,
                suggesting a straightforward response without solicitation for feedback.
    """
    return False  # Dummy return value for type checking


class Sentiment(Enum):
    """Classifies user text"""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


def get_feedback_inquiry(user_request_type: str, user_sentiment: Sentiment) -> str:
    """
    Generates a feedback inquiry based on the user's request type and sentiment.

    Args:
        user_request_type: A string categorizing the type of the user's request.
        user_sentiment: An enum value representing the sentiment of the user's request.

    Returns:
        str: A tailored feedback inquiry message based on the request type and
        sentiment.
    """
    feedback_inquiries = {
        "joke": {
            Sentiment.POSITIVE: "Did that joke make you smile?",
            Sentiment.NEGATIVE: "Did that joke help cheer you up a bit?",
            Sentiment.NEUTRAL: "What did you think of that joke?",
            DEFAULT_SENTIMENT_RESPONSE: "How was the joke?",
        },
        "picture": "Did you like the picture I sent?",
        "motivational_quote": {
            Sentiment.NEGATIVE: "Did that quote uplift you?",
            DEFAULT_SENTIMENT_RESPONSE: "How did you find that quote?",
        },
        DEFAULT_REQUEST_TYPE_RESPONSE: "Was this information helpful to you?",
    }

    feedback_for_request_type = feedback_inquiries.get(
        user_request_type, feedback_inquiries[DEFAULT_REQUEST_TYPE_RESPONSE]
    )

    if isinstance(feedback_for_request_type, dict):
        feedback_message = feedback_for_request_type.get(
            user_sentiment,
            feedback_for_request_type.get(DEFAULT_SENTIMENT_RESPONSE, ""),
        )
        return str(feedback_message)
    else:
        return str(feedback_for_request_type)


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
        self.chatbot = Application(
            instructions=(
                "A friendly, supportive chatbot."
                "It always provides an empathetic response when it detects"
                "negative sentiment."
            ),
            tools=[get_feedback_inquiry],
        )
        self.last_sentiment = Sentiment.NEUTRAL  # Initialize last sentiment as NEUTRAL
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
        marvin.settings.openai.chat.completions.model = config(
            "MARVIN_CHAT_COMPLETIONS_MODEL"
        )

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

    def categorize_user_request(self, user_input: str) -> str:
        # This function categorizes the user input into different request types.
        if "joke" in user_input.lower():
            return "joke"
        elif "motivational quote" in user_input.lower():
            return "motivational_quote"
        elif "picture" in user_input.lower():
            return "picture"
        # Add more categories as necessary
        else:
            return "general"

    def detect_sentiment(self, user_input: str) -> Sentiment:
        return marvin.classify(user_input, Sentiment)

    def _generate_response(self, user_input: str) -> str:
        """Generate a GPT response to the user's text input."""
        try:
            # Detect the sentiment of the user's current input
            current_sentiment = self.detect_sentiment(user_input)
            # Get the user's request type from the current input
            user_request_type = self.categorize_user_request(user_input)

            # Get the GPT response
            gpt_response = self.chatbot(user_input).content

            # If the last sentiment was negative, customize the response based on request type
            if self.last_sentiment == Sentiment.NEGATIVE and user_request_type in [
                "joke",
                "motivational_quote",
                "picture",
            ]:
                gpt_response += "\nI understand things might be tough. Here's something to brighten your day!"

            # Append feedback inquiry if not already present in the GPT response
            feedback_inquiry = get_feedback_inquiry(
                user_request_type, current_sentiment
            )
            if not is_feedback_inquiry_present(gpt_response):
                gpt_response += " " + feedback_inquiry

            # Update the conversation history with the user's input and the GPT response
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "gpt", "content": gpt_response})

            # Update self.last_sentiment to the current sentiment for use in the next interaction
            self.last_sentiment = current_sentiment

            return gpt_response
        except Exception as e:
            logging.error(f"Error generating response: {e}", exc_info=True)
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
