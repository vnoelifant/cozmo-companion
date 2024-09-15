import logging
import os
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

import marvin
import webbrowser
from decouple import config
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import ApiException, SpeechToTextV1, TextToSpeechV1
from marvin.beta.applications import Application
from pydub import AudioSegment
from pydub.playback import play
from .logging_config import setup_logging

from .recorder import Recorder

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


def send_picture_to_user(user_input: str) -> None:
    """Send a picture to the user based on the user's input."""
    image = marvin.paint(user_input)
    url = image.data[0].url
    webbrowser.open(url)


class Sentiment(Enum):
    """Classifies the sentiment of the user's input."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class SentimentState(BaseModel):
    """Represents the state of the user's sentiment."""

    sentiment: list[Sentiment] = []


@marvin.fn  # type: ignore
def check_exit_command(user_input: str) -> bool:
    """
    Analyzes the user's input to determine if it indicates a desire to end the session.
    This function detects standard exit phrases like 'exit', 'quit', 'goodbye', or 'stop'.

    Args:
        user_input (str): The text input from the user.

    Returns:
        bool: Returns True if an exit command is detected, otherwise False.
    """


class VoiceAssistant:

    """
    A class to represent a voice assistant capable of
    listening to voice input, generating a GPT response,
    and speaking the GPT response.
    """

    def __init__(self):
        """Initialize the VoiceAssistant and its services."""
        # Setup logging
        setup_logging(log_file="logs/chatbot_log.txt")

        # Configure and initialize external services (IBM, Marvin, etc.)
        self._configure_services()
        # Setting up the chatbot with instructions, state, and tools.
        self.chatbot = Application(
            name="Companion",
            model=config("MARVIN_CHAT_COMPLETIONS_MODEL"),
            instructions="""A friendly, supportive chatbot.It always provides an
                empathetic response when it detects negative sentiment. In
                addition, you have a crucial goal to keep track of the user's
                sentiment state, especially negative sentiment, to provide more
                emotionally and contextually aware responses. As an example,
                the user may have specific requests during an interaction after
                expressing negative sentiment. For instance, if the last sentiment
                from the user was negative, and the user asks for a joke, motivational
                quote, or picture in the next interaction, the chatbot should provide
                a feedback inquiry based on the user's request type, checking if the
                joke, quote or picture helped cheer the user up. Note that if the user
                asks for a picture, you will need to use the tool 'send_picture_to_user'
                to send a picture. Another scenario could be that you notice that the
                sentiment transitions from negative to positive, and in this case, you
                should be especially happy and excited for the user, and include these
                emotions in your responses.
                """,
            state=SentimentState(),
            tools=[send_picture_to_user],
        )
        # self.last_sentiment = Sentiment.NEUTRAL  # Initialize last sentiment as NEUTRAL
        # Initializing conversation history to store user and bot interactions
        self.conversation_history = []

    def __str__(self) -> str:
        """Return a string representation of the VoiceAssistant object."""
        return (
            f"Chatbot Object:\n"
            f"ID: {self.chatbot.id}\n"
            f"Name: {self.chatbot.name}\n"
            f"Model: {self.chatbot.model}\n"
            f"Instructions: {self.chatbot.instructions[:100]}... (truncated)\n"  # Shorten for readability
            f"Tools: {self.chatbot.tools}\n"
            f"State: {self.chatbot.state}\n"
        )

    def log_chatbot_details(self):
        """Logs the chatbot object and conversation history."""
        # Log chatbot details
        logging.info(str(self))

        # Log conversation history
        logging.info("Conversation History:")
        for entry in self.conversation_history:
            logging.info(f"{entry['role']}: {entry['content']}")

    def _configure_services(self):
        """Configure and initialize external services (IBM, Marvin, etc.)."""
        # Initialize IBM services for speech-to-text and text-to-speech
        self.SPEECH_TO_TEXT = self._initialize_ibm_service(
            config("IAM_APIKEY_STT"), config("URL_STT")
        )
        self.TEXT_TO_SPEECH = self._initialize_ibm_service(
            config("IAM_APIKEY_TTS"), config("URL_TTS")
        )

        # Configure Marvin settings
        self._configure_marvin_settings()

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

    def _configure_marvin_settings(self):
        """Configure Marvin settings for the voice assistant."""
        # Setting up Marvin settings
        marvin.settings.openai.api_key = config("MARVIN_OPENAI_API_KEY")
        marvin.settings.openai.chat.completions.model = config(
            "MARVIN_CHAT_COMPLETIONS_MODEL"
        )

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

    def detect_sentiment(self, user_input: str) -> Sentiment:
        """Detect the sentiment of the user's input using Marvin."""
        return marvin.classify(user_input, Sentiment)

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

    async def start_session(self):
        """Handle the conversation with the user."""
        # Start the session by speaking a greeting
        self._speak("Hello! Chat with GPT and I will speak its responses!")
        while True:
            # Listen to the user's speech and transcribe it
            user_input = self._listen()
            logging.info(f"User Speech Text: {user_input} \n")

            # Check if user_speech_text is not None
            if user_input:
                user_input = user_input.lower()
                # Exit the loop if the user says "exit"
                if check_exit_command(user_input):
                    self._speak("Goodbye!")
                    break
                # Process the input through Marvin
                try:
                    # detect user sentiment
                    self.detect_sentiment(user_input)

                    # Generate a GPT response based on the user's input
                    gpt_response = await self.chatbot.say_async(user_input)
                    # Extract the text from the GPT response
                    gpt_response_text = gpt_response.messages[-1].content[0].text.value
                    logging.info(f"GPT Response Message: {gpt_response_text} \n")  #
                    # Speak the GPT response
                    self._speak(gpt_response_text)
                    # Update the conversation history with the user's input and the GPT response
                    self.conversation_history.append(
                        {"role": "user", "content": user_input}
                    )
                    self.conversation_history.append(
                        {"role": "gpt", "content": gpt_response_text}
                    )
                except Exception as e:
                    logging.error(f"Failed to process input through Marvin: {e}")
                    self._speak(
                        "Sorry, I encountered an error processing your request."
                    )
            else:
                # If user_speech_text is None, handle the case appropriately
                logging.info("No valid input received. Please try speaking again.")
                self._speak("I didn't catch that, could you please repeat?")

        # Log the chatbot details and conversation history at the end of the session
        self.log_chatbot_details()
