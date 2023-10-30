import os
import logging

import marvin
from datetime import datetime
from decouple import config
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1, TextToSpeechV1, ApiException
from marvin import AIApplication, ai_classifier
from marvin.tools import tool
from pydub import AudioSegment
from pydub.playback import play
from enum import Enum
from typing import Optional

from .constants import (
    AUDIO_FORMAT,
    CONTENT_TYPE,
    KEYWORDS,
    KEYWORDS_THRESHOLD,
    VOICE,
    WORD_ALTERNATIVE_THRESHOLDS,
)
from .recorder import Recorder


# Enum class for sentiment analysis
@ai_classifier
class Sentiment(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class VoiceAssistant:

    """
    A class to represent a voice assistant capable of listening to voice input, generating a GPT response,
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
                "negative sentiment. For every response to the user, it"
                "asks the user if its response helped."
            ),
            tools=[self.get_feedback_inquiry],
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
        marvin.settings.openai.api_key = config("OPENAI_API_KEY")
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
                f"Invalid service URL: {url}. Expected 'speech-to-text' or 'text-to-speech' in the URL."
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

        print("Starting recording process")
        # Initialize the recorder
        recorder = Recorder(user_speech_file)

        print("Please say something to the microphone\n")
        # Start recording
        recorder.record()

        print("Transcribing audio....\n")
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
                # Extract the transcribed text from the result
                user_speech_text = speech_result["results"][0]["alternatives"][0][
                    "transcript"
                ]

            return user_speech_text
        # Handle exceptions from the IBM service
        except ApiException as ex:
            logging.error(
                "Method failed with status code " + str(ex.code) + ": " + ex.message
            )

    # @tool
    def get_feedback_inquiry(self, user_text: str = "") -> Optional[str]:
        """
        Checks if the user's text contains specific feedback phrases and
        returns an appropriate inquiry string based on the content.

        Args:
            user_text (str): The text provided by the user.

        Returns:
            Optional[str]: The feedback inquiry string or None if no match is found.
        """
        # List of feedback phrases to check against
        feedback_phrases = ["joke", "motivational quote"]
        for phrase in feedback_phrases:
            # If the user's text contains any of the feedback phrases, return the inquiry string
            if phrase in user_text.lower():
                return " Did this help put a smile to your face?"
        # If no match is found, return None
        return None

    def construct_gpt_prompt(self, text):
        """Construct the GPT-3 prompt based on the user's sentiment."""
        # Detect the sentiment of the user's text
        detected_sentiment = Sentiment(text)
        # If the sentiment is negative, append a request for an empathetic response
        if detected_sentiment == Sentiment.NEGATIVE:
            print("Negative Sentiment Detected...")
            return text + " Requesting an empathetic response."
        return text

    def _generate_response(self, text):
        """Generate a GPT response to the user's text input."""
        try:
            # Construct the GPT prompt based on the input text
            gpt_prompt = self.construct_gpt_prompt(text)

            # Update the conversation history with the user's input
            self.conversation_history.append({"role": "user", "content": gpt_prompt})

            # Get the chatbot's response content
            gpt_response = self.chatbot(gpt_prompt).content

            # If there's a feedback inquiry, append it to the chatbot's response
            feedback_inquiry = self.get_feedback_inquiry(gpt_prompt)
            if feedback_inquiry and feedback_inquiry not in gpt_response:
                gpt_response += feedback_inquiry

            # Update the conversation history with the chatbot's response
            self.conversation_history.append({"role": "gpt", "content": gpt_response})

            # Print the conversation log (this can be removed if not needed in production)
            print("Conversation Log: ", self.conversation_history)

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
            print(f"User Speech Text: {user_speech_text} \n")
            # Exit the loop if the user says "exit"
            if "exit" in user_speech_text.strip().lower():
                self._speak("Goodbye!")
                break
            # Generate a GPT response for the user's input
            gpt_response_msg = self._generate_response(user_speech_text)
            print(f"GPT Response Message: {gpt_response_msg} \n")
            # Speak the GPT response
            self._speak(gpt_response_msg)
