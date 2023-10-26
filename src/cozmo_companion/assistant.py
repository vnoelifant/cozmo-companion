import os
import logging

import marvin
from datetime import datetime
from decouple import config
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1, TextToSpeechV1, ApiException
from marvin import AIApplication
from pydub import AudioSegment
from pydub.playback import play

from .constants import (
    AUDIO_FORMAT,
    CONTENT_TYPE,
    KEYWORDS,
    KEYWORDS_THRESHOLD,
    VOICE,
    WORD_ALTERNATIVE_THRESHOLDS,
)
from .recorder import Recorder


class VoiceAssistant:

    """
    A class to represent a voice assistant capable of listening to voice input, generating a GPT response,
    and speaking the GPT response.
    """

    def __init__(self):
        """Initialize the VoiceAssistant and its services."""
        self._configure_services()
        self.chatbot = AIApplication(description=("A friendly, supportive chatbot."
                                                  "If it detects negative sentimentm it provides an"
                                                  "empathetic response to the user. Then, it asks the user "
                                                  "how it can make the user feel better. When the user responds," 
                                                  "it provides a response based on the user's preference. "))

        self.conversation_history = []

    def _configure_services(self):
        """Configure and initialize external services (IBM, Marvin, etc.)."""
        self.SPEECH_TO_TEXT = self._initialize_ibm_service(
            config("IAM_APIKEY_STT"), config("URL_STT")
        )
        self.TEXT_TO_SPEECH = self._initialize_ibm_service(
            config("IAM_APIKEY_TTS"), config("URL_TTS")
        )
        marvin.settings.openai.api_key = config("OPENAI_API_KEY")
        marvin.settings.llm_model = config("MARVIN_LLM_MODEL")

    def _initialize_ibm_service(self, api_key, url):
        """
        Helper method to initialize IBM services.
        :param api_key: The API key for the service.
        :param url: The URL for the service.
        :return: Initialized IBM service.
        """
        authenticator = IAMAuthenticator(api_key)
        service = None
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
        if not os.path.exists("wav_output"):
            os.makedirs("wav_output")

        # Generate filename based on current timestamp and provided prefix
        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{curr_time}.wav"
        speech_file = os.path.join(os.getcwd(), "wav_output", filename)

        return speech_file

    def _listen(self):
        """Record audio and transcribe the recorded speech."""

        user_speech_file = VoiceAssistant._create_wav_file(prefix="user")

        print("Starting recording process")

        recorder = Recorder(user_speech_file)

        print("Please say something to the microphone\n")

        recorder.record()

        print("Transcribing audio....\n")

        try:
            with open((user_speech_file), "rb") as audio:
                speech_result = self.SPEECH_TO_TEXT.recognize(
                    audio=audio,
                    content_type=CONTENT_TYPE,
                    word_alternatives_threshold=WORD_ALTERNATIVE_THRESHOLDS,
                    keywords=KEYWORDS,
                    keywords_threshold=KEYWORDS_THRESHOLD,
                ).get_result()

                user_speech_text = speech_result["results"][0]["alternatives"][0][
                    "transcript"
                ]

            return user_speech_text
        except ApiException as ex:
            logging.error(
                "Method failed with status code " + str(ex.code) + ": " + ex.message
            )

    def _generate_response(self, text):
        """Generate a GPT response to the user's text input."""

        try:
            # Add the user's input to the Chat GPT history log
            self.conversation_history.append({"role": "user", "content": text})

            gpt_response = self.chatbot(text)

            gpt_response_msg = gpt_response.content

            self.conversation_history.append(
                {
                    "role": "gpt",
                    "content": gpt_response_msg,
                }
            )

            print("Conversation Log: ", self.conversation_history)

            return gpt_response_msg
        except Exception as e:
            logging.error(f"Error getting GPT completion: {e}", exc_info=True)
            return "I'm sorry, I couldn't process that."

    def _speak(self, text):
        """Convert text input to speech."""
        
        bot_speech_file = VoiceAssistant._create_wav_file(prefix="bot")
        
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

            bot_speech_response = AudioSegment.from_wav(bot_speech_file)
            play(bot_speech_response)
        except ApiException as ex:
            logging.error(
                "Method failed with status code " + str(ex.code) + ": " + ex.message
            )

    def start_session(self):
        """Handle the conversation with the user."""
        self._speak("Hello! Chat with GPT and I will speak its responses!")
        while True:
            user_speech_text = self._listen()
            print(f"User Speech Text: {user_speech_text} \n")
            if "exit" in user_speech_text.strip().lower():
                self._speak("Goodbye!")
                break
            gpt_response_msg = self._generate_response(user_speech_text)
            print(f"GPT Response Message: {gpt_response_msg} \n")
            self._speak(gpt_response_msg)
