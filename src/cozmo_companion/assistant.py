import asyncio
import json
import os
import sys
import time
import traceback

import marvin
from decouple import config
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1, TextToSpeechV1
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

# Initialize speech to text service.
authenticator = IAMAuthenticator(config("IAM_APIKEY_STT"))

speech_to_text = SpeechToTextV1(authenticator=authenticator)

speech_to_text.set_service_url(config("URL_STT"))

# Initialize text to speech service.
authenticator = IAMAuthenticator(config("IAM_APIKEY_TTS"))

text_to_speech = TextToSpeechV1(authenticator=authenticator)

text_to_speech.set_service_url(config("URL_TTS"))

# Configure Open AI API KEY
marvin.settings.openai.api_key = config("OPENAI_API_KEY")

# Confuigure Marvin LLM
marvin.settings.llm_model = config("MARVIN_LLM_MODEL")


class VoiceAssistant:

    """
    A class to respresent a voice assistant who can listen to voice input, provide a gpt response,
    and speak the gpt response
    """

    def __init__(self):
        self.chatbot = AIApplication(description=("A friendly, supportive chatbot."))

        self.conversation_history = []

    @staticmethod
    def create_wav_file(filename):
        if not os.path.exists("wav_output"):
            os.makedirs("wav_output")

        curr_dir = os.getcwd()
        speech_file = os.path.join(curr_dir, "wav_output", f"{filename}.wav")

        return speech_file

    def listen(self, filename):
        """
        Function to record audio and transcribe recorded speech
        """

        user_speech_file = VoiceAssistant.create_wav_file(filename)

        print("Starting recording process")

        recorder = Recorder(user_speech_file)

        print("Please say something to the microphone\n")

        recorder.record()

        print("Transcribing audio....\n")

        with open((user_speech_file), "rb") as audio:
            speech_result = speech_to_text.recognize(
                audio=audio,
                content_type=CONTENT_TYPE,
                word_alternatives_threshold=WORD_ALTERNATIVE_THRESHOLDS,
                keywords=KEYWORDS,
                keywords_threshold=KEYWORDS_THRESHOLD,
            ).get_result()

            user_speech_text = speech_result["results"][0]["alternatives"][0]["transcript"]

        return user_speech_text

    def get_gpt_completion(self, text):
        """
        Function to generate a GPT response to the user's text input
        """

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

    def speak(self, text):
        """
        Function for Watson to convert text input to speech
        """

        bot_speech_file = VoiceAssistant.create_wav_file("bot_speech")

        with open(bot_speech_file, "wb") as audio_out:
            audio_out.write(
                text_to_speech.synthesize(
                    text,
                    voice=VOICE,
                    accept=AUDIO_FORMAT,
                )
                .get_result()
                .content
            )

        bot_speech_response = AudioSegment.from_wav(bot_speech_file)
        play(bot_speech_response)
