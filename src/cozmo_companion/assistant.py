import asyncio
import json
import os
import sys
import time
import traceback

import anki_vector
import openai
from decouple import config
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1
from PIL import Image

from .constants import (
    CONTENT_TYPE,
    GPT_MODEL,
    KEYWORDS,
    KEYWORDS_THRESHOLD,
    MAX_TOKENS,
    TEMPERATURE,
    WORD_ALTERNATIVE_THRESHOLDS,
)
from .recorder import Recorder

# Initialize speech to text service. Source: https://cloud.ibm.com/apidocs/speech-to-text
authenticator = IAMAuthenticator(config("IAM_APIKEY_STT"))

speech_to_text = SpeechToTextV1(authenticator=authenticator)

speech_to_text.set_service_url(config("URL_STT"))

# Configure Open AI API KEY
openai.api_key = config("OPENAI_API_KEY")


class VoiceAssistant:

    """
    A class to respresent a voice assistant who can listen to voice input, provide a gpt response,
    and speak the gpt response
    """

    def __init__(self):
        self.conversation_history = [
            {"role": "system", "content": "You are a helpful, friendly assistant"}
        ]

    def listen(self, audio_filename):
        """
        Function to record audio and transcribe recorded speech
        """

        if not os.path.exists("wav_output"):
            os.makedirs("wav_output")

        curr_dir = os.getcwd()
        audio_file = os.path.join(curr_dir, "wav_output", f"{audio_filename}.wav")

        print("Starting recording process")

        recorder = Recorder()

        print("Please say something to the microphone\n")

        recorder.record(audio_file)

        print("Transcribing audio....\n")

        with open((audio_file), "rb") as audio_file_obj:
            speech_result = speech_to_text.recognize(
                audio=audio_file_obj,
                content_type=CONTENT_TYPE,
                word_alternatives_threshold=WORD_ALTERNATIVE_THRESHOLDS,
                keywords=KEYWORDS,
                keywords_threshold=KEYWORDS_THRESHOLD,
            ).get_result()

            speech_text = speech_result["results"][0]["alternatives"][0]["transcript"]

        return speech_text

    def get_gpt_completion(self, text):
        """
        Function to generate a GPT response to the user's text input
        """

        # Add the user's input to the Chat GPT history log
        self.conversation_history.append({"role": "user", "content": text})

        gpt_response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=self.conversation_history,
        )

        gpt_response_role = gpt_response["choices"][0]["message"]["role"]
        gpt_response_msg = gpt_response["choices"][0]["message"]["content"]

        self.conversation_history.append(
            {
                "role": gpt_response_role,
                "content": gpt_response_msg,
            }
        )

        print("Conversation Log: ", self.conversation_history)

        return gpt_response_msg

    def speak(self, text):
        """
        Function for Cozmo robot to convert text input to speech
        """

        # Cozmo speaks the text input
        with anki_vector.Robot() as robot:
            robot.behavior.say_text(text)
