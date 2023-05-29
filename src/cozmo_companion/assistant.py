import asyncio
import json
import sys
import time
import traceback
import os

import cozmo
import anki_vector
import openai
from decouple import config
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1
from PIL import Image
from .recorder import Recorder
from .constants import (
    CONTENT_TYPE,
    WORD_ALTERNATIVE_THRESHOLDS,
    KEYWORDS,
    KEYWORDS_THRESHOLD,
    GPT_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
)


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
        curr_dir = os.getcwd()
        speech_out = os.path.join(curr_dir, "wav_output", audio_filename + ".wav")

        print("starting recording process")
        recorder = Recorder(speech_out)

        print("Please say something nice into the microphone\n")
        recorder.save_to_file()
        print("Transcribing audio....\n")

        # Initialize speech to text service. Source: https://cloud.ibm.com/apidocs/speech-to-text
        authenticator = IAMAuthenticator(config("IAM_APIKEY_STT"))

        speech_to_text = SpeechToTextV1(authenticator=authenticator)

        speech_to_text.set_service_url(config("URL_STT"))

        with open((speech_out), "rb") as audio_file:
            speech_result = speech_to_text.recognize(
                audio=audio_file,
                content_type=CONTENT_TYPE,
                word_alternatives_threshold=WORD_ALTERNATIVE_THRESHOLDS,
                keywords=KEYWORDS,
                keywords_threshold=KEYWORDS_THRESHOLD,
            ).get_result()

            speech_text = speech_result["results"][0]["alternatives"][0]["transcript"]
            print("User Speech Text: " + speech_text + "\n")

        return speech_text

    def get_gpt_completion(self, text):
        """
        Function to generate a GPT response to the user's text input
        """

        # Configure Open AI API KEY
        openai.api_key = config("OPENAI_API_KEY")

        # Add the user's input to the Chat GPT history log
        self.conversation_history.append({"role": "user", "content": text})

        gpt_response = openai.ChatCompletion.create(
            model=self.GPT_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=self.conversation_history,
        )

        gpt_response_msg = gpt_response.choices[0].message.content

        self.conversation_history.append(
            {
                "role": gpt_response_msg.choices[0].message.role,
                "content": gpt_response_msg,
            }
        )
        print("GPT Response Message: ", gpt_response_msg)

        return gpt_response_msg

    def speak(self, text):
        """
        Function for Cozmo robot to convert text input to speech
        """

        # Cozmo speaks the text input
        with anki_vector.Robot() as robot:
            robot.behavior.say_text(text)
