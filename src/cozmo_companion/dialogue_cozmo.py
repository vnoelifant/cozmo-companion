import asyncio
import json
import sys
import time
import traceback

import cozmo
import anki_vector
from decouple import config
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1
from PIL import Image


class Dialogue:
    def __init__(self, path_to_audio_file):
        self.path_to_audio_file = path_to_audio_file

    def transcribe_audio(self):
        """
        Function to transcribe recorded speech
        """

        # initialize speech to text service. Source: https://cloud.ibm.com/apidocs/speech-to-text
        authenticator = IAMAuthenticator(config("IAM_APIKEY_STT"))

        speech_to_text = SpeechToTextV1(authenticator=authenticator)

        speech_to_text.set_service_url(config("URL_STT"))

        with open((self.path_to_audio_file), "rb") as audio_file:
            speech_result = speech_to_text.recognize(
                audio=audio_file,
                content_type="audio/wav",
                word_alternatives_threshold=0.9,
                keywords=["hey", "hi", "watson", "friend", "meet"],
                keywords_threshold=0.5,
            ).get_result()

            speech_text = speech_result["results"][0]["alternatives"][0]["transcript"]
            print("User Speech Text: " + speech_text + "\n")

        return speech_text

    def get_cozmo_response(self, response):
        """
        Function to get Cozmo's response
        top_tone parameter was used for integration of Watson Tone Analyzer, which is depracated
        """

        # Cozmo speaks the text response from Watson
        with anki_vector.Robot() as robot:
            robot.behavior.say_text(response)
