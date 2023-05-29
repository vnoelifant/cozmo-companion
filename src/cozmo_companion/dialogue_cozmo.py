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
from .recorder_cozmo import Recorder

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
        self.content_type = "audio/wav"
        self.word_alternatives_threshold = 0.9
        self.keywords = ["hey", "hi", "watson", "friend", "meet"]
        self.keywords_threshold = 0.5
        self.gpt_model = "gpt-3.5-turbo"
        self.max_tokens = 1000
        self.temperature = 1.2
        self.conversation_history = [
            {"role": "system", "content": "You are a helpful, friendly assistant"}
        ]

    def listen(self, filename):
        """
        Function to record audio and transcribe recorded speech
        """
        curr_dir = os.getcwd()
        speech_out = os.path.join(curr_dir, "wav_output", filename + ".wav")

        print("starting recording process")
        recorder = Recorder(speech_out)

        print("Please say something nice into the microphone\n")
        recorder.save_to_file()
        print("Transcribing audio....\n")

        with open((speech_out), "rb") as audio_file:
            speech_result = speech_to_text.recognize(
                audio=audio_file,
                content_type=self.content_type,
                word_alternatives_threshold=self.word_alternatives_threshold,
                keywords=self.keywords,
                keywords_threshold=self.keywords_threshold,
            ).get_result()

            speech_text = speech_result["results"][0]["alternatives"][0]["transcript"]
            print("User Speech Text: " + speech_text + "\n")

        return speech_text

    def get_gpt_completion(self, text):
        """
        Function to generate a GPT response to the user's text input
        """

        # Add the user's input to the Chat GPT history log
        self.conversation_history.append({"role": "user", "content": text})

        gpt_response = openai.ChatCompletion.create(
            model=self.gpt_model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=self.conversation_history,
        )

        gpt_response_msg = gpt_response.choices[0].message.content

        self.conversation_history.append(
            {
                "role": gpt_response_msg.choices[0].message.role,
                "content": gpt_response_msg,
            }
        )
        print("GPT RESPONSE MESSAGE: ", gpt_response_msg)

        return gpt_response_msg

    def speak(self, gpt_response):
        """
        Function to get Cozmo to convert GPT text response to speech
        """

        # Cozmo speaks the text response from Watson
        with anki_vector.Robot() as robot:
            robot.behavior.say_text(gpt_response)
