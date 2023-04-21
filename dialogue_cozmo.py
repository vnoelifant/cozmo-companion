import json
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import traceback
import sys
import cozmo
import time
import asyncio
from PIL import Image
from decouple import config

# NOTE: Commented code is from prior iteration that utilized Watson Tone Analyzer
# and Watson Assistant services
# TODO: Remove comments for unused Watson services

"""

TOP_TONE_SCORE_THRESHOLD = 0.75


tone_analyzer = ToneAnalyzerV3(
    VERSION=config('VERSION_TONE'),
    IAM_APIKEY=config('IAM_APIKEY_TONE'),
    URL=config('URL_TONE')
)

# initialize the Watson Assistant
assistant = AssistantV1(
    VERSION=config('VERSION_ASST'),
    IAM_APIKEY=config('IAM_APIKEY_ASST'),
    URL=config('URL_ASST')
)



WORKSPACE_ID = config('WORKSPACE_ID')



text_to_speech = TextToSpeechV1(
    IAM_APIKEY=config('IAM_APIKEY_TTS'),
    URL=config('URL_TTS')
)

print('Workspace id {0}'.format(WORKSPACE_ID))

emotions = ['sadness', 'joy', 'anger', 'fear']

"""


class Dialogue:
    def __init__(self, path_to_audio_file):
        self.path_to_audio_file = path_to_audio_file

    def transcribe_audio(self):
        """
        Function to transcribe recorded speech
        """
        
        # initialize speech to text service. Source: https://cloud.ibm.com/apidocs/speech-to-text
        authenticator = IAMAuthenticator(config('IAM_APIKEY_STT'))
        
        speech_to_text = SpeechToTextV1(authenticator=authenticator)
        
        speech_to_text.set_service_url(config('URL_STT'))

        with open((self.path_to_audio_file), 'rb') as audio_file:
            speech_result = speech_to_text.recognize(
                audio=audio_file,
                content_type='audio/wav',
                word_alternatives_threshold=0.9,
                keywords=['hey', 'hi', 'watson', 'friend', 'meet'],
                keywords_threshold=0.5
            ).get_result()

            speech_text = speech_result['results'][0]['alternatives'][0]['transcript']
            print("User Speech Text: " + speech_text + "\n")

        # Below commented code was used for Watson Assistant Dialogue Service
        """
            input_text = {
                'workspace_id': WORKSPACE_ID,
                'input': {
                    'text': speech_text
                }
            }

        return input_text
        """

        return speech_text
    
    # Below commented code was used for Watson Assistant Dialogue Service
    
    """

    def get_watson_intent(self, input_text):

        intent_response = assistant.message(WORKSPACE_ID=WORKSPACE_ID,
                                            input=input_text['input'],
                                            ).get_result()

        if intent_response['intents'][0]['confidence'] > 0.7:
            intent_state = intent_response['intents'][0]['intent']
            print("response with detected intent")
            print(json.dumps(intent_response, indent=2))
            return intent_state

        return None

    def get_watson_entity(self, input_text):

        # print("intent state for entity", intent_state)
        entity_response = assistant.message(WORKSPACE_ID=WORKSPACE_ID,
                                            input=input_text['input'],
                                            ).get_result()

        if entity_response['entities']:
            if entity_response['entities'][0]['confidence'] > 0.5:
                entity_state = entity_response['entities'][0]['value']
                print("response with detected entity")
                print(json.dumps(entity_response, indent=2))
                return entity_state

        return None

    # get the top emotion
    def get_watson_tone(self, input_text):

        # initialize emotion data
        max_score = 0.0
        top_tone = None
        top_tone_score = None

        tone_analysis = tone_analyzer.tone(tone_input=input_text['input'],
                                           content_type='application/json').get_result()
        # print "tone response",json.dumps(tone_analysis, indent=2)
        print(tone_analysis)
        # create list of tones
        tone_list = tone_analysis['document_tone']['tones']

        # find the emotion with the highest tone score
        for tone_dict in tone_list:
            print("tone dict", tone_dict)
            if tone_dict['tone_id'] == "sadness" or tone_dict['tone_id'] == "joy" or \
                    tone_dict['tone_id'] == "anger" or tone_dict['tone_id'] == "fear":
                print("tone id", tone_dict['tone_id'])
                print("max score", max_score, top_tone, top_tone_score)
                if tone_dict['score'] > max_score:
                    max_score = tone_dict['score']
                    # top_tone = tone['tone_name'].lower()
                    top_tone = tone_dict['tone_id']
                    top_tone_score = tone_dict['score']
                    print("top emo", top_tone, top_tone_score)

                # set a neutral emotion if under tone score threshold
                if max_score <= TOP_TONE_SCORE_THRESHOLD:
                    print("tone score under threshold")
                    top_tone = 'neutral'
                    top_tone_score = None
                    print("top emotion, top score: ", top_tone, top_tone_score)

        #print("chosen top emo", top_tone, top_tone_score)
        return top_tone, top_tone_score

    def get_intent_response(self, intent_state):

        intent_res_list = self.build_intent_res(intent_state)

        try:
            for response_dict in intent_res_list:
                for response in response_dict[intent_state]:
                    yield response

        except KeyError:
            return

    def get_entity_response(self, entity_state):

        entity_res_list = self.build_entity_res(entity_state)

        try:
            for response_dict in entity_res_list:
                for response in response_dict[entity_state]:
                    yield response

        except KeyError:
            return

    def get_tone_response(self, entity_state, top_tone):

        tone_res_list = self.build_tone_res(entity_state, top_tone)

        try:
            for emo, response_dict in zip(emotions, tone_res_list):
                if emo == top_tone:
                    for response in response_dict[entity_state]:
                        yield response

        except KeyError:
            return

    def build_intent_res(self, intent_state):

        return [{'work': [f"Making that money eh? What did you do at your" + " " + intent_state + "?"],
                 'reading': ["What book did you read?"], 'friends': ["Which friend did you visit?"]}]

    def build_entity_res(self, entity_state):

        return [{'meeting': [f"Oh, how was the {entity_state}?",
                             f"Oh no, What happened at the {entity_state}?",
                             f"Oh no, What happened at the {entity_state}?"],
                 'coworker': [f"Oh, what bothered you about your {entity_state}?",
                              f"Ah. Does your {entity_state} at least try to come up with a middle ground?",
                              "Oh I am sorry to hear. Maybe if you speak to someone higher up they can help sort things out for you."],
                 'client': ["Wow! Congratulations! What does this mean?"],
                 'promotion': [
                     "That's so wonderful, congratulations again! I knew you were capable of such an amazing thing!"],
                 'Minor': [f"Oh, wow! What were your thoughts about the {entity_state}?",
                           f"I am so glad to hear you enjoyed the {entity_state}!",
                           "Anytime, Goodbye now!!"],
                 'Twilight': [f"Oh, I've heard of {entity_state} but forgot what it was about!",
                              "Haha, really, why?!",
                              f"Oh dear! Okay, I will be sure not to pick up {entity_state}.To lighten things up for you, let me recommend reading The Minor, by Sotseki."],
                 'concert': [f"Oh who did you see at the" + " " + entity_state + "?"],
                 'Beach House': [f"Cool! I love {entity_state}. How did they perform?"],
                 'John': [f"Okay, what did you do with {entity_state}?",
                          "Oh okay. Is eveyrthing alright with you though?",
                          f"I'm sorry to hear that. Well let me know if you ever want to talk about {entity_state}.You know I am here for you!",
                          "Anytime!"],
                 'Penny': [f"Okay, what did you do with {entity_state}?"]}]

    def build_tone_res(self, entity_state, top_tone):

        # if top_tone_score != None:
        # sadness, joy, anger, fear response lists for all entities
        return [{'meeting': [f"Oh no, you sound {top_tone}.What happened at the {entity_state}?",
            f"Oh sorry I missed that, you sound {top_tone}.What happened?"],
                 'coworker': [f"Oh no, you sound sad,what bothered you about your {entity_state}?",
                              f"Oh no, you sound sad again,what bothered you about your {entity_state}?"],
                 'Twilight': [
                     "Oh dear! Okay, I will be sure not to pick up that book..To lighten things up for you, let me recommend reading The Minor, by Sotseki"],
                 'John': ["Oh okay. Is eveyrthing alright with you though?",
                          "I'm sorry to hear that. Well let me know if you ever want to talk about your friend..You know I am here for you!",
                          "Anytime! I love you!"]},

                {'coworker': [f"Good to hear you are experiencing {top_tone}. I'm always here for you.",
                              "Good to hear you sound happy again"],
                 'client': ["Wow! Congratulations! What does this mean?"],
                 'promotion': [
                     "That's so wonderful, congratulations again! I knew you were capable of such an amazing thing!",
                     "Anytime!"],
                 'Minor': ["I am so glad to hear you enjoyed the book!",
                           "Anytime! Goodbye now dear Ronnie!"],
                 'Twilight': ["Anytime, Goodbye Now dear Ronnie!"],
                 'Beach House': [
                     f"Wow that's awesome! Any band that makes makes Penny feel {top_tone} is worth watching in my book!"],
                 'John': ["Anytime, I love you!"]},

                {'meeting': [
                    f"Oh you sound a bit angry,I'm here to help you. What happened at the {entity_state}?",
                    f"Oh, You sound angry again about the {entity_state}? What happened there?"],
                 'coworker': [f"Oh, you are angry, let me help. What bothered you about your {entity_state}?",
                              f"Oh, You sound angry again about your{entity_state}. What happened with you guys?"],
                 'Twilight': [
                     f"Oh dear! Okay, I will be sure not to pick up {entity_state}.To lighten things up for you, let me recommend reading The Minor, by Sotseki."],
                 'John': ["Oh okay. Is everything alright with you though?",
                          f"I'm sorry to hear that. Well let me know if you ever want to talk about {entity_state}. You know I am here for you!"]},

                {'meeting': [f"Oh no, I am sorry you sound scared, what happened at the {entity_state}?",
                             f"Oh, You sound scared again about the {entity_state}?"],
                 'coworker': [f"Oh don't be scared, what bothered you about {entity_state}?",
                              f"Oh, You sound scared again about {entity_state}?"],
                 'Twilight': ["Haha, really, why?!",
                              f"Oh dear! Okay, I will be sure not to pick up {entity_state}.To lighten things up for you, let me recommend reading The Miner, by Sotseki."],
                 'John': ["Oh okay. Is eveyrthing alright with you though?",
                          f"I'm sorry to hear that. Well let me know if you ever want to talk about" + " " + entity_state + ".You know I am here for you!"]}]
    """
    
    def get_cozmo_response(self, response, top_tone = ""):
        
        """
        Function to get Cozmo's response 
        top_tone parameter was used for integration of Watson Tone Analyzer, which is depracated
        """
    
        # Cozmo speaks the text response from Watson
        def cozmo_text_response(robot: cozmo.robot.Robot):
            """
            This function allows Cozmo to speak back the response to user
            """
            robot.say_text(response).wait_for_completed()

        
        def cozmo_face_response(robot: cozmo.robot.Robot):
            """
            # This function is used for Cozmo to detect facial emotion and make responses based on emotion
            """
            print("Running cozmo face response")
            try:

                robot.move_lift(-3)
                robot.set_head_angle(cozmo.robot.MAX_HEAD_ANGLE).wait_for_completed()
                robot.enable_facial_expression_estimation(True)
                face = None
                face = robot.world.wait_for_observed_face(timeout=30)
                if face and face.is_visible:
                    robot.set_all_backpack_lights(cozmo.lights.blue_light)
                else:
                    robot.set_backpack_lights_off()
                print(face.expression)
                print(face.name)
                print(face.face_id)
                print(face.expression_score)
                # Cozmo responds based on happy facial expression (transitioned from negative tone)
                # and a smiley face is displayed on his OLED screen
                face_response = f"I see your face with an {face.expression} expression. But I think you look happy!"
                robot.say_text(face_response).wait_for_completed()
                if face.expression == 'happy':
                    robot.say_text(f"Yay I am so glad you have a {face.expression} face!").wait_for_completed()
                # time.sleep(.1)
                image = Image.open("cozmo_smiley_2.jpg")
                image = image.resize(cozmo.oled_face.dimensions(), Image.NEAREST)
                image = cozmo.oled_face.convert_image_to_screen_data(image)
                seconds = 10

                for i in range(seconds):
                    robot.display_oled_face_image(image, 1000.0)
                    time.sleep(1.0)

            except asyncio.TimeoutError:
                print("did not find a face")
                pass
        
        
        def cozmo_program(robot: cozmo.robot.Robot):
            """
             # robot is an instance of class cozmo.robot.Robot, which is the interface 
             to a Cozmo robot. # http://cozmosdk.anki.com/docs/generated/cozmo.robot.html#cozmo.robot.Robot
            """
            
            cozmo_text_response(robot)
            
            # NOTE: Commented Code block below was used for integration of Watson Tone Analyzer. 
            # This service appears to have been depracated a/o 4/17/23. 
            """
            print("check the tone for face detection: ", top_tone)
            if top_tone and top_tone == "joy":
                print("detected joyful tone")
                cozmo_face_response(robot)
            else:
                pass
            """
        # Connect to Cozmo and run the provided program/function given as the parameter
        # http://cozmosdk.anki.com/docs/generated/cozmo.html
        cozmo.run_program(cozmo_program)
