import json
from dialogue_cozmo import Dialogue
import sys
from recorder_cozmo import Recorder
from face_emotion_cozmo import face_emo_cozmo
import traceback


def main():
    convo_loop_count = 0
    start_dialogue = False
    keep_intent = True
    print("keep intent", keep_intent)
    keep_entity = True
    print("keep_entity", keep_entity)
    tone_hist = []
    intent_list = []
    entity_list = []
    my_convo = Dialogue('speech.wav')
    my_convo.get_cozmo_response("Hello friend, what did you do today?")


    while not start_dialogue:
        print("convo turn loop count", convo_loop_count)

        try:
            print("starting recording process")
            recorder = Recorder("speech.wav")
            print("Please say something nice into the microphone\n")
            recorder.save_to_file()
            print("Transcribing audio....\n")

            try:
                input_text = my_convo.transcribe_audio()
                if "goodbye" in input_text:
                    break

                if input_text:
                    start_dialogue = True

                # begin detecting intent
                try:
                    if keep_intent:
                        if len(intent_list) > 0:
                            pass
                        else:
                            intent_state = my_convo.get_watson_intent(input_text)
                            if intent_state:
                                intent_list.append(intent_state)
                            # initialize intent response generator object
                            int_res = my_convo.get_intent_response(intent_state)
                            try:
                                my_convo.get_cozmo_response((next(int_res)))
                            except StopIteration:
                                pass
                            start_dialogue = False
                            continue

                    # detect entity  and tone based on intent
                    try:
                        if len(entity_list) > 0:
                            pass
                        else:
                            entity_state = my_convo.get_watson_entity(input_text)
                            if entity_state:
                                entity_list.append(entity_state)

                        # initialize entity response generator object
                        if len(entity_list) == 1 and entity_state:
                            # initialize entity response generator object
                            ent_res = my_convo.get_entity_response(entity_state)

                        top_tone, top_tone_score = my_convo.get_watson_tone(input_text)
                        if top_tone_score:
                            tone_hist.append({
                                'tone_name': top_tone,
                                'score': top_tone_score
                            })

                        # initialize tone response generator object
                        if len(tone_hist) == 1:
                            # initialize tone response generator object
                            tone_res = my_convo.get_tone_response(entity_state, top_tone)

                        # start dialogue with Nao
                        while start_dialogue:
                            # dialogue flow including entities and tone based on first detected intent
                            # maintained throughout conversation turn or not maintained
                            if keep_entity:
                                if entity_list and not tone_hist:
                                    print("detected unemotional entity")
                                    print("maintained entity", entity_state)
                                    try:
                                        my_convo.get_cozmo_response(next(ent_res))
                                        # new test to transition to other entity
                                        if entity_list[0] == "Penny":
                                            keep_entity = False
                                        if entity_list[0] == "concert":
                                            keep_entity = False
                                        if entity_list[0] == "client":
                                            keep_entity = False
                                    except StopIteration:
                                        pass

                                if tone_hist:
                                    try:
                                        my_convo.get_cozmo_response((next(tone_res)), top_tone)
                                        if entity_list[0] == "meeting":
                                            keep_entity = False
                                        if "coworker" in entity_list and top_tone == "joy":
                                            keep_intent = False
                                        if "Beach House" in entity_list and top_tone == "joy":
                                            keep_intent = False
                                        if "John" in entity_list and top_tone == "joy":
                                            keep_intent = False
                                        if entity_list[0] == "client":
                                            keep_entity = False
                                        if "promotion" in entity_list and top_tone == "joy":
                                            keep_intent = False
                                    except StopIteration:
                                        pass

                                    # entity_list.append(entity_state)

                                if entity_state != None:
                                    entity_list.append(entity_state)

                            if not keep_entity:
                                print("clearing entity list")
                                entity_list = []
                                print("clearing tone list")
                                tone_hist = []

                            # start detecting for new intent, clear prior intent, entity and tone lists
                            if not keep_intent:
                                print("clearing intent list")
                                intent_list = []
                                print("clearing entity list")
                                entity_list = []
                                print("clearing tone list")
                                tone_hist = []
                                my_convo.get_cozmo_response("Hello, new friend, what did you do today?")

                            start_dialogue = False
                            keep_entity = True
                            keep_intent = True
                            # end conversation
                    except:
                        traceback.print_exc()
                        print("bad initial entity detection")
                        print("I wasn't sure of your entity, what was that again?")
                        pass

                except:
                    traceback.print_exc()
                    print("bad initial intent detection")
                    print("I wasn't sure of your intent, what was that again?")
                    pass

            except:
                traceback.print_exc()
                print("error in speech detection")
                my_convo.get_cozmo_response("Oh Watson, a little rusty today in your detection eh? Sorry, user, can you repeat that?")
                pass

        except KeyboardInterrupt:
            print("closing via keyboard interrupt")
            sys.exit(0)
        convo_loop_count += 1
        
if __name__ == '__main__':
    main()
