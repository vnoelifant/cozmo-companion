
import wave
from array import array
from struct import pack

import pyaudio


class Recorder:
    def __init__(self, path):
        self.PATH = path
        self.THRESHOLD = 500
        self.CHUNK_SIZE = 1024
        self.FORMAT = pyaudio.paInt16
        self.RATE = 44100
        self.silent_count = 0
        self.is_recording = True
        self.RECORD_SECONDS = 5

    def record(self):
        # create an instance of pyAudio
        p = pyaudio.PyAudio()
        # create a stream
        stream = p.open(format=self.FORMAT, channels=1, rate=self.RATE,
                        input=True, output=True,
                        frames_per_buffer=self.CHUNK_SIZE)

        #starting recording
        frames= []

        while self.is_recording:
            """
            This operation is in charge of controlling how many iterations loop for makes. In every iteration, 
            1024 bytes are recorded (data = stream.read(CHUNK)) Therefore, to record 5 seconds, we have to take 44,100 samples/second * 5 seconds 
            = 220,500 samples. Finally, if each iteration (chunk) takes 1024 samples, the for will have to loop 220,500/1024 times = 215 samples
            
            44100 Hz - or 44100 samples per second. So you are basicly reading out that many digitized values from your device. 
            So if you want to record 5 seconds, you will have to save 
            5⋅44100 samples. Because most audio systems work with chunks (also called frames or blocks), the program will now read chunks of data. 
            One chunk in your case is 1024 samples. So basicly the system reads 215.33 chunks out of the buffer until it has saved the 
            whole 5 seconds of audio data.
            """
            for i in range(0,int(self.RATE/self.CHUNK_SIZE*self.RECORD_SECONDS)):
                #print("recording time",int(self.RATE/self.CHUNK_SIZE*self.RECORD_SECONDS))
                data=stream.read(self.CHUNK_SIZE)
                data_chunk=array('h',data)
                vol=max(data_chunk)
                if(vol>=self.THRESHOLD):
                    print("something said")
                    frames.append(data)
                else:
                    print("nothing")
                # print("\n")
            #print(len(frames))
            self.is_recording = False
            #print("exiting loop: recording bool", self.is_recording)

        sample_width = p.get_sample_size(self.FORMAT)
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Recording Complete")
        return sample_width, frames
       
    def save_to_file(self):
        sample_width,frames = self.record()
        wf = wave.open(self.PATH, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))#append frames recorded to file
        wf.close()

