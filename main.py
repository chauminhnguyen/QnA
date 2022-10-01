import pyaudio
import wave
import numpy as np
import audioop
from utils import *

p = pyaudio.PyAudio()

# Find out the index of Azure Kinect Microphone Array
azure_kinect_device_name = "Azure Kinect Microphone Array"
index = -1
for i in range(p.get_device_count()):
    print(p.get_device_info_by_index(i))
    if azure_kinect_device_name in p.get_device_info_by_index(i)["name"]:
        index = i
        break
if index == -1:
    print("Could not find Azure Kinect Microphone Array. Make sure it is properly connected.")
    exit()

# Open the stream for reading audio
input_format = pyaudio.paInt32
input_sample_width = 4
input_channels = 7
input_sample_rate = 48000

stream = p.open(format=input_format, channels=input_channels, rate=input_sample_rate, input=True, input_device_index=1)

# Read frames from microphone and write to wav file
with wave.open("output.wav", "wb") as outfile:
    outfile.setnchannels(1) # We want to write only first channel from each frame
    outfile.setsampwidth(input_sample_width)
    outfile.setframerate(input_sample_rate)

    time_to_read_in_seconds = 5
    frames_to_read = time_to_read_in_seconds * input_sample_rate
    total_frames_read = 0
    total_stop_time = 3 * input_sample_rate
    stop_time = 0

    while True:
        available_frames = stream.get_read_available()
        read_frames = stream.read(available_frames)
        rms = audioop.rms(read_frames, 2)
        if rms < 50:
            stop_time += available_frames
        else:
            stop_time = 0
        if stop_time > total_stop_time:
            break
        first_channel_data = np.fromstring(read_frames, dtype=np.int32)[0::7].tobytes()
        outfile.writeframesraw(first_channel_data)
        total_frames_read += available_frames

stream.stop_stream()
stream.close()

p.terminate()

# Speech recognition the wav file

import speech_recognition as sr

r = sr.Recognizer()
with sr.AudioFile("output.wav") as source:
    audio = r.record(source)

try:
    model_name = "deepset/roberta-large-squad2"
    s = r.recognize_google(audio)
    limit_predict, limit_real_predict, limit_scores = test2(s, model_name, 'BitCoin')
    print("Text: "+s)
    print("Predict: "+ str(limit_predict))
except Exception as e:
    print("Exception: "+str(e))

# "What's best to describe New York?"
