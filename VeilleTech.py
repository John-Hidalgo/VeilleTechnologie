import pyaudio
import wave
import keyboard
import whisper
import os
from openai import OpenAI
import subprocess
from pydub import AudioSegment
from pydub.playback import play

import numpy as np

def is_silent(data, threshold=500):
    audio_data = np.frombuffer(data, dtype=np.int16)
    return np.abs(audio_data).mean() < threshold

def InvitesCommencer():
    print("Please press 'r' to begin recording")
    while True:
        if keyboard.is_pressed('r'):
            break
        
def InvitesArreter():
    print("Now recording ... please press 's' when you've finished")
    while True:
        if keyboard.is_pressed('s'):
            break
    
def Enregistrer(fichier, silence_limit=2):
    CHUNK = 512
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    SILENCE_THRESHOLD = 500
    SILENCE_CHUNKS = int(RATE / CHUNK * silence_limit)

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    print("Now recording ... press 's' to stop manually or stop talking to end recording.")

    silent_chunks = 0
    while True:
        if keyboard.is_pressed('s'):
            break
        data = stream.read(CHUNK)
        frames.append(data)
        
        if is_silent(data, SILENCE_THRESHOLD):
            silent_chunks += 1
        else:
            silent_chunks = 0
        
        if silent_chunks > SILENCE_CHUNKS:
            break

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(fichier, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def OuvrirFichier(fichier):
    with open(fichier,'r',encoding='utf-8') as infile:
        return infile.read()
    
client = OpenAI(base_url="https://dragon.rapidotron.com/v1/", api_key="lm-studio") 
fichierHistoire = "conversionHistoire.txt"

def ParlerAvecChat(texteUtilisateur, system_message, histoire,bot_name):
    messages = [{"role":"system","content":system_message}] + histoire + [{"role":"user","content":texteUtilisateur}]
    streamed_completion = client.chat.completions.create(
        model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_S.gguf",
        messages=messages,
        temperature=1,
        stream=True
    )
    reponse = ""
    ligneTemp = ""
    with open(fichierHistoire, "a") as journaliseFichier:
        for chunk in streamed_completion:
            changements = chunk.choices[0].delta.content
            if changements is not None:
                ligneTemp += changements
                if '\n' in ligneTemp:
                    lines = ligneTemp.split('\n')
                    for line in lines[:-1]:
                        print(line)
                        reponse += line + '\n'
                        journaliseFichier.write(f"{bot_name}: {line}\n")
                    ligneTemp = lines[-1]
        if ligneTemp:
            print(ligneTemp)
            reponse += ligneTemp
            journaliseFichier.write(f"{bot_name}: {ligneTemp}\n")
    with open("reponse.txt", "w") as fichierReponse:
        fichierReponse.write(reponse)
    return reponse

def VersTexte(fichierAudio):
    model = whisper.load_model("base") 
    result = model.transcribe(fichierAudio)
    return result["text"]

def VersVoix():
    
    with open("reponse.txt", "r") as fichierReponse:
        reponseContent = fichierReponse.read()
        
    reponseEcho = reponseContent.replace('\n', ' ')

    commande = f'echo "{reponseEcho}" | piper\\piper.exe -m piper\\fr_FR-tom-medium.onnx -f rec2.wav -q'
    process = subprocess.Popen(commande, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    process.wait()
    voix = AudioSegment.from_wav('rec2.wav')
    play(voix)
    os.remove('rec2.wav')
    os.remove('reponse.txt')

def Conversation():
    histoire = []
    prompteAuLLM = OuvrirFichier("prompt.txt")
    while True:
        fichierAudio = "rec.wav"
        InvitesCommencer()
        Enregistrer(fichierAudio)
        texteUtilisateur = VersTexte(fichierAudio)
        os.remove(fichierAudio)
        
        print("You:", texteUtilisateur)
        histoire.append({"role":"user","content":texteUtilisateur})
        print("LLM:")
        reponse = ParlerAvecChat(texteUtilisateur,prompteAuLLM,histoire,"Chatbot")
        VersVoix()
        histoire.append({"role":"assistant","content":reponse})
    
Conversation()