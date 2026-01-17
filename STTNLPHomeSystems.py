import os

import numpy as np

import pyaudio

import wave

import struct

from collections import deque

from numpy.linalg import norm

from sentence_transformers import SentenceTransformer

import sys

import string

import whisper

import paho.mqtt.client as mqtt

import concurrent.futures

import re



# --- CONFIGURATION ---

THRESHOLD = 5       # Audio volume threshold to trigger recording (Adjust based on mic sensitivity)

SILENCE_LIMIT = 2     # Seconds of silence to wait before stopping recording

PREV_AUDIO = 0.5      # Seconds of audio to keep before trigger (prevents cutting off first syllable)



training = {

    "light_on": [

        "turn on the light","switch the lamp on","make it brighter","it is too dark in here",

        "i can't see anything","i need more light","brighten the lights","light please",

        "let there be light","start the lamp","my room is too dark","the room feels gloomy"

    ],

    "light_off": [

        "turn off the light","switch off the lamp","please dim the lights","it's too bright",

        "i want darkness","make it darker","it's bedtime","the glare is hurting my eyes",

        "i want to sleep","reduce the brightness","kill the light","turn down the lamp brightness"

    ],

    "fan_on": [

        "turn on the fan","switch the fan on","start the air cooler","i am hot",

        "it's too warm in here","i'm sweating","the air is stuffy","please blow some air",

        "it's humid here","make it cooler","start the ventilation","it's suffocating",

        "it's not cold","i'm not freezing","don't turn off the fan","turn on fan"

    ],

    "fan_off": [

        "turn off the fan","switch the fan off","stop the air cooler","it's too cold in here",

        "i'm freezing","the air is too strong","i feel chilly","there's too much wind",

        "my papers are flying","reduce the fan speed","please stop the fan","the fan is making me cold",

        "it's not hot","i'm not sweating","don't turn on the fan"

    ]

}



print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")

model = SentenceTransformer("all-MiniLM-L6-v2")



#MQTT Configuration

BROKER   = "08d5c716cf9f46518abcda4d565e5141.s1.eu.hivemq.cloud"

PORT     = 8883

USERNAME = "p_user"

PASSWORD = "P_user123"

CAFILE   = "/home/ddd/Desktop/isrgrootx1.pem"

TOPIC_PUB = "iot/pi/command"



def mqtt_publish(payload: str):

    try:

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

        client.tls_set(CAFILE)

        client.username_pw_set(USERNAME, PASSWORD)

        client.connect(BROKER, PORT, keepalive=20)

        client.loop_start()

        info = client.publish(TOPIC_PUB, payload, qos=0, retain=False)

        info.wait_for_publish()

        client.loop_stop()

        client.disconnect()

        print(f"[MQTT] Sent '{payload}' to {TOPIC_PUB}")

    except Exception as e:

        print(f"[MQTT] Publish error: {e}")



# --- NEW AUDIO FUNCTIONS ---



def get_rms(data):

    """Calculate Root Mean Square (volume) of the audio chunk"""

    # Convert byte data to integers

    count = len(data) / 2

    format = "%dh" % (count)

    shorts = struct.unpack(format, data)

    sum_squares = 0.0

    for sample in shorts:

        n = sample * (1.0 / 32768.0) # Normalize

        sum_squares += n * n

    # Return rms scaled to roughly 0-1000 range for easier reading

    return np.sqrt(sum_squares / count) * 1000



def record_on_detect(output_filename="recorded_audio.wav"):

    """

    Listens continuously. 

    1. Buffers audio in a loop.

    2. Triggered when volume > THRESHOLD.

    3. Records until silence lasts > SILENCE_LIMIT.

    """

    FORMAT = pyaudio.paInt16

    CHANNELS = 1

    RATE = 16000

    CHUNK = 1024 # Increased chunk size for better RMS calculation

    

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)



    print("\n[LISTENING] Waiting for sound...")

    

    audio2send = []

    

    # Pre-audio buffer (to catch the start of the sentence)

    rel = int(RATE / CHUNK * PREV_AUDIO)

    slid_win = deque(maxlen=rel * 2)

    

    started = False

    silence_count = 0

    silence_threshold_chunks = int(SILENCE_LIMIT * (RATE / CHUNK))



    while True:

        try:

            data = stream.read(CHUNK, exception_on_overflow=False)

            slid_win.append(data)

            

            rms = get_rms(data)



            if not started:

                # Waiting for trigger

                if rms > THRESHOLD:

                    print("[DETECTED] Sound detected, recording...")

                    started = True

                    # Add the previous seconds of audio so we don't cut the start off

                    audio2send.extend(list(slid_win))

            else:

                # Currently Recording

                audio2send.append(data)

                

                if rms < THRESHOLD:

                    silence_count += 1

                else:

                    silence_count = 0 # Reset if we hear sound again



                # Stop if silence exceeds limit

                if silence_count > silence_threshold_chunks:

                    print("[STOP] Silence detected. Processing...")

                    break

        except KeyboardInterrupt:

            print("\nStopping...")

            break



    # Cleanup

    stream.stop_stream()

    stream.close()

    p.terminate()



    # Save to file

    if len(audio2send) > 0:

        wf = wave.open(output_filename, 'wb')

        wf.setnchannels(CHANNELS)

        wf.setsampwidth(p.get_sample_size(FORMAT))

        wf.setframerate(RATE)

        wf.writeframes(b''.join(audio2send))

        wf.close()

        return True # Audio was recorded

    return False



# --- EXISTING NLP/WHISPER LOGIC ---



def transcribe_with_whisper(audio_filename, model_size="tiny"):

    """

    Transcribes audio using a local Whisper model with a 15-second timeout.

    """

    if not os.path.exists(audio_filename):

        print(f"❌ Audio file not found: {audio_filename}")

        return ""

        

    # Internal function to run the actual transcription

    def _run_transcription():

        try:

            print(f"Loading Whisper model: {model_size}")

            # Note: Loading the model takes time. On a Pi 4, consider loading this 

            # globally once at startup if timeouts occur frequently.

            model = whisper.load_model(model_size)

            result = model.transcribe(audio_filename, language='en')

            return result['text']

        except Exception as e:

            print(f"An error occurred with Whisper: {e}")

            return None



    # Use a ThreadPoolExecutor to run transcription with a timeout

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    future = executor.submit(_run_transcription)

    

    try:

        # Wait max 15 seconds for the result

        text = future.result(timeout=20)

        

        if text:

            print(f"Whisper Recognition: {text}")

            return text

        else:

            return ""

            

    except concurrent.futures.TimeoutError:

        print("Didn't understand the sentence. Please try again.")

        return "" # Return empty string to stop processing

        

    except Exception as e:

        print(f"Processing error: {e}")

        return ""

    finally:

        # Clean up threads

        executor.shutdown(wait=False)



def cosine(a, b):

    return np.dot(a, b) / (norm(a) * norm(b) + 1e-10)



def save_np_object(obj, filename):

    np.save(filename, obj)



def load_np_object(filename):

    if not os.path.exists(filename):

        return None

    return np.load(filename, allow_pickle=True).item()



def build_centroids(model_):

    centroids = {}

    for label, sents in training.items():

        vecs = model_.encode(sents, normalize_embeddings=True)

        centroids[label] = np.mean(vecs, axis=0)

    return centroids



STOCK_FAN_ON_PHRASES = [

    "turn on the fan", "turn on the fans",

    "turn on fan", "turn on fans",

    "open the fan", "open the fans",

    "open fan", "open fans",

    "start the fan", "start the fans"]



NEG_TOKENS = {"not","dont","don't","never","no"}

CONF_THRESH = 0.50

MARGIN_THRESH = 0.00



def classify(text, model_, centroids):

    label = "uncertain"

    conf = 0.0

    sims = {}



    # --- STEP 1: Check Stock Phrases (Regex Override) ---

    # We use regex to allow words in between, e.g., "turn on THE FIRST fan"

    # Patterns to detect "FAN ON":

    # 1. "turn on" followed eventually by "fan"

    # 2. "open" followed eventually by "fan"

    # 3. "start" followed eventually by "fan"

    fan_on_pattern = r"(turn on|switch on|open|start).*(fan|cooler|ventilation)"

    

    # Check for match

    if re.search(fan_on_pattern, text, re.IGNORECASE):

        label = "fan_on"

        conf = 1.0  # Force high confidence

        sims = {k: 0.0 for k in centroids.keys()}

        sims["fan_on"] = 1.0 

        is_stock_command = True

    else:

        is_stock_command = False

    

    # --- STEP 2: Run NLP (if no stock phrase found) ---

    if not is_stock_command:

        v = model_.encode([text], normalize_embeddings=True)[0]

        sims = {label: cosine(v, cvec) for label, cvec in centroids.items()}

        label = max(sims, key=sims.get)

        conf = sims[label]



    # --- STEP 3: Handle Negatives ---

    # This prevents "Don't open the first fan" from triggering ON

    toks = set(text.lower().split())

    if NEG_TOKENS & toks:

        if "on" in label:  label = label.replace("on","off")

        elif "off" in label: label = label.replace("off","on")



    # --- STEP 4: Threshold Checks ---

    if not is_stock_command:

        vals = sorted(sims.values())

        margin = vals[-1] - vals[-2] if len(vals) >= 2 else 1.0

        

        # Using the updated 0.50 threshold

        if conf < 0.50 or margin < MARGIN_THRESH:

            label = "uncertain"



    return label, conf, sims



def detect_room_from_speech(text):

    text_lower = text.lower()

    first_room_keywords = ['first', 'room1', 'room one', 'one', 'first room', 'first light', 'first fan', 'primary', 'main']

    second_room_keywords = ['second', 'room2', 'room two', 'two', 'second room', 'second light', 'second fan', 'secondary', 'other']

    

    first_detected = any(k in text_lower for k in first_room_keywords)

    second_detected = any(k in text_lower for k in second_room_keywords)

    

    if first_detected and not second_detected: return 'esp32-1'

    elif second_detected and not first_detected: return 'esp32-2'

    return 'both'



def get_mqtt_command(intent_label, room_target):

    action_map = {"light_on": "LED ON", "light_off": "LED OFF", "fan_on": "FAN ON", "fan_off": "FAN OFF"}

    action = action_map.get(intent_label)

    if not action: return None

    

    if room_target == 'both': return [f"ESP32-1:{action}", f"ESP32-2:{action}"]

    elif room_target == 'esp32-1': return f"ESP32-1:{action}"

    elif room_target == 'esp32-2': return f"ESP32-2:{action}"

    return None



# --- MAIN EXECUTION ---



if __name__ == "__main__":

    C_FILE = "centroids_sentence.npy"

    centroids = load_np_object(C_FILE)

    if centroids is None:

        centroids = build_centroids(model)

        save_np_object(centroids, C_FILE)



    print("\n=== Smart Home Classifier (Auto-Voice Detect) ===")

    

    # Pre-load Whisper model to avoid reloading it every loop (Huge performance boost)

    print("Loading Whisper model into memory...")

    whisper_model_instance = whisper.load_model("tiny") 



    try:

        while True:

            # 1. Wait for sound and record

            recorded = record_on_detect()

            

            if not recorded:

                continue



            audio_file = "recorded_audio.wav"

            print("Processing audio...")



            # 2. Transcribe

            # Note: We use the pre-loaded global 'whisper_model_instance' now

            untokenised = transcribe_with_whisper(audio_file)

            if not untokenised.strip():

                print("No speech detected in audio.")

                continue



            untokenised_nopunct = untokenised.translate(str.maketrans('', '', string.punctuation))

            tokenised = ' '.join(untokenised_nopunct.split()).lower()

            

            print(f"Tokenised Output: {tokenised}")



            # 3. Detect Room & Intent

            room_target = detect_room_from_speech(tokenised)

            print(f"Room Target: {room_target.upper()}")



            label, conf, sims = classify(tokenised, model, centroids)



            if label == "uncertain":

                print("Intent uncertain.\n")

            else:

                print(f"→ Intent: {label.upper()} | Confidence: {conf:.3f}")

                

                # 4. Execute MQTT

                mqtt_commands = get_mqtt_command(label, room_target)

                if mqtt_commands:

                    if isinstance(mqtt_commands, list):

                        for cmd in mqtt_commands:

                            mqtt_publish(cmd)

                    else:

                        mqtt_publish(mqtt_commands)

                else:

                    print(f"[MQTT] No command mapped for: {label}")

            

            print("\nReady for next command...\n")



    except KeyboardInterrupt:

        print("\nExiting system gracefully.")


