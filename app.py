import streamlit as st
import pickle
import numpy as np
import logging
import mediapipe as mp
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import threading
import time
from twilio.rest import Client
import os
from gtts import gTTS
import pygame
import tempfile

# Setup logger
logger = logging.getLogger(__name__)

# Initialize Pygame mixer
pygame.mixer.init()

@st.cache_data
def get_ice_servers():
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning("Twilio credentials are not set. Fallback to a free STUN server from Google.")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)
    token = client.tokens.create()
    return token.ice_servers

@st.cache_data(ttl=24*3600)
def load_model():
    return pickle.load(open('model.p', 'rb'))

model_dict = load_model()
model = model_dict['model']

# Setup Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# RTC Configuration
RTC_CONFIGURATION = RTCConfiguration({"iceServers": get_ice_servers()})

class SignDetection(VideoTransformerBase):
    def __init__(self) -> None:
        self.frame_lock = threading.Lock()
        self.text = None
        self.previous_prediction = None
        self.audio_playing = False

    def transform(self, frame):
        frame_input = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
        data_aux = []

        results = hands.process(frame_rgb)
        with self.frame_lock:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_input,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x)
                        data_aux.append(landmark.y)

            if data_aux:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_value = int(prediction[0])
                if predicted_value in labels_dict:
                    predicted_character = labels_dict[predicted_value]

                    # Check if the prediction has changed
                    if predicted_character != self.previous_prediction:
                        self.previous_prediction = predicted_character

                        # Generate and play audio for the predicted sign
                        text = predicted_character
                        tts = gTTS(text=text, lang='en')

                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                            temp_filename = fp.name

                        tts.save(temp_filename)
                        pygame.mixer.music.load(temp_filename)
                        pygame.mixer.music.play()
                        self.audio_playing = True

                    # Display the recognized character on the frame
                    self.text = predicted_character
                    cv2.putText(frame_input, f'Sign: {self.text}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (255, 0, 0), 3, cv2.LINE_AA)
                else:
                    self.text = "Unknown prediction"
            else:
                self.text = "No hand landmarks detected in the current frame."

        return frame_input

def main():
    st.title("Sign Language Translation App")
    activities = ["Home", "Webcam Sign Detection"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "Home":
        st.write("This is a sign language translation app.")
    elif choice == "Webcam Sign Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use the webcam and detect your sign")

        webrtc_ctx = webrtc_streamer(
            key="sign-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=SignDetection,
            media_stream_constraints={"video": True, "audio": False}
        )

        if webrtc_ctx.video_processor:
            while True:
                if webrtc_ctx.state.playing:
                    with webrtc_ctx.video_processor.frame_lock:
                        text = webrtc_ctx.video_processor.text
                    st.markdown(f"**Detected Sign:** {text}")
                    time.sleep(0.1)

if __name__ == "__main__":
    main()