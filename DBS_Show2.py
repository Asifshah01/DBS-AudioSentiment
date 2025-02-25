import streamlit as st
import io
import numpy as np
from pydub import AudioSegment
import noisereduce as nr
import whisper
import librosa
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import time

PASSWORD = "DBSSOLUTIONS"

# Create a password input field
password = st.text_input("Enter the password to access the app:", type="password")

# Check if the entered password is correct
if password != PASSWORD:
    st.error("Incorrect password. Please try again.")
    st.stop()  # Stop further execution if the password is wrong



st.title("DBS Audio Processing for Quality Assurance - Test Version")
st.subheader("This is a beta version --- Developed by @Alpha Shah")
st.subheader("Stay Tuned for more updates!!!")
# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file, format=uploaded_file.type)
    
    # Convert uploaded file to a readable format
    audio_bytes = uploaded_file.read()
    audio_buffer = io.BytesIO(audio_bytes)
    
    # Load audio with pydub
    audio = AudioSegment.from_file(audio_buffer, format=uploaded_file.type.split('/')[-1])
    duration = len(audio) / 1000  # Convert milliseconds to seconds
    
    st.write(f"**File Format:** {uploaded_file.type}")
    st.write(f"**Duration:** {duration:.2f} seconds")
    
    # Convert audio to NumPy array
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate
    
    # Apply noise reduction
    reduced_noise = nr.reduce_noise(y=samples, sr=sample_rate)
    
    st.success("Noise reduction applied successfully!")
    
    # Save cleaned audio to file for transcription & sentiment analysis
    temp_audio_path = "cleaned_audio.wav"
    audio.export(temp_audio_path, format="wav")
    
    # Display buttons for both transcription and sentiment analysis
    col1, col2 = st.columns(2)
    modl = st.sidebar.selectbox("Select the Conversion Accuracy option", ["small.en", "mediam.en", "large"])
    with col1:
        if st.button("Convert Voice to Text "):
            progress_bar = st.progress(0)
            st.write("Conversion in Progress...")
            
            # Load a faster Whisper model
            model = whisper.load_model(modl)  # Using 'small' for faster processing
            
            # Simulate progress bar updates
            for percent in range(0, 101, 20):
                time.sleep(1)
                progress_bar.progress(percent)
            
            result = model.transcribe(temp_audio_path, fp16=False)  # Disable fp16 for CPU use
            transcription_text = result["text"]
            
            progress_bar.progress(100)
            st.subheader("Converted Text:")
            st.write(transcription_text)
    
    with col2:
        if st.button("Analyze Sentiment"):
            progress_bar = st.progress(0)
            st.write("Analyzing sentiment...")
            
            try:
                # Use a smaller emotion recognition model with feature extractor
                model_name = "superb/wav2vec2-base-superb-er"
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
                model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
                
                # Simulate progress bar updates
                for percent in range(0, 101, 20):
                    time.sleep(1)
                    progress_bar.progress(percent)
                
                # Process only a short segment (first 10 seconds) for efficiency
                audio_data, sr = librosa.load(temp_audio_path, sr=16000, duration=10)
                inputs = feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
                
                # Perform sentiment analysis
                with torch.no_grad():
                    logits = model(**inputs).logits
                
                emotions = ["Neutral", "Happy", "Sad", "Angry"]  # Adjust based on model output mapping
                sentiment = torch.argmax(logits, dim=-1).item()
                sentiment_label = emotions[sentiment]
                
                progress_bar.progress(100)
                st.subheader("Sentiment Analysis Result:")
                st.write(f"Emotion: **{sentiment_label}**")
            except Exception as e:
                st.error(f"Error in sentiment analysis: {str(e)}")
