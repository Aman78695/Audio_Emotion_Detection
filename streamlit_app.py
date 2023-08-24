import streamlit as st
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline, RobertaTokenizer, RobertaForSequenceClassification
import librosa
import soundfile as sf
import os
import torch

# Load ASR model and processor
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Load emotion detection model
emotion_detection = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", tokenizer="roberta-base")

# Load text summarization pipeline
summarizer = pipeline("summarization")

# Helper function to remove noise from audio
def remove_noise(audio, sample_rate):
    denoised_audio, _ = librosa.effects.trim(audio)
    return denoised_audio

# Main Streamlit app
def main():
    
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Voice Analysis and Summarization")

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Load and process audio
        audio, sample_rate = torchaudio.load(uploaded_file)
        audio = audio.numpy()
        denoised_audio = remove_noise(audio, sample_rate)
        
        # Transcribe audio to text
        input_values = processor(denoised_audio, return_tensors="pt").input_values
        with torch.no_grad():
            logits = asr_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcribed_text = processor.batch_decode(predicted_ids)
        transcribed_text = transcribed_text[0]

        # Detect emotion in transcribed text
        emotion_result = emotion_detection(transcribed_text)
        emotion = emotion_result[0]['label']

        # Summarize transcribed text
        summary = summarizer(transcribed_text, max_length=100, min_length=5, do_sample=True)
        summarized_text = summary[0]['summary_text']

        # Display results
        st.header("Transcribed Text:")
        st.write(transcribed_text)
        
        st.header("Detected Emotion:")
        st.write(emotion)
        
        st.header("Summarized Text:")
        st.write(summarized_text)

if __name__ == "__main__":
    main()
