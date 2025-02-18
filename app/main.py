import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional
import pickle
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import logging
import requests
import re
import pandas as pd
import textstat
from deep_translator import GoogleTranslator
from langdetect import detect
import time
import yt_dlp as youtube_dl
import librosa
from scipy import stats
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# CORS setup
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    running_order: float
    num_participants: float
    artist_name: str
    song_title: str
    youtube_url: str
    country: str
    elo_score: float

# Load the XGBoost model
try:
    with open("./app/model(1).pkl", "rb") as f:  # Updated path
        model = pickle.load(f)
    
    # Initialize model with a dummy prediction
    dummy_data = np.zeros((1, 11))
    _ = model.predict(dummy_data)
except Exception as e:
    logging.error(f"Error loading XGBoost model: {str(e)}")
    raise

logging.basicConfig(level=logging.DEBUG)

# Function to calculate percentile rank
def calculate_percentile(r, n):
    percentile = 1 - (r - 1) / (n - 1)
    return percentile

# Function to clean lyrics
def clean_lyrics(lyrics):
    intro_index = lyrics.find("[Intro]")
    if intro_index != -1:
        lyrics = lyrics[intro_index:]
    pattern = re.compile(r"\[.*?\]")
    cleaned_lyrics = re.sub(pattern, "", lyrics)
    cleaned_lyrics = "\n".join([line.strip() for line in cleaned_lyrics.splitlines() if line.strip()])
    return cleaned_lyrics

# Function to clean text
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'[^\w\s.,!?-]', '', str(text))
    return text.strip()

# Function to load Dale-Chall familiar words
def load_dale_chall_familiar_words(file_path):
    with open(file_path, 'r') as file:
        familiar_words = {line.strip().lower() for line in file}
    return familiar_words

# Function to calculate lyrical complexity
async def calculate_lyrical_complexity(text, dale_chall_familiar_words):
    try:
        if not text or len(text.strip()) == 0:
            return None, "Empty text"
        cleaned_text = clean_text(text)
        try:
            source_lang = detect(cleaned_text)
        except:
            source_lang = 'unknown'
        original_language = source_lang
        translated_text = cleaned_text
        if source_lang != 'en' and source_lang != 'unknown':
            try:
                translator = GoogleTranslator(source=source_lang, target='en')
                translated_text = translator.translate(cleaned_text)
                await asyncio.sleep(2)
            except Exception as e:
                print(f"Translation failed: {str(e)}")
                translated_text = cleaned_text
        words = [word for word in translated_text.split() if word]
        if not words:
            return None, "No valid words after processing"
        total_words = len(words)
        unique_words = set(words)
        syllables_per_word = sum(textstat.syllable_count(word) for word in words) / total_words
        percentage_unique_words = (len(unique_words) / total_words) * 100
        difficult_words_count = sum(1 for word in words if word.lower() not in dale_chall_familiar_words)
        percentage_difficult_words = (difficult_words_count / total_words) * 100
        metrics = {
            "Average_Syllables_per_Word": round(syllables_per_word, 2),
            "Percentage_Unique_Words": round(percentage_unique_words, 2),
            "Percentage_Difficult_Words": round(percentage_difficult_words, 2),
            "Original_Language": original_language
        }
        return metrics, None
    except Exception as e:
        return None, str(e)

# Function to analyze text
async def analyze_text(input_text, dale_chall_familiar_words_path):
    dale_chall_familiar_words = load_dale_chall_familiar_words(dale_chall_familiar_words_path)
    metrics, error = await calculate_lyrical_complexity(input_text, dale_chall_familiar_words)
    if error:
        print(f"Error during analysis: {error}")
        return None
    else:
        return metrics

# Function to download YouTube video as audio
async def download_youtube_video_as_audio(url):
    ydl_opts = {
        'outtmpl': 'temp_audio.%(ext)s',
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'cookiefile': './cookies.txt',  # Ensure this path is correct
        'verbose': True,  # Enable verbose logging
        'extract_flat': True,  # Skip signature extraction
        'ignoreerrors': True,  # Ignore errors and continue
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info).replace('.webm', '.mp3')
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None

# Function to extract features
async def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_skewness = stats.skew(chroma.ravel())
    chroma_std_dev = np.std(chroma)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff_median = np.median(spectral_rolloff)
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    onset_strength_skewness = stats.skew(onset_strength)
    onset_strength_mean = np.mean(onset_strength)
    return {
        'chroma_skewness': chroma_skewness,
        'spectral_rolloff_median': spectral_rolloff_median,
        'onset_strength_skewness': onset_strength_skewness,
        'chroma_std_dev': chroma_std_dev,
        'onset_strength_mean': onset_strength_mean
    }

# Function to fetch lyrics from lyrics.ovh
async def fetch_lyrics(artist, song_title):
    base_url = "https://api.lyrics.ovh/v1"
    url = f"{base_url}/{artist}/{song_title}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('lyrics', '')
    else:
        print(f"Failed to fetch lyrics: {response.status_code}")
        return None

@app.post('/predict')
async def predict_model(input_parameters: model_input, request: Request):
    try:
        logging.debug(f"Received input: {input_parameters}")
        
        # Calculate percentile rank
        percentile_rank = calculate_percentile(input_parameters.running_order, input_parameters.num_participants)
        
        # Fetch lyrics
        lyrics = await fetch_lyrics(input_parameters.artist_name, input_parameters.song_title)
        if lyrics:
            lyrics = clean_lyrics(lyrics)
            dale_chall_word_list_path = "./app/DaleChallEasyWordList.txt"  # Updated path
            results = await analyze_text(lyrics, dale_chall_word_list_path)
            if results:
                if results["Original_Language"] == 'en':
                    Percentage_Difficult_Words = results['Percentage_Difficult_Words']
                    Percentage_Unique_Words = results['Percentage_Unique_Words']
                else:
                    Percentage_Difficult_Words = None
                    Percentage_Unique_Words = None
            else:
                Percentage_Difficult_Words = None
                Percentage_Unique_Words = None
        else:
            Percentage_Difficult_Words = None
            Percentage_Unique_Words = None
        
        # Analyze YouTube audio
        audio_path = await download_youtube_video_as_audio(input_parameters.youtube_url)
        if audio_path is None:
            raise HTTPException(status_code=400, detail="Failed to download audio")
        try:
            features = await extract_features(audio_path)
            chroma_skewness = features['chroma_skewness']
            spectral_rolloff_median = features['spectral_rolloff_median']
            onset_strength_skewness = features['onset_strength_skewness']
            chroma_std_dev = features['chroma_std_dev']
            onset_strength_mean = features['onset_strength_mean']
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        
        # Get LTO and Hierarchy values
        LTO_values = {
            'Albania': 56, 'Armenia': 38, 'Australia': 56, 'Austria': 47, 'Azerbaijan': 59,
            'Belgium': 61, 'Croatia': 40, 'Cyprus': 59, 'Czechia': 51, 'Denmark': 59,
            'Estonia': 71, 'Finland': 63, 'France': 60, 'Georgia': 24, 'Germany': 57,
            'Greece': 51, 'Iceland': 57, 'Ireland': 51, 'Israel': 47, 'Italy': 39,
            'Latvia': 69, 'Lithuania': 49, 'Luxembourg': 64, 'Malta': 47, 'Moldova': 71,
            'Montenegro': 40, 'Netherlands': 67, 'Norway': 55, 'Poland': 49, 'Portugal': 42,
            'Romania': 32, 'Russia': 58, 'San Marino': 39, 'Serbia': 37, 'Slovakia': 53,
            'Slovenia': 50, 'Spain': 47, 'Sweden': 52, 'Switzerland': 42, 'Turkey': 35,
            'Ukraine': 51, 'United Kingdom': 60, 'USA': 50, 'Canada': 54
        }
        Hierarchy_values = {
            'Australia': 2.29, 'Austria': 1.75, 'Belgium': 1.69, 'Croatia': 2.55,
            'Cyprus': 1.96, 'Czechia': 2.22, 'Denmark': 1.86, 'Estonia': 2.04,
            'Finland': 1.8, 'France': 2.21, 'Georgia': 2.46, 'Germany': 1.87,
            'Greece': 1.83, 'Ireland': 2.09, 'Israel': 2.51, 'Italy': 1.6,
            'Latvia': 1.8, 'Netherlands': 1.91, 'Norway': 1.49, 'Poland': 2.51,
            'Portugal': 1.89, 'Romania': 2, 'Russia': 2.72, 'Serbia': 1.61,
            'Slovakia': 2, 'Slovenia': 1.62, 'Spain': 1.84, 'Sweden': 1.83,
            'Switzerland': 2.42, 'Turkey': 2.97, 'Ukraine': 2.56, 'United Kingdom': 2.33
        }
        LTO = LTO_values.get(input_parameters.country, None)
        Hierarchy = Hierarchy_values.get(input_parameters.country, None)
        
        # Prepare new observation
        new_observation = np.array([[
            percentile_rank, chroma_skewness, Percentage_Difficult_Words,
            spectral_rolloff_median, onset_strength_skewness, chroma_std_dev,
            Percentage_Unique_Words, onset_strength_mean, LTO, Hierarchy, input_parameters.elo_score
        ]])
        
        # Predict using the model
        probabilities = model.predict_proba(new_observation)
        prediction = float(probabilities[0][1])
        
        return {'prediction': prediction}
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
