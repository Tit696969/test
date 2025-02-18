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
import random

# Initialize FastAPI
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

# Define client configurations
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 13; SM-S908B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36",
]

CLIENT_CONFIGS = {
    "web": {
        "user-agent": random.choice(USER_AGENTS),
        "extractor_args": {"youtube": {"player_client": ["web"]}},
    },
    "android": {
        "user-agent": "com.google.android.youtube/17.36.4 (Linux; U; Android 13; en_US)",
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    },
    "tv": {
        "user-agent": "com.google.android.youtube.tv/17.36.4 (Linux; U; Android 13; en_US)",
        "extractor_args": {"youtube": {"player_client": ["android", "tv"]}},
    }
}

# Define Pydantic model
class model_input(BaseModel):
    running_order: float
    num_participants: float
    artist_name: str
    song_title: str
    youtube_url: str
    country: str
    elo_score: float

# Load XGBoost model
try:
    with open("./app/model(1).pkl", "rb") as f:
        model = pickle.load(f)
    dummy_data = np.zeros((1, 11))
    _ = model.predict(dummy_data)
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

# Helper functions
def calculate_percentile(r, n):
    return 1 - (r - 1) / (n - 1)

def clean_lyrics(lyrics):
    intro_index = lyrics.find("[Intro]")
    if intro_index != -1:
        lyrics = lyrics[intro_index:]
    return re.sub(r"\[.*?\]", "", lyrics)

def clean_text(text):
    return re.sub(r'[^\w\s.,!?-]', '', str(text)).strip() if pd.notna(text) else ""

# Lyrics analysis functions
def load_dale_chall_familiar_words(file_path):
    with open(file_path, 'r') as file:
        return {line.strip().lower() for line in file}

async def calculate_lyrical_complexity(text, dale_chall_familiar_words):
    try:
        if not text.strip(): return None, "Empty text"
        cleaned_text = clean_text(text)
        source_lang = detect(cleaned_text) if cleaned_text else 'unknown'
        translated_text = cleaned_text
        
        if source_lang not in ['en', 'unknown']:
            try:
                translated_text = GoogleTranslator(source=source_lang, target='en').translate(cleaned_text)
                await asyncio.sleep(2)
            except: pass

        words = translated_text.split()
        if not words: return None, "No valid words"
        
        syllables = sum(textstat.syllable_count(word) for word in words) / len(words)
        unique = (len(set(words)) / len(words)) * 100
        difficult = (sum(1 for word in words if word.lower() not in dale_chall_familiar_words) / len(words)) * 100
        
        return {
            "Average_Syllables_per_Word": round(syllables, 2),
            "Percentage_Unique_Words": round(unique, 2),
            "Percentage_Difficult_Words": round(difficult, 2),
            "Original_Language": source_lang
        }, None
    except Exception as e:
        return None, str(e)

# YouTube download and analysis
async def download_youtube_video_as_audio(url):
    try:
        client_config = random.choice(list(CLIENT_CONFIGS.values()))
        
        ydl_opts = {
            "outtmpl": "temp_audio.%(ext)s",
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "user-agent": client_config["user-agent"],
            "extractor_args": client_config["extractor_args"],
            "cookiesfrombrowser": ("chrome",),
            "ignoreerrors": True,
            "retries": 3,
            "sleep_interval": random.randint(5, 10),
            "http_headers": {
                "Accept-Language": "en-US,en;q=0.9",
                "Sec-Fetch-Dest": "document",
            },
        }

        await asyncio.sleep(random.uniform(1, 3))
        
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info).replace(".webm", ".mp3")
    except Exception as e:
        logging.error(f"Download failed: {str(e)}")
        return None

async def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path)
        return {
            'chroma_skewness': stats.skew(librosa.feature.chroma_cqt(y=y, sr=sr).ravel()),
            'spectral_rolloff_median': np.median(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            'onset_strength_skewness': stats.skew(librosa.onset.onset_strength(y=y, sr=sr)),
            'chroma_std_dev': np.std(librosa.feature.chroma_cqt(y=y, sr=sr)),
            'onset_strength_mean': np.mean(librosa.onset.onset_strength(y=y, sr=sr))
        }
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# API endpoint
@app.post('/predict')
async def predict_model(input_parameters: model_input):
    try:
        # Lyrics analysis
        lyrics = await fetch_lyrics(input_parameters.artist_name, input_parameters.song_title)
        lyrics_metrics = None
        if lyrics:
            dale_chall = load_dale_chall_familiar_words("./app/DaleChallEasyWordList.txt")
            metrics, _ = await calculate_lyrical_complexity(clean_lyrics(lyrics), dale_chall)
            if metrics and metrics["Original_Language"] == 'en':
                lyrics_metrics = (metrics['Percentage_Difficult_Words'], metrics['Percentage_Unique_Words'])
        
        # Audio analysis
        audio_path = await download_youtube_video_as_audio(input_parameters.youtube_url)
        if not audio_path:
            raise HTTPException(400, "Audio download failed")
        
        features = await extract_features(audio_path)
        
        # Country metrics
        LTO_VALUES = {}  # Add your country data here
        HIERARCHY_VALUES = {}  # Add your country data here
        
        # Prepare model input
        observation = np.array([[
            calculate_percentile(input_parameters.running_order, input_parameters.num_participants),
            features['chroma_skewness'],
            lyrics_metrics[0] if lyrics_metrics else 0,
            features['spectral_rolloff_median'],
            features['onset_strength_skewness'],
            features['chroma_std_dev'],
            lyrics_metrics[1] if lyrics_metrics else 0,
            features['onset_strength_mean'],
            LTO_VALUES.get(input_parameters.country, 50),
            HIERARCHY_VALUES.get(input_parameters.country, 2.0),
            input_parameters.elo_score
        ]])
        
        # Make prediction
        probabilities = model.predict_proba(observation)
        return {'prediction': float(probabilities[0][1])}
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
