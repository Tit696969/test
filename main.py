import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
import yt_dlp as youtube_dl
import librosa
from scipy import stats
import os
import asyncio
import random

# Initialize FastAPI
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Client configurations
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
]

CLIENT_CONFIGS = {
    "web": {
        "user_agent": random.choice(USER_AGENTS),
        "extractor_args": {"youtube": {"player_client": ["web"]}},
    },
    "mobile": {
        "user_agent": "com.google.android.youtube/17.36.4 (Linux; U; Android 13; en_US)",
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }
}

# Pydantic model
class ModelInput(BaseModel):
    running_order: float
    num_participants: float
    artist_name: str
    song_title: str
    youtube_url: str
    country: str
    elo_score: float

# Load model
try:
    with open("./app/model.pkl", "rb") as f:
        model = pickle.load(f)
    # Warmup model
    model.predict(np.zeros((1, 11)))
except Exception as e:
    logging.error(f"Model loading error: {str(e)}")
    raise

# Helper functions
def calculate_percentile(running_order, num_participants):
    return 1 - (running_order - 1) / (num_participants - 1)

def clean_lyrics(lyrics):
    return re.sub(r"\[.*?\]", "", lyrics.split("[Intro]")[-1]) if "[Intro]" in lyrics else lyrics

# Lyrics processing
async def fetch_lyrics(artist, title):
    """Fetch lyrics from Lyrics.ovh API"""
    try:
        url = f"https://api.lyrics.ovh/v1/{artist}/{title}"
        response = requests.get(url, timeout=10)
        return response.json().get("lyrics", "") if response.status_code == 200 else ""
    except Exception as e:
        logging.error(f"Lyrics fetch error: {str(e)}")
        return ""

def load_dale_chall_words(file_path):
    with open(file_path, "r") as f:
        return {line.strip().lower() for line in f}

async def analyze_lyrics(text, dale_chall_words):
    if not text.strip():
        return None
    
    try:
        # Language detection and translation
        lang = detect(text)
        if lang != "en":
            text = GoogleTranslator(source=lang, target="en").translate(text)
            await asyncio.sleep(1.5)

        # Text processing
        words = [w.lower() for w in re.findall(r"\w+", text)]
        if not words:
            return None

        # Calculate metrics
        return {
            "difficult": sum(1 for w in words if w not in dale_chall_words) / len(words),
            "unique": len(set(words)) / len(words),
            "syllables": sum(textstat.syllable_count(w) for w in words) / len(words)
        }
    except Exception as e:
        logging.error(f"Lyrics analysis error: {str(e)}")
        return None

# Audio processing
async def download_audio(url):
    ydl_opts = {
        "outtmpl": "temp_audio.%(ext)s",
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "cookiesfrombrowser": ("chrome",),
        "ignoreerrors": True,
        "retries": 2,
        "sleep_interval": random.randint(3, 7),
        "http_headers": {"User-Agent": random.choice(USER_AGENTS)},
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = await asyncio.to_thread(ydl.extract_info, url, download=True)
            return ydl.prepare_filename(info).replace(".webm", ".mp3")
    except Exception as e:
        logging.error(f"Audio download failed: {str(e)}")
        return None

def extract_audio_features(path):
    try:
        y, sr = librosa.load(path)
        return {
            "chroma_skew": stats.skew(librosa.feature.chroma_cqt(y=y, sr=sr).ravel()),
            "spectral_rolloff": np.median(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            "onset_strength": np.mean(librosa.onset.onset_strength(y=y, sr=sr))
        }
    finally:
        if os.path.exists(path):
            os.remove(path)

# Country data (example values - fill with real data)
LTO_VALUES = {"Sweden": 53, "Norway": 47}
HIERARCHY_VALUES = {"Sweden": 1.8, "Norway": 2.1}

# API endpoint
@app.post("/predict")
async def predict(input_data: ModelInput):
    try:
        # Lyrics analysis
        lyrics = await fetch_lyrics(input_data.artist_name, input_data.song_title)
        lyrics_metrics = await analyze_lyrics(
            clean_lyrics(lyrics), 
            load_dale_chall_words("./app/DaleChallEasyWordList.txt")
        ) if lyrics else None

        # Audio analysis
        audio_path = await download_audio(input_data.youtube_url)
        if not audio_path:
            raise HTTPException(400, "Audio download failed")
        audio_features = await asyncio.to_thread(extract_audio_features, audio_path)

        # Prepare features
        features = [
            calculate_percentile(input_data.running_order, input_data.num_participants),
            audio_features["chroma_skew"],
            lyrics_metrics["difficult"] if lyrics_metrics else 0.5,
            audio_features["spectral_rolloff"],
            lyrics_metrics["unique"] if lyrics_metrics else 0.3,
            audio_features["onset_strength"],
            LTO_VALUES.get(input_data.country, 50),
            HIERARCHY_VALUES.get(input_data.country, 2.0),
            input_data.elo_score
        ]

        # Make prediction
        prediction = model.predict_proba(np.array([features]))[0][1]
        return {"prediction": float(prediction)}
    
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(500, "Prediction failed") from e

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
