import librosa
import numpy as np
from PIL import Image
import scipy.signal as signal
import logging

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename: str, allowed_extensions: set) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def preprocess_audio(file_path: str, target_size=(224, 224)):
    """
    Advanced preprocessing pipeline:
    1. Load Audio -> 2. Bandpass Filter -> 3. Mel-Spectrogram -> 4. Image Conversion
    """
    try:
        # 1. Load Audio (Force 16kHz mono)
        y, sr = librosa.load(file_path, sr=16000, mono=True)
        
        # 2. Band-Pass Filter (The "Noise Killer")
        # Removes AC hum (<100Hz) and high-freq hiss (>8000Hz)
        sos = signal.butter(10, [100, 8000], 'bp', fs=sr, output='sos')
        y_filtered = signal.sosfilt(sos, y)
        
        # 3. Generate Mel-Spectrogram
        mel_spect = librosa.feature.melspectrogram(
            y=y_filtered, 
            sr=sr, 
            n_mels=128, 
            fmax=8000
        )
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

        # 4. Normalize to 0-255 (Standard Image format)
        min_val, max_val = mel_spect_db.min(), mel_spect_db.max()
        if max_val - min_val == 0:
            mel_norm = np.zeros_like(mel_spect_db, dtype=np.uint8)
        else:
            mel_norm = 255 * (mel_spect_db - min_val) / (max_val - min_val)
            mel_norm = mel_norm.astype(np.uint8)

        # 5. Resize and Convert to RGB
        # We stack the single channel 3 times because ResNet expects RGB
        img = Image.fromarray(mel_norm)
        img = img.resize(target_size)
        img = img.convert('RGB')
        
        # 6. Prepare Tensor
        img_array = np.array(img) / 255.0  # Normalize to 0-1 range
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension: (1, 224, 224, 3)
        
        return img_array

    except Exception as e:
        logger.error(f"Error processing audio file {file_path}: {e}")
        raise e