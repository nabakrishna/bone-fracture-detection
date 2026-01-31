import os

class Config:
    # Basic Config
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-prod'
    
    # File Storage
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Max upload size: 16MB
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'webm'}
    
    # Model Config
    MODEL_PATH = os.path.join(os.getcwd(), 'model', 'cough_classifier.h5')
    
    # DSP Config
    SAMPLE_RATE = 16000
    N_MELS = 128
    FMIN = 100   # High-pass filter (remove rumble)
    FMAX = 8000  # Low-pass filter (human voice limit)