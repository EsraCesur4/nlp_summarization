"""
Konfigürasyon dosyası - Tüm proje ayarları burada
"""

import os
from dataclasses import dataclass
import torch

@dataclass
class Config:
    # Model ayarları
    MODEL_NAME = "t5-small"  # Alternatif: "facebook/bart-base"
    MODEL_TYPE = "t5"  # "t5" veya "bart"
    
    # Veri seti ayarları
    DATASET_NAME = "cnn_dailymail"
    DATASET_VERSION = "3.0.0"
    
    # Metin işleme ayarları
    MAX_INPUT_LENGTH = 512   # Giriş metni maksimum uzunluğu
    MAX_TARGET_LENGTH = 128  # Özet maksimum uzunluğu
    MIN_TARGET_LENGTH = 10   # Özet minimum uzunluğu
    
    # Eğitim ayarları
    BATCH_SIZE = 4          # Düşük RAM için küçük batch size
    EVAL_BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Veri ayarları
    TRAIN_SIZE = 5000       # Hızlı prototip için küçük veri seti
    VAL_SIZE = 1000
    TEST_SIZE = 1000
    
    # Dosya yolları
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Windows'ta Türkçe karakter sorunu için güvenli path oluştur
    SAFE_PROJECT_NAME = "nlp_summarization_project"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
    EVALUATION_DIR = os.path.join(OUTPUT_DIR, "evaluation_results")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    
    # Değerlendirme ayarları
    ROUGE_METRICS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    
    # Sistem ayarları
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 0 if DEVICE == "cpu" else 2  # CPU'da multiprocessing sorun çıkarabilir
    SEED = 42
    
    # Logging ayarları
    LOG_LEVEL = "INFO"
    SAVE_STEPS = 500
    EVAL_STEPS = 500
    LOGGING_STEPS = 100
    
    # Çıkarım ayarları
    DO_SAMPLE = True
    NUM_BEAMS = 4
    TEMPERATURE = 0.7
    TOP_P = 0.9
    NO_REPEAT_NGRAM_SIZE = 2
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        dirs = [
            cls.DATA_DIR, cls.PROCESSED_DATA_DIR, cls.RAW_DATA_DIR,
            cls.MODEL_DIR, cls.CHECKPOINT_DIR, cls.OUTPUT_DIR,
            cls.PREDICTIONS_DIR, cls.EVALUATION_DIR, cls.LOG_DIR
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
    @classmethod
    def print_config(cls):
        """Print configuration"""
        print("=== PROJECT CONFIGURATION ===")
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Number of Epochs: {cls.NUM_EPOCHS}")
        print(f"Training Data Size: {cls.TRAIN_SIZE}")
        print(f"Max Input Length: {cls.MAX_INPUT_LENGTH}")
        print(f"Max Summary Length: {cls.MAX_TARGET_LENGTH}")
        print("=" * 30)