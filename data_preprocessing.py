"""
Veri ön işleme modülü
CNN/DailyMail veri setini indirme, temizleme ve hazırlama
"""

import os
import re
import logging
from typing import Dict, List
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from config import Config
import nltk
from tqdm import tqdm

# NLTK punkt tokenizer'ı indir
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DataPreprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        
        # T5 için özel prefix ekleme
        if config.MODEL_TYPE == "t5":
            self.prefix = "summarize: "
        else:
            self.prefix = ""
            
        self._setup_logging()
        
    def _setup_logging(self):
        """Logging kurulumu"""
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.LOG_DIR, 'preprocessing.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def clean_text(self, text: str) -> str:
        """Metni temizle"""
        if not text:
            return ""
            
        # HTML etiketlerini kaldır
        text = re.sub(r'<[^>]+>', '', text)
        
        # Fazla boşlukları kaldır
        text = ' '.join(text.split())
        
        # Özel karakterleri normalize et
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Çok uzun cümleleri ayır
        sentences = nltk.sent_tokenize(text)
        text = ' '.join(sentences)
        
        return text.strip()

    def preprocess_function(self, examples):
        """Veri setindeki her örneği işle"""
        # Giriş metinlerini temizle ve prefix ekle
        inputs = [self.prefix + self.clean_text(doc) for doc in examples["article"]]
        
        # Özet metinlerini temizle
        targets = [self.clean_text(summary) for summary in examples["highlights"]]
        
        # Tokenize et
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.MAX_INPUT_LENGTH,
            truncation=True,
            padding=False  # Batch içinde padding yapacağız
        )
        
        # Özet tokenize et
        labels = self.tokenizer(
            targets,
            max_length=self.config.MAX_TARGET_LENGTH,
            truncation=True,
            padding=False
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def load_and_preprocess_data(self):
        """Veri setini yükle ve ön işleme yap"""
        self.logger.info("CNN/DailyMail veri seti yükleniyor...")
        
        try:
            # Veri setini yükle
            dataset = load_dataset(self.config.DATASET_NAME, self.config.DATASET_VERSION)
            
            self.logger.info(f"Orijinal veri boyutları:")
            self.logger.info(f"Train: {len(dataset['train'])}")
            self.logger.info(f"Validation: {len(dataset['validation'])}")
            self.logger.info(f"Test: {len(dataset['test'])}")
            
            # Küçük alt küme al (hızlı prototip için)
            train_dataset = dataset['train'].select(range(min(self.config.TRAIN_SIZE, len(dataset['train']))))
            val_dataset = dataset['validation'].select(range(min(self.config.VAL_SIZE, len(dataset['validation']))))
            test_dataset = dataset['test'].select(range(min(self.config.TEST_SIZE, len(dataset['test']))))
            
            self.logger.info(f"Kullanılacak veri boyutları:")
            self.logger.info(f"Train: {len(train_dataset)}")
            self.logger.info(f"Validation: {len(val_dataset)}")
            self.logger.info(f"Test: {len(test_dataset)}")
            
            # Ön işleme uygula
            self.logger.info("Veri ön işleme yapılıyor...")
            
            train_dataset = train_dataset.map(
                self.preprocess_function,
                batched=True,
                desc="Training data preprocessing"
            )
            
            val_dataset = val_dataset.map(
                self.preprocess_function,
                batched=True,
                desc="Validation data preprocessing"
            )
            
            test_dataset = test_dataset.map(
                self.preprocess_function,
                batched=True,
                desc="Test data preprocessing"
            )
            
            # Gereksiz sütunları kaldır
            columns_to_remove = ["article", "highlights", "id"]
            train_dataset = train_dataset.remove_columns(columns_to_remove)
            val_dataset = val_dataset.remove_columns(columns_to_remove)
            test_dataset = test_dataset.remove_columns(columns_to_remove)
            
            # Processed klasörüne kaydet
            self.logger.info("İşlenmiş veri kaydediliyor...")
            
            train_dataset.save_to_disk(os.path.join(self.config.PROCESSED_DATA_DIR, "train"))
            val_dataset.save_to_disk(os.path.join(self.config.PROCESSED_DATA_DIR, "validation"))
            test_dataset.save_to_disk(os.path.join(self.config.PROCESSED_DATA_DIR, "test"))
            
            self.logger.info("Veri ön işleme tamamlandı!")
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            self.logger.error(f"Veri yükleme hatası: {str(e)}")
            raise
    
    def load_processed_data(self):
        """İşlenmiş veriyi yükle"""
        try:
            train_dataset = Dataset.load_from_disk(os.path.join(self.config.PROCESSED_DATA_DIR, "train"))
            val_dataset = Dataset.load_from_disk(os.path.join(self.config.PROCESSED_DATA_DIR, "validation"))
            test_dataset = Dataset.load_from_disk(os.path.join(self.config.PROCESSED_DATA_DIR, "test"))
            
            self.logger.info("İşlenmiş veri başarıyla yüklendi!")
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            self.logger.warning(f"İşlenmiş veri yüklenemedi: {str(e)}")
            self.logger.info("Veri ön işleme başlatılıyor...")
            return self.load_and_preprocess_data()
    
    def get_sample_data(self, dataset, num_samples=5):
        """Örnek veri göster"""
        samples = []
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            
            # Token'ları tekrar metne çevir
            input_text = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
            if sample.get('labels'):
                target_text = self.tokenizer.decode(sample['labels'], skip_special_tokens=True)
            else:
                target_text = "N/A"
            
            samples.append({
                'input': input_text,
                'target': target_text,
                'input_length': len(sample['input_ids']),
                'target_length': len(sample.get('labels', []))
            })
        
        return samples

def main():
    """Ana fonksiyon"""
    # Konfigürasyonu yükle
    config = Config()
    config.create_dirs()
    config.print_config()
    
    # Veri ön işleyiciyi başlat
    preprocessor = DataPreprocessor(config)
    
    # Veriyi yükle ve işle
    train_dataset, val_dataset, test_dataset = preprocessor.load_processed_data()
    
    # Örnek verileri göster
    print("\n=== ÖRNEK VERİLER ===")
    samples = preprocessor.get_sample_data(train_dataset, 3)
    
    for i, sample in enumerate(samples, 1):
        print(f"\nÖrnek {i}:")
        print(f"Giriş ({sample['input_length']} token): {sample['input'][:200]}...")
        print(f"Hedef ({sample['target_length']} token): {sample['target']}")
        print("-" * 50)

if __name__ == "__main__":
    main()