"""
Model eğitim modülü
T5/BART modelini CNN/DailyMail veri seti üzerinde eğitme
"""

import os
import logging
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
import numpy as np
from config import Config
from data_preprocessing import DataPreprocessor
import matplotlib.pyplot as plt
import json

class ModelTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.model = None
        self.trainer = None
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Logging kurulumu"""
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.LOG_DIR, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        """Modeli yükle"""
        self.logger.info(f"Model yükleniyor: {self.config.MODEL_NAME}")
        
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.MODEL_NAME)
            
            # GPU varsa modeli GPU'ya taşı
            if self.config.DEVICE == "cuda":
                self.model = self.model.to(self.config.DEVICE)
                self.logger.info("Model GPU'ya taşındı")
            else:
                self.logger.info("CPU kullanılıyor")
                
            self.logger.info("Model başarıyla yüklendi!")
            
        except Exception as e:
            self.logger.error(f"Model yükleme hatası: {str(e)}")
            raise
    
    def setup_training_arguments(self):
        """Eğitim argümanlarını ayarla"""
        # Güvenli checkpoint path oluştur
        checkpoint_dir = os.path.normpath(self.config.CHECKPOINT_DIR)
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=checkpoint_dir,
            
            # Eğitim ayarları
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.EVAL_BATCH_SIZE,
            learning_rate=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            warmup_steps=self.config.WARMUP_STEPS,
            
            # Değerlendirme ayarları
            eval_strategy="steps",
            eval_steps=self.config.EVAL_STEPS,
            save_steps=self.config.SAVE_STEPS,
            logging_steps=self.config.LOGGING_STEPS,
            
            # Kaydetme ayarları
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Çıkarım ayarları
            predict_with_generate=True,
            generation_max_length=self.config.MAX_TARGET_LENGTH,
            generation_num_beams=self.config.NUM_BEAMS,
            
            # Sistem ayarları
            dataloader_num_workers=self.config.NUM_WORKERS,
            fp16=True if self.config.DEVICE == "cuda" else False,  # GPU'da mixed precision
            
            # Logging
            logging_dir=os.path.normpath(os.path.join(self.config.LOG_DIR, "tensorboard")),
            report_to=None,  # Tensorboard kapalı
            
            # Memory optimization
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        return training_args
    
    def setup_trainer(self, train_dataset, val_dataset):
        """Trainer'ı kur"""
        self.logger.info("Trainer kuruluyor...")
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        # Training arguments
        training_args = self.setup_training_arguments()
        
        # Trainer
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        self.logger.info("Trainer başarıyla kuruldu!")
    
    def train_model(self, train_dataset, val_dataset):
        """Modeli eğit"""
        self.logger.info("Model eğitimi başlıyor...")
        
        try:
            # Modeli yükle
            self.load_model()
            
            # Trainer'ı kur
            self.setup_trainer(train_dataset, val_dataset)
            
            # Eğitimi başlat
            self.logger.info(f"Eğitim parametreleri:")
            self.logger.info(f"- Epoch sayısı: {self.config.NUM_EPOCHS}")
            self.logger.info(f"- Batch size: {self.config.BATCH_SIZE}")
            self.logger.info(f"- Learning rate: {self.config.LEARNING_RATE}")
            self.logger.info(f"- Toplam eğitim örneği: {len(train_dataset)}")
            self.logger.info(f"- Toplam validasyon örneği: {len(val_dataset)}")
            
            # Eğitim
            train_result = self.trainer.train()
            
            # Eğitim sonuçlarını kaydet
            train_metrics = train_result.metrics
            self.logger.info("Eğitim tamamlandı!")
            self.logger.info(f"Final train loss: {train_metrics['train_loss']:.4f}")
            
            # Modeli kaydet - Güvenli path için temp klasör kullan
            import tempfile
            import shutil
            
            # Geçici güvenli bir path oluştur
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_model_path = os.path.join(temp_dir, "model")
                
                # Geçici yere kaydet
                self.trainer.save_model(temp_model_path)
                self.tokenizer.save_pretrained(temp_model_path)
                
                # Final lokasyona kopyala
                final_model_path = os.path.join(self.config.MODEL_DIR, "final_model")
                os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
                
                # Eğer klasör varsa sil
                if os.path.exists(final_model_path):
                    shutil.rmtree(final_model_path)
                
                # Kopyala
                shutil.copytree(temp_model_path, final_model_path)
            
            self.logger.info(f"Model kaydedildi: {final_model_path}")
            
            # Eğitim geçmişini kaydet
            self._save_training_history(train_result)
            
            return train_result
            
        except Exception as e:
            self.logger.error(f"Eğitim hatası: {str(e)}")
            raise
    
    def _save_training_history(self, train_result):
        """Eğitim geçmişini kaydet"""
        try:
            # Log geçmişini al
            log_history = self.trainer.state.log_history
            
            # JSON olarak kaydet
            history_path = os.path.join(self.config.OUTPUT_DIR, "training_history.json")
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(log_history, f, indent=2, ensure_ascii=False)
            
            # Grafik oluştur
            self._plot_training_curves(log_history)
            
            self.logger.info(f"Eğitim geçmişi kaydedildi: {history_path}")
            
        except Exception as e:
            self.logger.warning(f"Eğitim geçmişi kaydetme hatası: {str(e)}")
    
    def _plot_training_curves(self, log_history):
        """Eğitim grafiklerini çiz"""
        try:
            # Loss değerlerini ayır
            train_losses = []
            eval_losses = []
            steps = []
            
            for log in log_history:
                if 'train_loss' in log:
                    train_losses.append(log['train_loss'])
                    steps.append(log['step'])
                if 'eval_loss' in log:
                    eval_losses.append(log['eval_loss'])
            
            # Grafik çiz
            plt.figure(figsize=(12, 5))
            
            # Training loss
            plt.subplot(1, 2, 1)
            plt.plot(steps, train_losses, 'b-', label='Training Loss')
            plt.title('Training Loss')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Evaluation loss
            if eval_losses:
                plt.subplot(1, 2, 2)
                eval_steps = [log['step'] for log in log_history if 'eval_loss' in log]
                plt.plot(eval_steps, eval_losses, 'r-', label='Validation Loss')
                plt.title('Validation Loss')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            
            # Kaydet
            plot_path = os.path.join(self.config.OUTPUT_DIR, "training_curves.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Eğitim grafikleri kaydedildi: {plot_path}")
            
        except Exception as e:
            self.logger.warning(f"Grafik oluşturma hatası: {str(e)}")
    
    def load_trained_model(self, model_path=None):
        """Eğitilmiş modeli yükle"""
        if model_path is None:
            model_path = os.path.join(self.config.MODEL_DIR, "final_model")
        
        # Path'i normalize et
        model_path = os.path.normpath(model_path)
        
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if self.config.DEVICE == "cuda":
                self.model = self.model.to(self.config.DEVICE)
            
            self.logger.info(f"Eğitilmiş model yüklendi: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Model yükleme hatası: {str(e)}")
            raise

def main():
    """Ana fonksiyon"""
    # Konfigürasyonu yükle
    config = Config()
    config.create_dirs()
    config.print_config()
    
    # Veri ön işleyiciyi başlat
    preprocessor = DataPreprocessor(config)
    
    # Veriyi yükle
    train_dataset, val_dataset, _ = preprocessor.load_processed_data()
    
    # Model trainer'ı başlat
    trainer = ModelTrainer(config)
    
    # Modeli eğit
    train_result = trainer.train_model(train_dataset, val_dataset)
    
    print(f"\n=== EĞİTİM TAMAMLANDI ===")
    print(f"Final loss: {train_result.metrics['train_loss']:.4f}")
    print(f"Model kaydedildi: {os.path.join(config.MODEL_DIR, 'final_model')}")

if __name__ == "__main__":
    main()