"""
Model değerlendirme modülü
ROUGE metrikleri ile özet kalitesini ölçme
"""

import os
import logging
import json
import pandas as pd
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm
from config import Config
from data_preprocessing import DataPreprocessor
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.rouge_scorer = rouge_scorer.RougeScorer(
            config.ROUGE_METRICS, 
            use_stemmer=True
        )
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Logging kurulumu"""
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.LOG_DIR, 'evaluation.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, model_path=None):
        """Eğitilmiş modeli yükle - Yoksa pre-trained model kullan"""
        if model_path is None:
            model_path = os.path.join(self.config.MODEL_DIR, "final_model")
        
        # Path'i normalize et
        model_path = os.path.normpath(model_path)
        
        try:
            # Önce lokal modeli dene
            if os.path.exists(model_path) and os.path.isdir(model_path):
                self.logger.info(f"Lokal model yükleniyor: {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            else:
                # Lokal model yoksa pre-trained model kullan
                self.logger.info(f"Lokal model bulunamadı, pre-trained model kullanılıyor: {self.config.MODEL_NAME}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.MODEL_NAME)
            
            if self.config.DEVICE == "cuda":
                self.model = self.model.to(self.config.DEVICE)
            
            self.model.eval()  # Değerlendirme moduna geç
            
            self.logger.info("Model başarıyla yüklendi!")
            
        except Exception as e:
            self.logger.error(f"Model yükleme hatası: {str(e)}")
            raise
    
    def generate_summary(self, input_text: str) -> str:
        """Tek bir metin için özet üret"""
        try:
            # T5 için prefix ekle
            if self.config.MODEL_TYPE == "t5":
                input_text = "summarize: " + input_text
            
            # Tokenize et
            inputs = self.tokenizer(
                input_text,
                max_length=self.config.MAX_INPUT_LENGTH,
                truncation=True,
                return_tensors="pt"
            )
            
            if self.config.DEVICE == "cuda":
                inputs = {k: v.to(self.config.DEVICE) for k, v in inputs.items()}
            
            # Özet üret
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config.MAX_TARGET_LENGTH,
                    min_length=self.config.MIN_TARGET_LENGTH,
                    num_beams=self.config.NUM_BEAMS,
                    do_sample=self.config.DO_SAMPLE,
                    temperature=self.config.TEMPERATURE,
                    top_p=self.config.TOP_P,
                    no_repeat_ngram_size=self.config.NO_REPEAT_NGRAM_SIZE,
                    early_stopping=True
                )
            
            # Decode et
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary.strip()
            
        except Exception as e:
            self.logger.error(f"Özet üretme hatası: {str(e)}")
            return ""
    
    def evaluate_dataset(self, dataset: Dataset, num_samples=None):
        """Veri seti üzerinde değerlendirme yap"""
        if num_samples is None:
            num_samples = len(dataset)
        else:
            num_samples = min(num_samples, len(dataset))
        
        self.logger.info(f"Değerlendirme başlıyor: {num_samples} örnek")
        
        # Sonuçları sakla
        results = {
            'predictions': [],
            'references': [],
            'inputs': [],
            'rouge_scores': []
        }
        
        # Her örnek için değerlendirme
        for i in tqdm(range(num_samples), desc="Değerlendirme"):
            try:
                # Örneği al
                sample = dataset[i]
                
                # Giriş metnini decode et
                input_text = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
                if self.config.MODEL_TYPE == "t5":
                    input_text = input_text.replace("summarize: ", "")
                
                # Hedef özeti decode et
                reference = self.tokenizer.decode(sample['labels'], skip_special_tokens=True)
                
                # Özet üret
                prediction = self.generate_summary(input_text)
                
                # ROUGE skorlarını hesapla
                rouge_scores = self.rouge_scorer.score(reference, prediction)
                
                # Sonuçları sakla
                results['inputs'].append(input_text)
                results['references'].append(reference)
                results['predictions'].append(prediction)
                results['rouge_scores'].append({
                    metric: {
                        'precision': score.precision,
                        'recall': score.recall,
                        'fmeasure': score.fmeasure
                    }
                    for metric, score in rouge_scores.items()
                })
                
            except Exception as e:
                self.logger.warning(f"Örnek {i} değerlendirme hatası: {str(e)}")
                continue
        
        self.logger.info("Değerlendirme tamamlandı!")
        
        # Ortalama skorları hesapla
        avg_scores = self._calculate_average_scores(results['rouge_scores'])
        
        # Sonuçları kaydet
        evaluation_results = {
            'average_scores': avg_scores,
            'detailed_results': results,
            'num_samples': len(results['predictions']),
            'config': {
                'model_name': self.config.MODEL_NAME,
                'max_input_length': self.config.MAX_INPUT_LENGTH,
                'max_target_length': self.config.MAX_TARGET_LENGTH,
                'num_beams': self.config.NUM_BEAMS
            }
        }
        
        return evaluation_results
    
    def _calculate_average_scores(self, rouge_scores: List[Dict]) -> Dict:
        """Ortalama ROUGE skorlarını hesapla"""
        if not rouge_scores:
            return {}
        
        # Her metrik için skorları topla
        metrics_sum = {}
        for scores in rouge_scores:
            for metric, values in scores.items():
                if metric not in metrics_sum:
                    metrics_sum[metric] = {'precision': [], 'recall': [], 'fmeasure': []}
                
                metrics_sum[metric]['precision'].append(values['precision'])
                metrics_sum[metric]['recall'].append(values['recall'])
                metrics_sum[metric]['fmeasure'].append(values['fmeasure'])
        
        # Ortalamaları hesapla
        avg_scores = {}
        for metric, values in metrics_sum.items():
            avg_scores[metric] = {
                'precision': np.mean(values['precision']),
                'recall': np.mean(values['recall']),
                'fmeasure': np.mean(values['fmeasure'])
            }
        
        return avg_scores
    
    def save_results(self, results: Dict, filename="evaluation_results.json"):
        """Sonuçları kaydet"""
        try:
            results_path = os.path.join(self.config.EVALUATION_DIR, filename)
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Sonuçlar kaydedildi: {results_path}")
            
            # Özet rapor oluştur
            self._create_summary_report(results)
            
            # Grafikler oluştur
            self._create_visualizations(results)
            
        except Exception as e:
            self.logger.error(f"Sonuç kaydetme hatası: {str(e)}")
    
    def _create_summary_report(self, results: Dict):
        """Özet rapor oluştur"""
        try:
            report_lines = []
            report_lines.append("# DEĞERLENDIRME RAPORU")
            report_lines.append("=" * 50)
            report_lines.append("")
            
            # Genel bilgiler
            report_lines.append(f"Model: {results['config']['model_name']}")
            report_lines.append(f"Değerlendirilen örnek sayısı: {results['num_samples']}")
            report_lines.append("")
            
            # ROUGE skorları
            report_lines.append("## ROUGE Skorları (F1-Score)")
            report_lines.append("-" * 30)
            
            avg_scores = results['average_scores']
            for metric in self.config.ROUGE_METRICS:
                if metric in avg_scores:
                    f1_score = avg_scores[metric]['fmeasure']
                    report_lines.append(f"{metric.upper()}: {f1_score:.4f}")
            
            report_lines.append("")
            
            # En iyi ve en kötü performans
            detailed_results = results['detailed_results']
            rouge_l_scores = [scores['rougeL']['fmeasure'] for scores in detailed_results['rouge_scores']]
            
            best_idx = np.argmax(rouge_l_scores)
            worst_idx = np.argmin(rouge_l_scores)
            
            report_lines.append("## En İyi Örnek")
            report_lines.append(f"ROUGE-L F1: {rouge_l_scores[best_idx]:.4f}")
            report_lines.append(f"Referans: {detailed_results['references'][best_idx][:100]}...")
            report_lines.append(f"Üretilen: {detailed_results['predictions'][best_idx][:100]}...")
            report_lines.append("")
            
            report_lines.append("## En Kötü Örnek")
            report_lines.append(f"ROUGE-L F1: {rouge_l_scores[worst_idx]:.4f}")
            report_lines.append(f"Referans: {detailed_results['references'][worst_idx][:100]}...")
            report_lines.append(f"Üretilen: {detailed_results['predictions'][worst_idx][:100]}...")
            
            # Raporu kaydet
            report_path = os.path.join(self.config.EVALUATION_DIR, "summary_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            self.logger.info(f"Özet rapor kaydedildi: {report_path}")
            
        except Exception as e:
            self.logger.warning(f"Rapor oluşturma hatası: {str(e)}")
    
    def _create_visualizations(self, results: Dict):
        """Görselleştirmeler oluştur"""
        try:
            # ROUGE skorları bar chart
            avg_scores = results['average_scores']
            
            plt.figure(figsize=(10, 6))
            
            metrics = list(avg_scores.keys())
            f1_scores = [avg_scores[metric]['fmeasure'] for metric in metrics]
            precision_scores = [avg_scores[metric]['precision'] for metric in metrics]
            recall_scores = [avg_scores[metric]['recall'] for metric in metrics]
            
            x = np.arange(len(metrics))
            width = 0.25
            
            plt.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
            plt.bar(x, recall_scores, width, label='Recall', alpha=0.8)
            plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
            
            plt.xlabel('ROUGE Metrikleri')
            plt.ylabel('Skor')
            plt.title('ROUGE Skorları Karşılaştırması')
            plt.xticks(x, [m.upper() for m in metrics])
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Kaydet
            plt.tight_layout()
            plot_path = os.path.join(self.config.EVALUATION_DIR, "rouge_scores.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"ROUGE grafikleri kaydedildi: {plot_path}")
            
        except Exception as e:
            self.logger.warning(f"Görselleştirme hatası: {str(e)}")
    
    def print_sample_results(self, results: Dict, num_samples=5):
        """Örnek sonuçları yazdır"""
        detailed_results = results['detailed_results']
        
        print(f"\n=== ÖRNEK SONUÇLAR ({num_samples} adet) ===")
        
        for i in range(min(num_samples, len(detailed_results['predictions']))):
            rouge_l_score = detailed_results['rouge_scores'][i]['rougeL']['fmeasure']
            
            print(f"\nÖrnek {i+1} (ROUGE-L F1: {rouge_l_score:.4f})")
            print("-" * 50)
            print(f"GİRİŞ: {detailed_results['inputs'][i][:200]}...")
            print(f"\nREFERANS: {detailed_results['references'][i]}")
            print(f"\nÜRETİLEN: {detailed_results['predictions'][i]}")
            print("=" * 70)

def main():
    """Ana fonksiyon"""
    # Konfigürasyonu yükle
    config = Config()
    config.create_dirs()
    config.print_config()
    
    # Veri ön işleyiciyi başlat
    preprocessor = DataPreprocessor(config)
    
    # Test verisini yükle
    _, _, test_dataset = preprocessor.load_processed_data()
    
    # Model değerlendiricisini başlat
    evaluator = ModelEvaluator(config)
    
    # Modeli yükle
    evaluator.load_model()
    
    # Değerlendirme yap (hızlı test için 100 örnek)
    results = evaluator.evaluate_dataset(test_dataset, num_samples=100)
    
    # Sonuçları kaydet
    evaluator.save_results(results)
    
    # Örnek sonuçları göster
    evaluator.print_sample_results(results, num_samples=5)
    
    # Ortalama skorları yazdır
    print("\n=== ORTALAMA ROUGE SKORLARI ===")
    avg_scores = results['average_scores']
    for metric in config.ROUGE_METRICS:
        if metric in avg_scores:
            f1_score = avg_scores[metric]['fmeasure']
            print(f"{metric.upper()} F1-Score: {f1_score:.4f}")

if __name__ == "__main__":
    main()