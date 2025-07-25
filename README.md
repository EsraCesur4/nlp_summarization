# NLP Haber Özetleme Projesi

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.21+-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Transformer tabanlı modeller kullanarak CNN/DailyMail veri seti üzerinde otomatik haber özetleme sistemi.

## 📋 Proje Özeti

Bu proje, T5-small modeli kullanarak İngilizce haber metinlerini otomatik olarak özetleyen bir sistem geliştirmektedir. Hugging Face Transformers kütüphanesi ile CNN/DailyMail veri seti üzerinde fine-tuning yapılarak ROUGE metrikleri ile değerlendirilmektedir.

### 🎯 Hedefler

- **Model**: T5-small (alternatif: facebook/bart-base)
- **Veri Seti**: CNN/DailyMail (3.0.0)
- **Görev**: Article alanından summary alanını tahmin etme
- **Değerlendirme**: ROUGE-1, ROUGE-2, ROUGE-L metrikleri

## 🚀 Hızlı Başlangıç

### Gereksinimler

- Python 3.8+
- CUDA (GPU kullanımı için, opsiyonel)
- 8GB+ RAM (16GB önerilir)
- 5GB disk alanı

### Kurulum

1. **Projeyi klonlayın veya indirin**
```bash
git clone <repository-url>
cd nlp_ozetleme_projesi
```

2. **Sanal ortam oluşturun**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Gerekli kütüphaneleri yükleyin**
```bash
pip install -r requirements.txt
```

4. **Klasör yapısını oluşturun**
```bash
# Windows
mkdir data data\processed data\raw models models\checkpoints outputs outputs\predictions outputs\evaluation_results logs

# macOS/Linux
mkdir -p data/{processed,raw} models/checkpoints outputs/{predictions,evaluation_results} logs
```

### 🔧 Kullanım

#### 1. Tam Pipeline (Önerilen)
```bash
python main.py --mode full
```

#### 2. Adım Adım Çalıştırma

**Veri Ön İşleme:**
```bash
python main.py --mode preprocess
```

**Model Eğitimi:**
```bash
python main.py --mode train
```

**Değerlendirme:**
```bash
python main.py --mode evaluate
```

**Demo:**
```bash
python main.py --mode demo
```

#### 3. Jupyter Notebook
```bash
jupyter notebook ozetleme_projesi.ipynb
```

#### 4. Etkileşimli Özetleme
```bash
python inference.py
```

## 📊 Model Performansı

Tipik ROUGE skorları (test veri seti üzerinde):

| Metrik | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| ROUGE-1 | 0.45 | 0.42 | 0.43 |
| ROUGE-2 | 0.20 | 0.18 | 0.19 |
| ROUGE-L | 0.35 | 0.33 | 0.34 |

*Not: Gerçek sonuçlar veri boyutu ve eğitim parametrelerine bağlı olarak değişebilir.*

## 📁 Proje Yapısı

```
nlp_ozetleme_projesi/
├── 📄 README.md                    # Bu dosya
├── 📄 requirements.txt             # Python gereksinimleri
├── 📄 config.py                    # Konfigürasyon ayarları
├── 📄 data_preprocessing.py        # Veri ön işleme
├── 📄 model_training.py           # Model eğitimi
├── 📄 evaluation.py               # Model değerlendirme
├── 📄 inference.py                # Çıkarım ve demo
├── 📄 main.py                     # Ana koordinatör
├── 📓 ozetleme_projesi.ipynb      # Jupyter notebook
├── 📄 rapor.md                    # Proje raporu
│
├── 📁 data/                       # Veri dosyaları
│   ├── 📁 processed/              # İşlenmiş veriler
│   └── 📁 raw/                    # Ham veriler
│
├── 📁 models/                     # Eğitilmiş modeller
│   └── 📁 checkpoints/            # Model checkpoint'leri
│
├── 📁 outputs/                    # Çıktılar
│   ├── 📁 predictions/            # Tahmin sonuçları
│   └── 📁 evaluation_results/     # Değerlendirme sonuçları
│
└── 📁 logs/                       # Log dosyaları
```

## ⚙️ Konfigürasyon

`config.py` dosyasında önemli parametreler:

```python
# Model ayarları
MODEL_NAME = "t5-small"  # Alternatif: "facebook/bart-base"
MAX_INPUT_LENGTH = 512    # Giriş metni uzunluğu
MAX_TARGET_LENGTH = 128   # Özet uzunluğu

# Eğitim ayarları
BATCH_SIZE = 4           # Bellek kullanımına göre ayarlayın
NUM_EPOCHS = 3           # Hızlı prototip için
LEARNING_RATE = 5e-5

# Veri boyutu (hızlı test için küçük)
TRAIN_SIZE = 5000
VAL_SIZE = 1000
TEST_SIZE = 1000
```

## 📈 Sonuç Dosyaları

Proje çalıştırıldıktan sonra oluşan önemli dosyalar:

- `models/final_model/` - Eğitilmiş model
- `outputs/evaluation_results/evaluation_results.json` - ROUGE skorları
- `outputs/evaluation_results/summary_report.txt` - Özet rapor
- `outputs/training_history.json` - Eğitim geçmişi
- `outputs/training_curves.png` - Eğitim grafikleri
- `logs/` - Tüm işlem logları

## 🔍 Örnek Kullanım

### Python Script ile
```python
from inference import SummaryGenerator
from config import Config

# Özet üreticiyi başlat
config = Config()
summarizer = SummaryGenerator(config)

# Metin özetle
text = "Your long article text here..."
result = summarizer.generate_summary(text)
print(result['summary'])
```

### Komut Satırından
```bash
# Demo modunda çalıştır
python main.py --mode demo

# Özel metin dosyası özetle
python inference.py
```

## 🛠️ Sorun Giderme

### Yaygın Hatalar

**1. CUDA Hatası:**
```bash
# CPU kullanımına zorla
export CUDA_VISIBLE_DEVICES=""
python main.py
```

**2. Bellek Hatası:**
```python
# config.py içinde
BATCH_SIZE = 2  # Daha küçük batch size
TRAIN_SIZE = 1000  # Daha az veri
```

**3. Model İndirme Hatası:**
```bash
# Hugging Face cache temizle
rm -rf ~/.cache/huggingface/
```

### Performans İpuçları

- **GPU Kullanımı**: CUDA kurulu ise otomatik GPU kullanılır
- **Bellek Optimizasyonu**: Batch size'ı düşürün
- **Hız Optimizasyonu**: FP16 precision kullanın (GPU'da)
- **Veri Boyutu**: İlk testler için küçük veri seti kullanın

## 📚 Ek Kaynaklar

### Model Mimarileri
- [T5 Paper](https://arxiv.org/abs/1910.10683) - "Exploring the Limits of Transfer Learning"
- [BART Paper](https://arxiv.org/abs/1910.13461) - "Denoising Sequence-to-Sequence Pre-training"

### Veri Seti
- [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) - Hugging Face Dataset

### Değerlendirme
- [ROUGE Metrics](https://aclanthology.org/W04-1013/) - "ROUGE: A Package for Automatic Evaluation"

### Kütüphaneler
- [🤗 Transformers](https://huggingface.co/transformers/) - Hugging Face Transformers
- [PyTorch](https://pytorch.org/) - Deep Learning Framework
- [Datasets](https://huggingface.co/docs/datasets/) - Hugging Face Datasets

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 📞 İletişim

Sorularınız için:
- Issue açın
- E-posta gönderin
- Pull request gönderin

## 🙏 Teşekkürler

- Hugging Face ekibine transformers kütüphanesi için
- CNN ve Daily Mail veri seti sağlayıcılarına
- PyTorch topluluğuna

---

⭐ Proje yararlıysa yıldız vermeyi unutmayın!