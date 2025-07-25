"""
Çıkarım (Inference) modülü
Eğitilmiş model ile yeni metinleri özetleme
"""

import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import Config
import json
from datetime import datetime

class SummaryGenerator:
    def __init__(self, config: Config, model_path=None):
        self.config = config
        if model_path is None:
            model_path = os.path.join(config.MODEL_DIR, "final_model")
        self.model_path = os.path.normpath(model_path)
        self.tokenizer = None
        self.model = None
        
        self._setup_logging()
        self.load_model()
        
    def _setup_logging(self):
        """Logging kurulumu"""
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.LOG_DIR, 'inference.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """Load trained model - Use pre-trained if not available"""
        # Normalize path
        model_path = os.path.normpath(self.model_path)
        
        try:
            # Try local model first
            if os.path.exists(model_path) and os.path.isdir(model_path):
                self.logger.info(f"Loading local model: {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            else:
                # Use pre-trained model if local model not found
                self.logger.info(f"Local model not found, using pre-trained model: {self.config.MODEL_NAME}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.MODEL_NAME)
            
            if self.config.DEVICE == "cuda":
                self.model = self.model.to(self.config.DEVICE)
                self.logger.info("Model loaded on GPU")
            else:
                self.logger.info("Model running on CPU")
            
            self.model.eval()  # Set to evaluation mode
            
            self.logger.info("Model loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"Model loading error: {str(e)}")
            raise
    
    def generate_summary(self, text: str, custom_params: dict = None) -> dict:
        """
        Verilen metin için özet üret
        
        Args:
            text (str): Özetlenecek metin
            custom_params (dict): Özel üretim parametreleri
            
        Returns:
            dict: Özet ve meta bilgiler
        """
        try:
            # Empty text check
            if not text or not text.strip():
                return {
                    'summary': '',
                    'error': 'Empty text',
                    'input_length': 0,
                    'summary_length': 0
                }
            
            # Metni temizle
            text = text.strip()
            
            # T5 için prefix ekle
            input_text = text
            if self.config.MODEL_TYPE == "t5":
                input_text = "summarize: " + text
            
            # Tokenize et
            inputs = self.tokenizer(
                input_text,
                max_length=self.config.MAX_INPUT_LENGTH,
                truncation=True,
                return_tensors="pt"
            )
            
            if self.config.DEVICE == "cuda":
                inputs = {k: v.to(self.config.DEVICE) for k, v in inputs.items()}
            
            # Üretim parametrelerini hazırla
            generation_params = {
                'max_length': self.config.MAX_TARGET_LENGTH,
                'min_length': self.config.MIN_TARGET_LENGTH,
                'num_beams': self.config.NUM_BEAMS,
                'do_sample': self.config.DO_SAMPLE,
                'temperature': self.config.TEMPERATURE,
                'top_p': self.config.TOP_P,
                'no_repeat_ngram_size': self.config.NO_REPEAT_NGRAM_SIZE,
                'early_stopping': True,
                'pad_token_id': self.tokenizer.pad_token_id
            }
            
            # Özel parametreler varsa güncelle
            if custom_params:
                generation_params.update(custom_params)
            
            # Özet üret
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_params
                )
            
            # Decode et
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = summary.strip()
            
            # Sonuç hazırla
            result = {
                'summary': summary,
                'input_text': text,
                'input_length': len(text),
                'input_tokens': len(inputs['input_ids'][0]),
                'summary_length': len(summary),
                'summary_tokens': len(outputs[0]),
                'generation_params': generation_params,
                'timestamp': datetime.now().isoformat(),
                'model_name': self.config.MODEL_NAME,
                'error': None
            }
            
            self.logger.info(f"Summary generated: {len(text)} -> {len(summary)} characters")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Özet üretme hatası: {str(e)}")
            return {
                'summary': '',
                'error': str(e),
                'input_length': len(text) if text else 0,
                'summary_length': 0
            }
    
    def batch_summarize(self, texts: list, custom_params: dict = None) -> list:
        """
        Birden fazla metin için toplu özet üretme
        
        Args:
            texts (list): Özetlenecek metinler listesi
            custom_params (dict): Özel üretim parametreleri
            
        Returns:
            list: Özet sonuçları listesi
        """
        results = []
        
        self.logger.info(f"Starting batch summarization: {len(texts)} texts")
        
        for i, text in enumerate(texts):
            self.logger.info(f"Processing text {i+1}/{len(texts)}...")
            result = self.generate_summary(text, custom_params)
            result['batch_index'] = i
            results.append(result)
        
        self.logger.info("Batch summarization completed!")
        
        return results
    
    def summarize_from_file(self, file_path: str, custom_params: dict = None) -> dict:
        """
        Dosyadan metin okuyup özetleme
        
        Args:
            file_path (str): Metin dosyası yolu
            custom_params (dict): Özel üretim parametreleri
            
        Returns:
            dict: Özet sonucu
        """
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            self.logger.info(f"File read: {file_path}")
            
            # Perform summarization
            result = self.generate_summary(text, custom_params)
            result['source_file'] = file_path
            
            return result
            
        except Exception as e:
            self.logger.error(f"File reading error: {str(e)}")
            return {
                'summary': '',
                'error': f"File reading error: {str(e)}",
                'source_file': file_path
            }
    
    def save_result(self, result: dict, output_file: str = None):
        """Save result to file"""
        try:
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"summary_{timestamp}.json"
            
            output_path = os.path.join(self.config.PREDICTIONS_DIR, output_file)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Result saved: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Result saving error: {str(e)}")
    
    def interactive_summarization(self):
        """Interactive summarization mode"""
        print("=== INTERACTIVE SUMMARIZATION MODE ===")
        print("Type 'quit' to exit")
        print("Type 'help' for help")
        print("-" * 40)
        
        while True:
            try:
                print("\nPlease enter the text you want to summarize:")
                user_input = input("> ").strip()
                
                if user_input.lower() == 'quit':
                    print("Exiting...")
                    break
                elif user_input.lower() == 'help':
                    self._print_help()
                    continue
                elif not user_input:
                    print("Empty text entered, please try again.")
                    continue
                
                print("\nGenerating summary...")
                result = self.generate_summary(user_input)
                
                if result['error']:
                    print(f"Error: {result['error']}")
                else:
                    print(f"\n--- SUMMARY ---")
                    print(result['summary'])
                    print(f"\nInfo: {result['input_length']} -> {result['summary_length']} characters")
                    
                    # Save?
                    save_choice = input("\nWould you like to save the result? (y/n): ").lower()
                    if save_choice == 'y':
                        self.save_result(result)
                        print("Result saved!")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def _print_help(self):
        """Print help message"""
        help_text = """
        === HELP ===
        
        Commands:
        - quit: Exit the program
        - help: Show this help message
        
        Usage:
        1. Enter the text you want to summarize
        2. Press Enter
        3. The program will generate a summary
        4. Optionally save the result
        
        Note: Long texts will be automatically truncated.
        """
        print(help_text)

def main():
    """Main function - Demo usage"""
    # Load configuration
    config = Config()
    config.create_dirs()
    config.print_config()
    
    # Start summarizer
    summarizer = SummaryGenerator(config)
    
    # Demo text
    demo_text = """
    Artificial intelligence technology has been rapidly developing in recent years and has 
    started to be used in many areas of our lives. Especially developments in natural language 
    processing enable computers to better understand and generate human language. Thanks to these 
    technologies, applications such as automatic translation, text summarization, and question 
    answering have become possible. The Transformer architecture was an important turning point 
    in this field and great successes were achieved with models like GPT and BERT. However, 
    the ethical use of these technologies and their social impacts have also become important issues.
    """
    
    print("=== DEMO SUMMARIZATION ===")
    print("Summarizing demo text...")
    
    # Generate demo summary
    result = summarizer.generate_summary(demo_text)
    
    if result['error']:
        print(f"Error: {result['error']}")
    else:
        print(f"\nORIGINAL TEXT ({result['input_length']} characters):")
        print(demo_text)
        print(f"\nGENERATED SUMMARY ({result['summary_length']} characters):")
        print(result['summary'])
        
        # Save result
        summarizer.save_result(result, "demo_summary.json")
        print(f"\nDemo result saved!")
    
    # Start interactive mode?
    choice = input("\nWould you like to enter interactive mode? (y/n): ").lower()
    if choice == 'y':
        summarizer.interactive_summarization()

if __name__ == "__main__":
    main()