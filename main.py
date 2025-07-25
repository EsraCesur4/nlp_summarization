"""
Ana script - Tüm süreçleri yöneten merkezi dosya
NLP Haber Özetleme Projesi
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import torch

# Proje modüllerini import et
from config import Config
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from evaluation import ModelEvaluator
from inference import SummaryGenerator

class NLPSummarizationPipeline:
    def __init__(self, config: Config):
        self.config = config
        self._setup_logging()
        
    def _setup_logging(self):
        """Ana logging kurulumu"""
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.LOG_DIR, 'main.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_environment(self):
        """Setup working environment"""
        self.logger.info("Setting up working environment...")
        
        # Create directories
        self.config.create_dirs()
        
        # Print configuration
        self.config.print_config()
        
        # CUDA info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            self.logger.info("No GPU found, using CPU")
        
        self.logger.info("Working environment ready!")
        
    def run_data_preprocessing(self):
        """Run data preprocessing step"""
        self.logger.info("=== DATA PREPROCESSING STARTING ===")
        
        try:
            preprocessor = DataPreprocessor(self.config)
            train_dataset, val_dataset, test_dataset = preprocessor.load_processed_data()
            
            # Show sample data
            samples = preprocessor.get_sample_data(train_dataset, 2)
            
            self.logger.info("Data preprocessing completed successfully!")
            self.logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            self.logger.error(f"Data preprocessing error: {str(e)}")
            raise
    
    def run_training(self, train_dataset, val_dataset):
        """Run model training step"""
        self.logger.info("=== MODEL TRAINING STARTING ===")
        
        try:
            trainer = ModelTrainer(self.config)
            train_result = trainer.train_model(train_dataset, val_dataset)
            
            self.logger.info("Model training completed successfully!")
            self.logger.info(f"Final train loss: {train_result.metrics['train_loss']:.4f}")
            
            return train_result
            
        except Exception as e:
            self.logger.error(f"Model training error: {str(e)}")
            raise
    
    def run_evaluation(self, test_dataset):
        """Run model evaluation step"""
        self.logger.info("=== MODEL EVALUATION STARTING ===")
        
        try:
            evaluator = ModelEvaluator(self.config)
            evaluator.load_model()
            
            # Quick evaluation with 100 samples
            results = evaluator.evaluate_dataset(test_dataset, num_samples=100)
            
            # Save results
            evaluator.save_results(results)
            
            # Print average scores
            avg_scores = results['average_scores']
            self.logger.info("=== ROUGE SCORES ===")
            for metric in self.config.ROUGE_METRICS:
                if metric in avg_scores:
                    f1_score = avg_scores[metric]['fmeasure']
                    self.logger.info(f"{metric.upper()} F1-Score: {f1_score:.4f}")
            
            self.logger.info("Model evaluation completed successfully!")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Model evaluation error: {str(e)}")
            raise
    
    def run_demo_inference(self):
        """Run demo inference step"""
        self.logger.info("=== DEMO INFERENCE STARTING ===")
        
        try:
            summarizer = SummaryGenerator(self.config)
            
            # Demo texts
            demo_texts = [
                """
                Artificial intelligence has made remarkable progress in recent years, particularly in 
                natural language processing. Large language models like GPT and T5 have revolutionized 
                how computers understand and generate human language. These models are trained on massive 
                datasets and use transformer architectures to achieve state-of-the-art performance on 
                various tasks including text summarization, question answering, and machine translation. 
                However, concerns about bias, privacy, and the environmental impact of training these 
                large models have also emerged. Researchers are now focusing on developing more efficient 
                and responsible AI systems.
                """,
                """
                Climate change continues to be one of the most pressing global challenges of our time. 
                Rising global temperatures, melting ice caps, and extreme weather events are becoming 
                increasingly frequent. Scientists warn that without immediate action to reduce greenhouse 
                gas emissions, the consequences could be catastrophic. Renewable energy sources like 
                solar and wind power are becoming more cost-effective and widely adopted. Many countries 
                have committed to achieving net-zero emissions by 2050, but experts argue that more 
                aggressive measures are needed.
                """,
                """
                The global economy has shown signs of recovery following the challenges posed by the 
                COVID-19 pandemic. Supply chain disruptions that affected industries worldwide are 
                gradually being resolved. Inflation rates have been a concern for many central banks, 
                leading to adjustments in monetary policy. The technology sector has remained resilient, 
                with increased demand for digital services. Meanwhile, emerging markets are experiencing 
                varied recovery patterns, with some showing robust growth while others struggle with 
                debt and political instability.
                """
            ]
            
            # Generate summaries for each demo text
            demo_results = []
            for i, text in enumerate(demo_texts, 1):
                self.logger.info(f"Summarizing demo {i}...")
                result = summarizer.generate_summary(text.strip())
                result['demo_id'] = i
                demo_results.append(result)
                
                # Show result
                print(f"\n=== DEMO {i} ===")
                print(f"ORIGINAL ({result['input_length']} chars):")
                print(text.strip()[:200] + "..." if len(text.strip()) > 200 else text.strip())
                print(f"\nSUMMARY ({result['summary_length']} chars):")
                print(result['summary'])
                print("-" * 60)
            
            # Save demo results
            import json
            demo_path = os.path.join(self.config.PREDICTIONS_DIR, "demo_results.json")
            with open(demo_path, 'w', encoding='utf-8') as f:
                json.dump(demo_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Demo results saved: {demo_path}")
            self.logger.info("Demo inference completed successfully!")
            
            return demo_results
            
        except Exception as e:
            self.logger.error(f"Demo inference error: {str(e)}")
            raise
    
    def run_full_pipeline(self):
        """Run full pipeline"""
        start_time = datetime.now()
        self.logger.info("=== FULL PIPELINE STARTING ===")
        
        try:
            # 1. Environment setup
            self.setup_environment()
            
            # 2. Data preprocessing
            train_dataset, val_dataset, test_dataset = self.run_data_preprocessing()
            
            # 3. Model training
            train_result = self.run_training(train_dataset, val_dataset)
            
            # 4. Model evaluation
            eval_results = self.run_evaluation(test_dataset)
            
            # 5. Demo inference
            demo_results = self.run_demo_inference()
            
            # Pipeline summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info("=== PIPELINE SUMMARY ===")
            self.logger.info(f"Total duration: {duration}")
            self.logger.info(f"Training loss: {train_result.metrics['train_loss']:.4f}")
            
            # Best ROUGE-L score
            rouge_l_score = eval_results['average_scores']['rougeL']['fmeasure']
            self.logger.info(f"ROUGE-L F1: {rouge_l_score:.4f}")
            
            self.logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
            
            return {
                'train_result': train_result,
                'eval_results': eval_results,
                'demo_results': demo_results,
                'duration': str(duration)
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='NLP News Summarization Project')
    parser.add_argument('--mode', choices=['full', 'preprocess', 'train', 'evaluate', 'demo'], 
                       default='full', help='Mode to run')
    parser.add_argument('--config', type=str, help='Custom configuration file')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = Config()
        
        # Start pipeline
        pipeline = NLPSummarizationPipeline(config)
        
        if args.mode == 'full':
            # Run full pipeline
            results = pipeline.run_full_pipeline()
            print("\n🎉 Pipeline completed successfully!")
            
        elif args.mode == 'preprocess':
            # Only data preprocessing
            pipeline.setup_environment()
            datasets = pipeline.run_data_preprocessing()
            print("\n✅ Data preprocessing completed!")
            
        elif args.mode == 'train':
            # Only model training
            pipeline.setup_environment()
            preprocessor = DataPreprocessor(config)
            train_dataset, val_dataset, _ = preprocessor.load_processed_data()
            train_result = pipeline.run_training(train_dataset, val_dataset)
            print("\n🚀 Model training completed!")
            
        elif args.mode == 'evaluate':
            # Only evaluation
            pipeline.setup_environment()
            preprocessor = DataPreprocessor(config)
            _, _, test_dataset = preprocessor.load_processed_data()
            eval_results = pipeline.run_evaluation(test_dataset)
            print("\n📊 Model evaluation completed!")
            
        elif args.mode == 'demo':
            # Only demo
            pipeline.setup_environment()
            demo_results = pipeline.run_demo_inference()
            print("\n✨ Demo completed!")
            
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline stopped by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()