"""
Prediction Pipeline - Orchestrates the entire inference workflow
Entry point for end-to-end predictions
"""

from typing import Union, List, Dict
from src.components.model_loader import ModelLoader
from src.components.predictor import HallucinationPredictor
from src.components.validator import InputValidator
from src.config.config import config
from src.exception import CustomException
from src.logger import logging
import sys


class PredictionPipeline:
    """
    End-to-end prediction pipeline
    
    Handles:
    1. Model/tokenizer loading (with caching)
    2. Input validation
    3. Batch processing
    4. Result formatting
    """
    
    def __init__(self, threshold: float = config.DEFAULT_THRESHOLD):
        """
        Initialize pipeline
        
        Args:
            threshold: Confidence threshold for predictions
        """
        try:
            logging.info("Initializing Prediction Pipeline...")
            
            # Validate config
            config.validate_paths()
            
            # Load model and tokenizer (cached)
            model, tokenizer = ModelLoader.load_model_and_tokenizer()
            
            # Initialize predictor
            self.predictor = HallucinationPredictor(
                model=model,
                tokenizer=tokenizer,
                threshold=threshold
            )
            
            logging.info("Prediction Pipeline ready")
            
        except Exception as e:
            raise CustomException(f"Pipeline initialization failed: {str(e)}", sys)
    
    def predict(self, texts: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Run prediction
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Prediction result(s)
        """
        try:
            result = self.predictor.predict(texts)
            return result
        except Exception as e:
            raise CustomException(f"Prediction failed: {str(e)}", sys)
    
    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """
        Run batch prediction
        
        Args:
            texts: List of texts
            
        Returns:
            List of predictions
        """
        return self.predict(texts)
    
    def set_threshold(self, threshold: float):
        """Update confidence threshold"""
        self.predictor.set_threshold(threshold)
    
    def get_summary(self, results: Union[Dict, List[Dict]]) -> Dict:
        """
        Generate summary statistics from predictions
        
        Args:
            results: Single result or list of results
            
        Returns:
            Summary dict with statistics
        """
        try:
            if isinstance(results, dict):
                results = [results]
            
            if not results:
                raise ValueError("Empty results")
            
            total = len(results)
            hallucinations = sum(1 for r in results if r.get("is_hallucination"))
            truths = total - hallucinations
            
            avg_confidence = sum(r.get("confidence", 0) for r in results) / total
            
            summary = {
                "total_samples": total,
                "hallucinations": hallucinations,
                "truths": truths,
                "hallucination_rate": round(hallucinations / total, 4),
                "avg_confidence": round(avg_confidence, 4),
                "threshold_used": self.predictor.threshold
            }
            
            return summary
            
        except Exception as e:
            raise CustomException(f"Summary generation failed: {str(e)}", sys)
