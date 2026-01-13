"""
Predictor component
Handles inference and confidence scoring
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config.config import config
from src.components.validator import InputValidator
from src.components.model_loader import ModelLoader
from src.exception import CustomException
from src.logger import logging
import sys


class HallucinationPredictor:
    """Inference engine for hallucination detection"""
    
    def __init__(
        self,
        model: AutoModelForSequenceClassification = None,
        tokenizer: AutoTokenizer = None,
        threshold: float = config.DEFAULT_THRESHOLD
    ):
        """
        Initialize predictor
        
        Args:
            model: Pretrained model (loads if None)
            tokenizer: Tokenizer (loads if None)
            threshold: Confidence threshold for predictions
        """
        self.model = model or ModelLoader.load_model()
        self.tokenizer = tokenizer or ModelLoader.load_tokenizer()
        self.device = ModelLoader.get_device()
        self.threshold = InputValidator.validate_threshold(threshold)
        self.label_names = config.LABEL_NAMES
        
        logging.info(f"Predictor initialized with threshold: {self.threshold}")
    
    def predict_single(self, text: str) -> Dict[str, Union[str, float, int]]:
        """
        Predict on single text
        
        Args:
            text: Input text
            
        Returns:
            Dict with prediction, confidence, label_id
            
        Example:
            {
                "text": "Paris is the capital of France",
                "label": "Truth",
                "confidence": 0.98,
                "label_id": 0
            }
        """
        try:
            # Validate input
            text = InputValidator.validate_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=config.MAX_LENGTH
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Extract results
            confidence, pred_id = torch.max(probs, dim=-1)
            confidence = confidence.item()
            pred_id = pred_id.item()
            label = self.label_names[pred_id]
            
            result = {
                "text": text,
                "label": label,
                "label_id": pred_id,
                "confidence": round(confidence, 4),
                "is_hallucination": pred_id == 1,
                "meets_threshold": confidence >= self.threshold
            }
            
            logging.debug(f"Prediction: {label} ({confidence:.4f})")
            return result
            
        except Exception as e:
            raise CustomException(f"Single prediction failed: {str(e)}", sys)
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict on batch of texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of predictions
        """
        try:
            # Validate inputs
            texts = InputValidator.validate_texts(texts)
            
            predictions = []
            
            # Process in batches
            for i in range(0, len(texts), config.BATCH_SIZE):
                batch = texts[i:i + config.BATCH_SIZE]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=config.MAX_LENGTH
                ).to(self.device)
                
                # Inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Extract results
                confidences, pred_ids = torch.max(probs, dim=-1)
                
                for j, text in enumerate(batch):
                    result = {
                        "text": text,
                        "label": self.label_names[pred_ids[j].item()],
                        "label_id": pred_ids[j].item(),
                        "confidence": round(confidences[j].item(), 4),
                        "is_hallucination": pred_ids[j].item() == 1,
                        "meets_threshold": confidences[j].item() >= self.threshold
                    }
                    predictions.append(result)
            
            logging.info(f"Batch prediction complete: {len(predictions)} samples")
            return predictions
            
        except Exception as e:
            raise CustomException(f"Batch prediction failed: {str(e)}", sys)
    
    def predict(
        self,
        texts: Union[str, List[str]]
    ) -> Union[Dict, List[Dict]]:
        """
        Unified predict interface (handles single or batch)
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Single prediction dict or list of dicts
        """
        if isinstance(texts, str):
            return self.predict_single(texts)
        elif isinstance(texts, list):
            return self.predict_batch(texts)
        else:
            raise TypeError(f"Expected str or list, got {type(texts).__name__}")
    
    def set_threshold(self, threshold: float):
        """Update confidence threshold"""
        self.threshold = InputValidator.validate_threshold(threshold)
        logging.info(f"Threshold updated to {self.threshold}")
