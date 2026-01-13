"""
Model loader component
Handles loading pretrained model and tokenizer with caching
"""

import torch
from typing import Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config.config import config
from src.exception import CustomException
from src.logger import logging
import sys


class ModelLoader:
    """Loads and caches model + tokenizer"""
    
    _model_cache = None
    _tokenizer_cache = None
    _device = None
    
    @classmethod
    def get_device(cls) -> torch.device:
        """Get device (GPU or CPU) - compute once"""
        if cls._device is None:
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                logging.info(f"CUDA Version: {torch.version.cuda}")
            else:
                logging.warning("CUDA not available, falling back to CPU (inference will be slow)")
        
        return cls._device
    
    @classmethod
    def load_model(cls, model_path: Optional[str] = None) -> AutoModelForSequenceClassification:
        """
        Load pretrained model with caching
        
        Args:
            model_path: Path to model (uses config default if None)
            
        Returns:
            Loaded model
            
        Raises:
            CustomException: If model loading fails
        """
        try:
            # Return cached model if available
            if cls._model_cache is not None:
                logging.info("Using cached model")
                return cls._model_cache
            
            path = model_path or config.MODEL_INPUT_PATH
            logging.info(f"Loading model from {path}")
            
            if not path.exists():
                raise FileNotFoundError(f"Model not found at {path}")
            
            model = AutoModelForSequenceClassification.from_pretrained(
                str(path),
                num_labels=config.NUM_LABELS,
                torch_dtype=torch.float32
            )
            
            # Move to device
            device = cls.get_device()
            model = model.to(device)
            model.eval()  # Set to eval mode
            
            # Cache it
            cls._model_cache = model
            logging.info(f"Model loaded successfully on {device}")
            
            return model
            
        except Exception as e:
            raise CustomException(f"Failed to load model: {str(e)}", sys)
    
    @classmethod
    def load_tokenizer(cls, tokenizer_path: Optional[str] = None) -> AutoTokenizer:
        """
        Load tokenizer with caching
        
        Args:
            tokenizer_path: Path to tokenizer (uses config default if None)
            
        Returns:
            Loaded tokenizer
            
        Raises:
            CustomException: If tokenizer loading fails
        """
        try:
            # Return cached tokenizer if available
            if cls._tokenizer_cache is not None:
                logging.info("Using cached tokenizer")
                return cls._tokenizer_cache
            
            path = tokenizer_path or config.MODEL_INPUT_PATH
            logging.info(f"Loading tokenizer from {path}")
            
            if not path.exists():
                raise FileNotFoundError(f"Tokenizer not found at {path}")
            
            tokenizer = AutoTokenizer.from_pretrained(str(path))
            
            # Cache it
            cls._tokenizer_cache = tokenizer
            logging.info("Tokenizer loaded successfully")
            
            return tokenizer
            
        except Exception as e:
            raise CustomException(f"Failed to load tokenizer: {str(e)}", sys)
    
    @classmethod
    def load_model_and_tokenizer(cls) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Load both model and tokenizer
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model = cls.load_model()
        tokenizer = cls.load_tokenizer()
        return model, tokenizer
    
    @classmethod
    def clear_cache(cls):
        """Clear model and tokenizer cache"""
        cls._model_cache = None
        cls._tokenizer_cache = None
        logging.info("Model cache cleared")
