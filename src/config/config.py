

import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Central configuration class"""
    
    #  PATHS 
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    NOTEBOOK_DIR = PROJECT_ROOT / "notebook"
    
    # Model paths (fallback for local)
    MODEL_INPUT_PATH = NOTEBOOK_DIR / "hallucination_model_v1"
    MODEL_OUTPUT_PATH = NOTEBOOK_DIR / "modernbert_final"
    TEMP_DIR = NOTEBOOK_DIR / "temp"
    
    # Which model to use: "hf" for Hugging Face Hub
    ACTIVE_MODEL = "hf"
    
    # Hugging Face Hub Model ID (Primary - use this for cloud deployment)
    HF_MODEL_ID = "baguestto/hallucination-model-v1"  # This one is correct (10.9% local test)
    
    #  MODEL SETTINGS
    MODEL_NAME = "ModernBERT-Hallucination-Detector"
    MODEL_TYPE = "sequence-classification"
    NUM_LABELS = 2
    LABEL_NAMES = ["Truth", "Hallucination"]
    
    #  INFERENCE SETTINGS 
    DEFAULT_THRESHOLD = 0.75
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    
    #  DEVICE SETTINGS 
    USE_GPU = True  # Will auto-detect at runtime
    
    #  LOGGING 
    LOG_DIR = PROJECT_ROOT / "logs"
    LOG_LEVEL = "INFO"
    
    @classmethod
    def validate_paths(cls) -> bool:
        """Validate that required directories exist"""
        # Only require NOTEBOOK_DIR to exist
        # Models are loaded from Hugging Face Hub by default
        required_paths = [
            cls.NOTEBOOK_DIR,
        ]
        
        missing = [p for p in required_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing required paths: {missing}")
        
        # Create directories if they don't exist
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        return True
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export config as dictionary"""
        return {
            "model_name": cls.MODEL_NAME,
            "model_type": cls.MODEL_TYPE,
            "num_labels": cls.NUM_LABELS,
            "label_names": cls.LABEL_NAMES,
            "max_length": cls.MAX_LENGTH,
            "batch_size": cls.BATCH_SIZE,
            "default_threshold": cls.DEFAULT_THRESHOLD,
        }


# Create global config instance
config = Config()
