"""
Input validation module for hallucination detector
Provides robust validation for text inputs and datasets
"""

import re
from typing import Union, List, Dict, Any
from src.exception import CustomException
import sys


class InputValidator:
    """Validates inputs before processing"""

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text to make inference more robust to casing and noisy punctuation.

        Args:
            text: Raw input text

        Returns:
            Normalized text
        """
        # Normalize common unicode punctuation to ASCII
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2013", "-").replace("\u2014", "-")

        # Drop punctuation so casing/punctuation variants behave consistently
        text = re.sub(r"[^\w\s']", " ", text)

        # Normalize whitespace inside text
        text = re.sub(r"\s+", " ", text)

        # Lowercase for case-insensitive behavior
        text = text.strip().lower()

        # Remove common function words so phrasing variants map similarly
        stop_words = {
            "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
            "of", "to", "in", "on", "at", "by", "for", "from", "with", "and", "or", "but"
        }
        tokens = [tok for tok in text.split() if tok not in stop_words]

        # Keep a safe fallback if everything is removed by stop-word filtering
        return " ".join(tokens) if tokens else text
    
    @staticmethod
    def validate_text(text: str, min_length: int = 1, max_length: int = 5000) -> str:
        """
        Validate and clean text input
        
        Args:
            text: Input text to validate
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Cleaned text
            
        Raises:
            CustomException: If validation fails
        """
        try:
            if text is None:
                raise ValueError("Text cannot be None")
            
            if not isinstance(text, str):
                raise TypeError(f"Expected str, got {type(text).__name__}")
            
            text = InputValidator.normalize_text(text)
            
            if len(text) < min_length:
                raise ValueError(f"Text too short (min {min_length} chars, got {len(text)})")
            
            if len(text) > max_length:
                raise ValueError(f"Text too long (max {max_length} chars, got {len(text)})")
            
            return text
            
        except Exception as e:
            raise CustomException(f"Text validation failed: {str(e)}", sys)
    
    @staticmethod
    def validate_texts(texts: List[str]) -> List[str]:
        """
        Validate batch of texts
        
        Args:
            texts: List of text inputs
            
        Returns:
            List of validated texts
            
        Raises:
            CustomException: If validation fails
        """
        try:
            if not texts:
                raise ValueError("Text list cannot be empty")
            
            if not isinstance(texts, list):
                raise TypeError(f"Expected list, got {type(texts).__name__}")
            
            validated = []
            for i, text in enumerate(texts):
                try:
                    validated.append(InputValidator.validate_text(text))
                except CustomException:
                    raise CustomException(f"Validation failed for text at index {i}", sys)
            
            return validated
            
        except Exception as e:
            raise CustomException(f"Batch validation failed: {str(e)}", sys)
    
    @staticmethod
    def validate_dataset(data: Union[List[Dict], List]) -> List[Dict]:
        """
        Validate dataset format
        
        Args:
            data: Dataset (list of dicts or list of items)
            
        Returns:
            Validated dataset
            
        Raises:
            CustomException: If validation fails
        """
        try:
            if not data:
                raise ValueError("Dataset cannot be empty")
            
            if not isinstance(data, list):
                raise TypeError(f"Expected list, got {type(data).__name__}")
            
            if len(data) == 0:
                raise ValueError("Dataset has zero items")
            
            return data
            
        except Exception as e:
            raise CustomException(f"Dataset validation failed: {str(e)}", sys)
    
    @staticmethod
    def validate_threshold(threshold: float) -> float:
        """
        Validate confidence threshold
        
        Args:
            threshold: Confidence threshold (0.0 to 1.0)
            
        Returns:
            Validated threshold
            
        Raises:
            CustomException: If validation fails
        """
        try:
            if not isinstance(threshold, (int, float)):
                raise TypeError(f"Expected float, got {type(threshold).__name__}")
            
            if not (0.0 <= threshold <= 1.0):
                raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
            
            return float(threshold)
            
        except Exception as e:
            raise CustomException(f"Threshold validation failed: {str(e)}", sys)
