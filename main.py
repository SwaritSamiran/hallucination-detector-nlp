"""
Main entry point for Hallucination Detector
CLI interface for predictions
"""

import argparse
import json
from pathlib import Path
from typing import Union, List

from src.pipeline.predict_pipeline import PredictionPipeline
from src.logger import logging
from src.exception import CustomException


def create_parser():
    """Create CLI argument parser"""
    parser = argparse.ArgumentParser(
        description="Hallucination Detector - Detect hallucinations in text using ModernBERT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single text prediction
  python main.py --text "Paris is the capital of France"
  
  # File-based prediction
  python main.py --file input.txt
  
  # Batch prediction from JSON
  python main.py --json samples.json
  
  # Custom threshold
  python main.py --text "Some text" --threshold 0.8
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        type=str,
        help="Single text for prediction"
    )
    input_group.add_argument(
        "--file",
        type=str,
        help="Path to text file (one text per line)"
    )
    input_group.add_argument(
        "--json",
        type=str,
        help="Path to JSON file with texts (list or dict with 'texts' key)"
    )
    
    # Options
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Confidence threshold (0.0-1.0), default=0.75"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    return parser


def load_texts_from_file(filepath: str) -> List[str]:
    """Load texts from file (one per line)"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        if not texts:
            raise ValueError("File is empty")
        
        logging.info(f"Loaded {len(texts)} texts from {filepath}")
        return texts
    except Exception as e:
        raise CustomException(f"Failed to load file: {str(e)}", __name__)


def load_texts_from_json(filepath: str) -> List[str]:
    """Load texts from JSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, list):
            texts = data
        elif isinstance(data, dict) and "texts" in data:
            texts = data["texts"]
        else:
            raise ValueError("JSON must be a list or dict with 'texts' key")
        
        if not texts:
            raise ValueError("No texts found in JSON")
        
        logging.info(f"Loaded {len(texts)} texts from {filepath}")
        return texts
    except Exception as e:
        raise CustomException(f"Failed to load JSON: {str(e)}", __name__)


def save_results(results: Union[dict, list], filepath: str):
    """Save results to JSON"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"Results saved to {filepath}")
    except Exception as e:
        raise CustomException(f"Failed to save results: {str(e)}", __name__)


def format_result(result: dict) -> str:
    """Format single result for display"""
    label_emoji = "✅" if result["label"] == "Truth" else "⚠️"
    return f"""{label_emoji} {result['label']}
   Confidence: {result['confidence']:.2%}
   Meets Threshold: {result['meets_threshold']}"""


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        logging.info("Starting Hallucination Detector...")
        pipeline = PredictionPipeline(threshold=args.threshold)
        
        # Load texts
        if args.text:
            texts = [args.text]
        elif args.file:
            texts = load_texts_from_file(args.file)
        else:  # args.json
            texts = load_texts_from_json(args.json)
        
        # Predict
        logging.info(f"Running predictions on {len(texts)} text(s)...")
        results = pipeline.predict(texts)
        
        # Format output
        if isinstance(results, list):
            print("\n" + "="*60)
            print(f"Predictions ({len(results)} samples)")
            print("="*60)
            for i, result in enumerate(results, 1):
                print(f"\n[{i}] {result['text'][:80]}...")
                print(format_result(result))
        else:
            print("\n" + "="*60)
            print("Prediction")
            print("="*60)
            print(f"\nText: {results['text']}")
            print(format_result(results))
        
        # Summary
        if args.summary:
            summary = pipeline.get_summary(results)
            print("\n" + "="*60)
            print("Summary Statistics")
            print("="*60)
            for key, value in summary.items():
                print(f"{key}: {value}")
        
        # Save results
        if args.output:
            save_results(results, args.output)
            print(f"\n✓ Results saved to {args.output}")
        
        print("\n✓ Complete!")
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
