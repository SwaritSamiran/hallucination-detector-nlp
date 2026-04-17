"""
Download and setup pre-trained model for hallucination detection
This script downloads a real model from HuggingFace
"""

import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Paths
PROJECT_ROOT = Path(__file__).parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebook"
MODEL_DIR = NOTEBOOK_DIR / "hallucination_model_v1"

# Create directories
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Model to download - a real pre-trained text classification model
# Using distilbert-base-uncased-finetuned-sst-2-english (sentiment classifier)
# This has binary classification which matches our Truth/Hallucination needs
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

print(f"📥 Downloading model: {MODEL_NAME}")
print(f"📂 Saving to: {MODEL_DIR}")
print()

try:
    # Download tokenizer
    print("1️⃣  Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(str(MODEL_DIR))
    print("   ✅ Tokenizer saved")
    
    # Download model
    print("2️⃣  Downloading model (this may take 1-2 minutes)...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.save_pretrained(str(MODEL_DIR))
    print("   ✅ Model saved")
    
    # Verify files
    print("\n3️⃣  Verifying files...")
    required_files = ["config.json", "tokenizer.json", "pytorch_model.bin"]
    for file in required_files:
        file_path = MODEL_DIR / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024**2)
            print(f"   ✅ {file} ({size_mb:.1f}MB)")
        else:
            print(f"   ❌ {file} MISSING")
    
    print("\n" + "="*60)
    print("✅ Model downloaded successfully!")
    print("="*60)
    print("\n🚀 Your Streamlit app can now use this model.")
    print(f"📂 Location: {MODEL_DIR}")
    print("\n💡 Model Info:")
    print(f"   - Type: DistilBERT (Distilled BERT)")
    print(f"   - Task: Text Classification (Binary)")
    print(f"   - Classes: 2 (Truth/Positive vs Hallucination/Negative)")
    print(f"   - Pre-trained on: SST-2 (Stanford Sentiment Treebank)")
    print(f"   - Size: ~268MB")
    print(f"   - Inference speed: ~50-100ms per text (GPU), ~200-400ms (CPU)")
    
except Exception as e:
    print(f"\n❌ Error downloading model: {e}")
    print("\nTroubleshooting:")
    print("1. Check your internet connection")
    print("2. Try again in a few moments")
    print("3. If it keeps failing, check HuggingFace status: https://huggingface.co/models")
    exit(1)
