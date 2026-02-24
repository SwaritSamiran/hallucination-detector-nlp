
#  Hallucination Detector - NLP

A production-grade PyTorch-based hallucination detection system using **ModernBERT**. Detects whether text contains factual hallucinations or truthful statements.

---

## Features

-  **Fast Inference** - GPU-optimized predictions with caching
-  **Batch Processing** - Process multiple texts efficiently  
-  **Confidence Scoring** - Get confidence metrics for each prediction
-  **Configurable Threshold** - Adjust sensitivity dynamically
-  **Production-Ready** - Clean FAANG-style architecture with proper error handling
-  **CLI Interface** - Easy-to-use command-line tool
-  **Flexible Input** - Single text, files, or JSON batches

---

### Installation

```bash
# Clone and install dependencies
pip install -r requirements.txt
```

### Basic Usage

**Single Text Prediction:**
```bash
python main.py --text "Paris is the capital of France"
```

**From File (one text per line):**
```bash
python main.py --file input.txt
```

**From JSON:**
```bash
python main.py --json samples.json
```

**With Custom Threshold & Output:**
```bash
python main.py --text "Some text" --threshold 0.8 --output results.json --summary
```

---

### Command Line Options

```bash
python main.py --help
```

**Input Options (choose one):**
- `--text TEXT` - Single text for prediction
- `--file PATH` - Path to text file (one text per line)
- `--json PATH` - Path to JSON file with texts

**Output Options:**
- `--threshold FLOAT` - Confidence threshold (0.0-1.0), default=0.75
- `--output PATH` - Save results to JSON file
- `--summary` - Print summary statistics
- `--verbose` - Verbose output

### Examples

**1. Single Prediction**
```bash
python main.py --text "The Moon is made of cheese"
```

**2. Batch from File**
```bash
# Create input.txt
cat > input.txt << EOF
Paris is the capital of France
The Earth is flat
Water boils at 100 degrees Celsius
EOF

python main.py --file input.txt --output results.json --summary
```

**3. Batch from JSON**
```bash
# Create samples.json
cat > samples.json << EOF
{
  "texts": [
    "Paris is the capital of France",
    "The capital of France is Lyon",
    "Water boils at 100 degrees Celsius"
  ]
}
EOF

python main.py --json samples.json --threshold 0.8 --summary
```

---

##  Architecture

```
src/
├── config/
│   └── config.py              # Centralized configuration
├── components/
│   ├── validator.py           # Input validation
│   ├── model_loader.py        # Model loading with caching
│   └── predictor.py           # Inference engine
├── pipeline/
│   └── predict_pipeline.py    # Orchestrator
├── logger.py                  # Logging
├── exception.py               # Custom exceptions
└── __init__.py

main.py                         # CLI entry point
```

---

## Python API

### Using the Pipeline Directly

```python
from src.pipeline.predict_pipeline import PredictionPipeline

# Initialize pipeline
pipeline = PredictionPipeline(threshold=0.75)

# Single prediction
result = pipeline.predict("Paris is the capital of France")
print(result)
# {
#     'text': 'Paris is the capital of France',
#     'label': 'Truth',
#     'label_id': 0,
#     'confidence': 0.9832,
#     'is_hallucination': False,
#     'meets_threshold': True
# }

# Batch prediction
texts = [
    "Paris is the capital of France",
    "The Earth is flat",
    "Water boils at 100 degrees Celsius"
]
results = pipeline.predict(texts)

# Get summary
summary = pipeline.get_summary(results)
print(summary)
# {
#     'total_samples': 3,
#     'hallucinations': 1,
#     'truths': 2,
#     'hallucination_rate': 0.3333,
#     'avg_confidence': 0.9521,
#     'threshold_used': 0.75
# }

# Update threshold
pipeline.set_threshold(0.85)
```

### Using Components Directly

```python
from src.components.model_loader import ModelLoader
from src.components.predictor import HallucinationPredictor
from src.components.validator import InputValidator

# Validate input
text = InputValidator.validate_text("Some text here")

# Load model (cached)
model, tokenizer = ModelLoader.load_model_and_tokenizer()
device = ModelLoader.get_device()

# Initialize predictor
predictor = HallucinationPredictor(model=model, tokenizer=tokenizer)

# Predict
result = predictor.predict_single(text)
print(result)
```

---

##  Output Format

### Single Prediction
```json
{
  "text": "Paris is the capital of France",
  "label": "Truth",
  "label_id": 0,
  "confidence": 0.9832,
  "is_hallucination": false,
  "meets_threshold": true
}
```

### Batch Prediction (List)
```json
[
  {
    "text": "Paris is the capital of France",
    "label": "Truth",
    "label_id": 0,
    "confidence": 0.9832,
    "is_hallucination": false,
    "meets_threshold": true
  },
  {
    "text": "The Earth is flat",
    "label": "Hallucination",
    "label_id": 1,
    "confidence": 0.9521,
    "is_hallucination": true,
    "meets_threshold": true
  }
]
```

### Summary Statistics
```json
{
  "total_samples": 100,
  "hallucinations": 35,
  "truths": 65,
  "hallucination_rate": 0.35,
  "avg_confidence": 0.9234,
  "threshold_used": 0.75
}
```

---

##  Configuration

Edit `src/config/config.py` to modify:

```python
# Model paths
MODEL_INPUT_PATH = "notebook/hallucination_model_v1"
MODEL_OUTPUT_PATH = "notebook/modernbert_final"

# Inference settings
DEFAULT_THRESHOLD = 0.75
MAX_LENGTH = 512
BATCH_SIZE = 16

# Device (auto-detects GPU)
USE_GPU = True
```

---

##  Models

The detector uses **ModernBERT-base**, a state-of-the-art transformer model fine-tuned for hallucination detection.

**Models included:**
- `notebook/hallucination_model_v1/` - Initial trained model
- `notebook/modernbert_final/` - Production model (recommended)

Both models support inference on GPU and CPU.

### Training Enhancement: DistilRoBERTa-base (GAN)

During training, we use **DistilRoBERTa-base** as a mutation engine to generate adversarial examples:

**Role:** Acts as the "Generator" in a GAN framework to create hard negative examples (plausible lies)

**Why DistilRoBERTa?**
- **40% smaller** than RoBERTa - fast mutation generation
- **Fill-Mask Pipeline** - excels at predicting plausible word replacements
- **Quality Mutations** - creates contextually relevant but false statements
- **Pre-trained on RoBERTa** - strong linguistic understanding

**How it works:**
1. Takes truthful facts: *"Paris is the capital of France"*
2. Masks key words: *"Paris is the capital of [MASK]"*
3. Predicts alternatives: "Madrid", "Rome", "London"
4. Creates false statements: *"Paris is the capital of Madrid"*
5. Our detector tries to identify it
6. If fooled → Add as hard negative for adversarial retraining

**Performance:** 50-100ms per mutation, 70-80% fool-rate on well-trained models

---

##  Performance

| Metric | Value |
|--------|-------|
| Single Inference | ~50-100ms (GPU), ~500-1000ms (CPU) |
| Batch Size | 16 (configurable) |
| Max Sequence Length | 512 tokens |
| Model Size | ~330MB (fp32), ~165MB (fp16) |
| Supported Devices | CUDA, CPU |

---

##  Troubleshooting

### "Model not found" Error
```bash
# Make sure model files exist
ls notebook/hallucination_model_v1/
# Should contain: config.json, tokenizer.json, model.safetensors, etc.
```

### GPU Not Available
```python
# Check CUDA availability
import torch
print(torch.cuda.is_available())

# If False, install CPU version or proper CUDA toolkit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Out of Memory (OOM)
```bash
# Reduce batch size in config
# Edit src/config/config.py:
# BATCH_SIZE = 8  # Instead of 16
```

---

##  Authors & Collaboration

This project was collaboratively developed with [@kavay-dev](https://github.com/kavay-dev).

- **Kavya Baxi** - kavyabaxi2018@gmail.com
- **Swarit Samiran** - swaritsamiran@gmail.com

---

## References

- [ModernBERT Paper](https://arxiv.org/abs/2105.05095)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---


