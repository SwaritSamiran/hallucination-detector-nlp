# 🎨 Streamlit UI for Hallucination Detector

A beautiful, interactive web interface for the hallucination detection model.

## ✨ Features

- **Single Text Input** - Check one sentence at a time
- **Batch Processing** - Upload multiple texts (one per line)
- **JSON Support** - Import batch predictions from JSON
- **Visual Confidence Bars** - See prediction confidence as gorgeous gradient bars
- **Real-time Threshold Control** - Adjust sensitivity with a slider (0.0-1.0)
- **Detailed Results** - View per-sample predictions with full metrics
- **Summary Statistics** - Get hallucination rate, average confidence, etc.
- **Export Results** - Download predictions as JSON or CSV
- **Responsive Design** - Works on desktop, tablet, mobile

## 🚀 Getting Started

### Prerequisites
- Python 3.8+ (3.12 recommended)
- All packages from `requirements.txt` installed (including streamlit)

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the model files exist at:
   - `notebook/hallucination_model_v1/` (training model)
   - OR `notebook/modernbert_final/` (production model - recommended)

### Running the App

#### Option 1: PowerShell (Recommended)
```powershell
.\run_app.ps1
```

#### Option 2: Command Prompt
```cmd
run_app.bat
```

#### Option 3: Manual Command
```bash
python -m streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`

## 📋 Usage Guide

### Single Text Mode
1. Enter text in the textbox (e.g., "Paris is the capital of France")
2. Click **Check Text**
3. View:
   - Label (Truth or Hallucination)
   - Confidence percentage (visual bar + percentage)
   - Meets Threshold indicator

### Batch Mode
1. Enter multiple texts, one per line
2. Click **Check All Texts**
3. View:
   - Summary statistics (total, hallucination count/rate, avg confidence)
   - Table of all results
   - Expandable detailed view per result

### JSON Mode
Paste JSON in either format:
```json
["text 1", "text 2", "text 3"]
```
OR
```json
{"texts": ["text 1", "text 2", "text 3"]}
```

## ⚙️ Configuration

### Adjust Threshold
Use the slider in the left sidebar (0.0-1.0):
- **Lower threshold** (e.g., 0.5) = More permissive, catches potential hallucinations
- **Higher threshold** (e.g., 0.9) = More strict, only flags very confident hallucinations

### Custom Model Path
Edit `src/config/config.py`:
```python
MODEL_INPUT_PATH = "path/to/your/model"
```

## 📊 Output Format

### Results Display
Each result shows:
- **Text** - The input text
- **Label** - "Truth" or "Hallucination"
- **Confidence** - Probability score (0.0-1.0) as visual bar
- **Meets Threshold** - Whether confidence exceeds the threshold

### Export Formats

**JSON:**
```json
[
  {
    "text": "Paris is the capital of France",
    "label": "Truth",
    "label_id": 0,
    "confidence": 0.9832,
    "is_hallucination": false,
    "meets_threshold": true
  }
]
```

**CSV:**
```
Text,Label,Confidence,Hallucination,Meets Threshold
"Paris is the capital of France",Truth,0.9832,No,Yes
```

## 🎨 UI Components

### Confidence Bar
The gradient bar shows confidence on a scale:
- Red (left) = Low confidence
- Yellow (center) = Medium confidence
- Green (right) = High confidence

The percentage and visual width both indicate confidence level.

### Summary Card
For batch results:
- Total samples processed
- Count of truths vs hallucinations
- Hallucination rate (in %)
- Average confidence across all predictions
- Current threshold used

## 🔧 Troubleshooting

### "Model not found" Error
**Solution:** Ensure model files exist:
```bash
ls notebook/hallucination_model_v1/
# Should show: config.json, tokenizer.json, model.safetensors, etc.
```

### Slow Inference (CPU)
**Solution:** GPU is recommended but optional. It will work on CPU but slower (~500-1000ms per prediction).

### Out of Memory
**Solution:** Reduce batch size in `src/config/config.py`:
```python
BATCH_SIZE = 8  # Instead of 16
```

### Port Already in Use
The app defaults to port 8501. To use a different port:
```bash
python -m streamlit run app.py --server.port 8502
```

## 📈 Performance

- **Single Prediction:** ~50-100ms (GPU), ~500-1000ms (CPU)
- **Batch (16 texts):** ~100-200ms (GPU), ~1-2s (CPU)
- **Model Size:** ~330MB (fp32), ~165MB (fp16)

## 🎯 Model Info

- **Architecture:** ModernBERT
- **Task:** Binary Classification (Truth vs Hallucination)
- **Classes:** 2 (Truth, Hallucination)
- **Max Input Length:** 512 tokens
- **Training Method:** GAN-enhanced with DistilRoBERTa adversarial examples

## 📝 API Usage (Python)

If you want to use the detector programmatically without Streamlit:

```python
from src.pipeline.predict_pipeline import PredictionPipeline

# Initialize
pipeline = PredictionPipeline(threshold=0.75)

# Single prediction
result = pipeline.predict("Paris is the capital of France")
print(result)
# {'text': '...', 'label': 'Truth', 'confidence': 0.98, ...}

# Batch prediction
results = pipeline.predict(["text 1", "text 2", "text 3"])
print(results)
# [{'text': '...', 'label': 'Truth', ...}, ...]

# Summary
summary = pipeline.get_summary(results)
print(summary)
# {'hallucinations': 1, 'truths': 2, 'hallucination_rate': 0.33, ...}
```

## 🤝 Contributing

Contributions welcome! The UI is in `app.py` and styling can be customized in the CSS section.

## 📧 Authors

- Kavya Baxi - baxikavya2018@gmail.com
- Swarit Samiran - swaritsamiran@gmail.com

---

**Happy detecting! 🎉**
