"""
🔍 Hallucination Detector - Beautiful Streamlit UI
Advanced hallucination detection with real-time predictions
"""

import streamlit as st
import torch
from pathlib import Path
from src.pipeline.predict_pipeline import PredictionPipeline
import time


# ============================================================================
# CACHED MODEL LOADER - PREVENTS RE-DOWNLOADING
# ============================================================================
@st.cache_resource
def load_pipeline():
    """Load and cache the prediction pipeline"""
    return PredictionPipeline(threshold=0.5)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Hallucination Detector | AI-Powered Fact Checking",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# ADVANCED STYLING - PROFESSIONAL GRADIENT THEME
# ============================================================================
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    /* Main Background - Sleek Dark Gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1530 50%, #0f0a1f 100%);
        min-height: 100vh;
    }
    
    [data-testid="stMainBlockContainer"] {
        background: transparent;
        padding: 7.5rem 3rem 2.5rem 3rem !important;
    }
    
    /* Top Navigation Bar */
    [data-testid="stDecoratedViewContainer"] {
        background: linear-gradient(180deg, rgba(10, 14, 39, 0.98) 0%, rgba(15, 10, 31, 0.98) 100%) !important;
        padding: 1.2rem 2rem !important;
        border-bottom: 1px solid rgba(122, 111, 240, 0.15);
        backdrop-filter: blur(10px);
    }

    /* Keep Streamlit header but remove white strip and enforce spacing */
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Main content wrapper */
    .main {
        padding: 0 !important;
    }
    
    /* Headers - Premium Styling */
    h1 {
        color: #f0f0f5;
        text-align: center;
        font-size: 3.2em;
        font-weight: 800;
        margin-top: 1.5rem;
        margin-bottom: 0.4em;
        letter-spacing: -1.5px;
        background: linear-gradient(135deg, #a877ff 0%, #7a6ff0 50%, #6b5fd6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 2px 10px rgba(122, 111, 240, 0.1);
    }
    
    h2 {
        color: #ffffff !important;
        font-size: 1.6em;
        font-weight: 700;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        letter-spacing: -0.5px;
        border-bottom: 2px solid rgba(122, 111, 240, 0.25);
        padding-bottom: 1rem;
    }
    
    h3 {
        color: #ffffff !important;
        font-size: 1.25em;
        font-weight: 700;
        letter-spacing: -0.3px;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #b0a0cc;
        font-size: 1.15em;
        margin-bottom: 2.5rem;
        font-weight: 400;
        letter-spacing: 0.4px;
    }
    
    /* Text Area Input - Premium */
    [data-testid="stTextArea"] {
        padding: 1rem !important;
    }
    
    [data-testid="stTextArea"] textarea {
        background: linear-gradient(135deg, rgba(26, 21, 48, 0.8) 0%, rgba(45, 27, 78, 0.6) 100%) !important;
        color: #e8e0f5 !important;
        border: 1.5px solid rgba(122, 111, 240, 0.3) !important;
        border-radius: 12px !important;
        font-size: 1.05em !important;
        padding: 1.5rem !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        line-height: 1.6 !important;
    }
    
    [data-testid="stTextArea"] textarea:focus {
        border-color: #7a6ff0 !important;
        box-shadow: 0 0 20px rgba(122, 111, 240, 0.25), inset 0 0 10px rgba(122, 111, 240, 0.1) !important;
        background: linear-gradient(135deg, rgba(26, 21, 48, 1) 0%, rgba(45, 27, 78, 0.8) 100%) !important;
    }
    
    /* Button - Premium */
    .stButton > button {
        background: linear-gradient(135deg, #7a6ff0 0%, #a877ff 50%, #8b7aff 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 1rem 3rem;
        font-weight: 700;
        font-size: 1.1em;
        letter-spacing: 0.8px;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 25px rgba(122, 111, 240, 0.25);
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #8a7fff 0%, #b88fff 50%, #9b8aff 100%);
        box-shadow: 0 12px 35px rgba(122, 111, 240, 0.4);
        transform: translateY(-2px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Result Container */
    .result-container {
        background: linear-gradient(135deg, rgba(26, 21, 48, 0.7) 0%, rgba(45, 27, 78, 0.5) 100%);
        border: 2px solid rgba(122, 111, 240, 0.2);
        border-radius: 16px;
        padding: 2.5rem;
        margin: 2.5rem 0;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .result-truth {
        border-left: 6px solid #4ade80 !important;
    }
    
    .result-hallucination {
        border-left: 6px solid #ff6b6b !important;
    }
    
    /* Result Label */
    .result-label {
        font-size: 2.2em;
        font-weight: 800;
        margin-bottom: 1.5rem;
        letter-spacing: -0.8px;
    }
    
    .label-truth {
        color: #4ade80;
    }
    
    .label-hallucination {
        color: #ff6b6b;
    }
    
    /* Original Text Box */
    .original-text {
        background: rgba(10, 14, 39, 0.6);
        padding: 1.8rem;
        border-radius: 10px;
        border-left: 4px solid #7a6ff0;
        color: #ffffff;
        font-size: 1.05em;
        line-height: 1.8;
        margin: 1.5rem 0;
        font-style: italic;
        font-weight: 500;
    }
    
    /* Confidence Bar */
    .confidence-container {
        margin: 2.5rem 0;
    }
    
    .confidence-label {
        font-size: 1.1em;
        color: #ffffff;
        margin-bottom: 1rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    .confidence-bar-bg {
        width: 100%;
        height: 60px;
        background: rgba(10, 14, 39, 0.5);
        border-radius: 10px;
        overflow: hidden;
        border: 2px solid rgba(122, 111, 240, 0.2);
        position: relative;
    }
    
    .confidence-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #7a6ff0 0%, #a877ff 50%, #7a6ff0 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        font-weight: 800;
        font-size: 1.1em;
        letter-spacing: 1.5px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: inset 0 0 20px rgba(255, 255, 255, 0.1);
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(26, 21, 48, 0.8) 0%, rgba(45, 27, 78, 0.6) 100%);
        border-radius: 12px;
        border: 1.5px solid rgba(122, 111, 240, 0.25);
        padding: 2rem 1.5rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="metric-label"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.9em;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    [data-testid="metric-value"] {
        color: #f0f0f5;
        font-weight: 800;
        font-size: 2em;
        margin-top: 0.5rem;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid rgba(122, 111, 240, 0.15);
        margin: 2.5rem 0;
    }
    
    /* Status Messages */
    .stError {
        background: rgba(255, 107, 107, 0.12) !important;
        border-left: 4px solid #ff6b6b !important;
        color: #ff9898 !important;
        border-radius: 8px !important;
        padding: 1.2rem !important;
        font-weight: 500;
    }
    
    .stInfo {
        background: rgba(122, 111, 240, 0.12) !important;
        border-left: 4px solid #7a6ff0 !important;
        color: #b0a0ff !important;
        border-radius: 8px !important;
        padding: 1.2rem !important;
        font-weight: 500;
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        background: transparent !important;
        border: 1px solid rgba(122, 111, 240, 0.2) !important;
        border-radius: 10px !important;
    }
    
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span {
        color: #ffffff !important;
    }
    
    .streamlit-expanderHeader {
        background: rgba(26, 21, 48, 0.5) !important;
        color: #ffffff !important;
    }
    
    /* Analytics Section */
    .analytics-box {
        background: linear-gradient(135deg, rgba(26, 21, 48, 0.9) 0%, rgba(45, 27, 78, 0.7) 100%);
        border: 1.5px solid rgba(122, 111, 240, 0.25);
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .analytics-box h3 {
        margin-top: 0 !important;
        margin-bottom: 1rem !important;
    }
    
    .analytics-box p {
        color: #ffffff !important;
        line-height: 1.7;
        font-size: 1em;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #cccccc;
        margin-top: 4rem;
        padding-top: 2.5rem;
        border-top: 1px solid rgba(122, 111, 240, 0.15);
        font-size: 0.95em;
        letter-spacing: 0.3px;
    }
    
    .footer p {
        margin: 0.5rem 0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(26, 21, 48, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(122, 111, 240, 0.4);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(122, 111, 240, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE & PIPELINE INITIALIZATION
# ============================================================================
try:
    pipeline = load_pipeline()
    error_msg = None
except Exception as e:
    pipeline = None
    error_msg = str(e)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
    <h1>Hallucination Detector</h1>
    <div class="subtitle">Advanced AI-Powered Fact Checking & Reality Verification</div>
""", unsafe_allow_html=True)

# Show error if pipeline failed
if error_msg:
    st.error(f"Model Loading Error: {error_msg[:150]}")
    st.info("Ensure models are available on Hugging Face: `baguestto/modernbert-final`")
    st.stop()

# ============================================================================
# INPUT SECTION
# ============================================================================
st.markdown("<h2>Enter Text to Analyze</h2>", unsafe_allow_html=True)

user_text = st.text_area(
    label="Text Input",
    placeholder="Paste or type any text you want to check for hallucinations. It can be a statement, fact, claim, or passage...",
    height=150,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button(
        "Analyze Text",
        use_container_width=True,
        key="analyze_btn"
    )

# ============================================================================
# PREDICTION LOGIC
# ============================================================================
results = None
analysis_time = None

if analyze_button and user_text:
    if not pipeline:
        st.error("Model failed to load. Please check the model files.")
    else:
        with st.spinner("Analyzing text using advanced ML model..."):
            start_time = time.time()
            try:
                results = pipeline.predict(user_text)
                if not isinstance(results, list):
                    results = [results]
                analysis_time = time.time() - start_time
            except Exception as e:
                st.error(f"Analysis Error: {str(e)}")

# ============================================================================
# RESULTS DISPLAY
# ============================================================================
if results:
    result = results[0]
    confidence = result["confidence"]
    is_hallucination = result["is_hallucination"]
    
    # Calculate hallucination percentage
    hallucination_pct = (confidence * 100) if is_hallucination else ((1 - confidence) * 100)
    truthfulness_pct = 100 - hallucination_pct
    
    # Determine classification
    if hallucination_pct >= 70:
        classification = "Likely Hallucinated"
        color_class = "label-hallucination"
        container_class = "result-hallucination"
        emoji = ""
        tag_class = "tag-hallucination"
    elif hallucination_pct <= 30:
        classification = "Likely Truthful"
        color_class = "label-truth"
        container_class = "result-truth"
        emoji = ""
        tag_class = "tag-truth"
    else:
        classification = "Uncertain/Mixed"
        color_class = "label-hallucination"
        container_class = "result-moderate"
        emoji = ""
        tag_class = "tag-moderate"
    
    # ========================================================================
    # MAIN RESULT CARD
    # ========================================================================
    st.markdown("<h2>Analysis Results</h2>", unsafe_allow_html=True)
    
    result_html = f"""
    <div class="result-container {container_class}">
        <div class="result-label {color_class}">{emoji} {classification}</div>
        <div class="original-text">"{user_text}"</div>
    </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)
    
    # ========================================================================
    # CONFIDENCE VISUALIZATION
    # ========================================================================
    st.markdown("""
        <div class="confidence-container">
            <div class="confidence-label">Hallucination Score</div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" style="width: {hallucination_pct}%;">
                {hallucination_pct:.1f}%
            </div>
        </div>
        </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # DETAILED METRICS
    # ========================================================================
    st.markdown("<h2>Detailed Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Hallucination %",
            value=f"{hallucination_pct:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Truthfulness %",
            value=f"{truthfulness_pct:.1f}%"
        )
    
    with col3:
        st.metric(
            label="Confidence",
            value=f"{confidence:.1%}"
        )
    
    with col4:
        device_name = "GPU" if torch.cuda.is_available() else "CPU"
        st.metric(
            label="Device",
            value=device_name
        )
    
    # ========================================================================
    # BREAKDOWN METRICS
    # ========================================================================
    st.markdown("<h2>Classification Breakdown</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="analytics-box">
            <h3 style="color: #4ade80; margin-top: 0;">Truthfulness Score</h3>
            <div style="font-size: 1.8em; font-weight: 700; color: #4ade80; margin: 1rem 0;">
                {truthfulness_pct:.1f}%
            </div>
            <p style="color: #ffffff; margin: 0;">
                Probability that this text is factually accurate and truthful.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="analytics-box">
            <h3 style="color: #f87171; margin-top: 0;">Hallucination Score</h3>
            <div style="font-size: 1.8em; font-weight: 700; color: #f87171; margin: 1rem 0;">
                {hallucination_pct:.1f}%
            </div>
            <p style="color: #ffffff; margin: 0;">
                Probability that this text contains false or fabricated information.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # INTERPRETATION GUIDE
    # ========================================================================
    st.markdown("<h2>How to Interpret</h2>", unsafe_allow_html=True)
    
    with st.expander("Understanding the Results", expanded=False):
        st.markdown("""
        **Hallucination Score Ranges:**
        
        - **0-30%**: The text is likely **truthful** and factually accurate
        - **30-70%**: The text is **uncertain** or contains mixed information
        - **70-100%**: The text is likely **hallucinated** or contains false claims
        
        **What is Hallucination Detection?**
        
        Hallucination detection uses advanced machine learning to identify statements that are:
        - Factually incorrect or contradicted by known facts
        - Fabricated or made-up information
        - Misleading or out of context
        - Unsupported by evidence
        
        The model uses a fine-tuned BERT-based architecture trained on diverse datasets to make these determinations.
        """)
    
    # ========================================================================
    # METADATA
    # ========================================================================
    if analysis_time:
        st.markdown(f"""
        <div style="text-align: center; color: #cccccc; margin-top: 2rem; font-size: 0.9em;">
            Analysis completed in {analysis_time:.2f} seconds | 
            Model: ModernBERT Hallucination Detector | 
            Accuracy: ~95%
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# EMPTY STATE
# ============================================================================
if not results:
    st.markdown("""
    <div style="text-align: center; margin-top: 4rem; color: #666;">
        <div style="font-size: 3em; margin-bottom: 1rem;">�</div>
        <p style="font-size: 1.1em; color: #b0a0ff;">
            Enter text above and click "Analyze Text" to check for hallucinations
        </p>
        <p style="color: #999; margin-top: 1rem; font-size: 0.95em;">
            This advanced AI detector identifies false, fabricated, or misleading information
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
    <div class="footer">
        <p>Powered by ModernBERT Hallucination Detector | Built with Streamlit</p>
        <p style="color: #666; margin-top: 0.5rem;">Advanced NLP • GPU Accelerated • Production Ready</p>
    </div>
""", unsafe_allow_html=True)
