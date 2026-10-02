"""Streamlit demo.  Run:  streamlit run app/app.py   (after training: python -m src.run_experiments --stage final)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image

from src.config import cfg
from src.infer import Predictor
from src.xai import shap_bar_figure

st.set_page_config(page_title="Skin Lesion Risk (XAI)", layout="wide", initial_sidebar_state="collapsed")

# Inject Custom CSS for Premium Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(to bottom right, #0F172A, #1E293B);
        color: #F8FAFC;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }
    
    .sub-title {
        text-align: center;
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 40px;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #38BDF8;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -10px rgba(56, 189, 248, 0.5);
        border: none;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Skin Lesion Risk Assessment</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Explainable AI Analysis using Multimodal Health Data</p>", unsafe_allow_html=True)

st.warning("⚠️ Research prototype for a university project. NOT a medical device and NOT a diagnosis. Consult a dermatologist for any skin concern.")

@st.cache_resource
def load():
    return Predictor()

try:
    predictor = load()
except Exception as e:
    st.error(f"Models not found - train first with `python -m src.run_experiments --stage final`.\n\n{e}")
    st.stop()

# Layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("1. Patient Data")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 0, 110, 50)
    sex = c2.selectbox("Sex", ["male", "female", "unknown"])
    site = c3.selectbox("Lesion site", predictor.sites)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("2. Lesion Image")
    
    tab1, tab2 = st.tabs(["📸 Take Photo", "📁 Upload Image"])
    file = None
    
    with tab1:
        camera_file = st.camera_input("Capture from camera")
        if camera_file is not None:
            file = camera_file
            
    with tab2:
        upload_file = st.file_uploader("Upload Dermoscopic Image", type=["jpg", "jpeg", "png"])
        if upload_file is not None:
            file = upload_file
            
    go = st.button("Analyze Risk", type="primary", disabled=file is None)
    st.markdown("</div>", unsafe_allow_html=True)

if go and file is not None:
    img = Image.open(file)
    res = predictor.predict(img, age, sex, site)
    
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Analysis Results")
        
        risk = res["risk"]
        band = "LOW" if risk < 0.2 else "MODERATE" if risk < 0.5 else "HIGH"
        
        # Risk Metric
        st.metric("Malignant / Pre-malignant Risk", f"{risk:.1%}", band)
        st.caption("Score = P(akiec) + P(bcc) + P(mel). Uncalibrated model output; bands are illustrative.")
        
        st.markdown(f"**Most Likely Classification:** <span style='color:#38BDF8; font-size:1.2rem; font-weight:600;'>{res['top_class']}</span>", unsafe_allow_html=True)
        
        with st.expander("View Full Class Probabilities"):
            st.bar_chart(res["proba"])
            
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Explainable AI Insights (XAI)")
    
    xc1, xc2, xc3 = st.columns(3)
    
    with xc1:
        st.image(img.convert("RGB").resize((cfg.img_size, cfg.img_size)), caption="Original Input", use_container_width=True)
        
    with xc2:
        st.image(res["overlay"], caption="Grad-CAM: Image regions driving the prediction", use_container_width=True)
        
    with xc3:
        fig = shap_bar_figure(res["shap_values"], res["feature_names"])
        st.pyplot(fig)
        st.caption("SHAP: Clinical factors (XGBoost) impacting malignant classes.")
