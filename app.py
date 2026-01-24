import streamlit as st
import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
import tempfile
import os
from datetime import datetime
from inference import run_yolo, run_gradcam, disease_info
from fpdf import FPDF  # from fpdf2

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
FONT_REGULAR = os.path.join(FONT_DIR, "DejaVuSans.ttf")
FONT_BOLD = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Multi-Disease Detection and Severity Grading in Guava Leaves using Deep Learning",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PDF GENERATOR ENGINE ---
class DiagnosticReport(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("DejaVu", "", FONT_REGULAR, uni=True)
        self.add_font("DejaVu", "B", FONT_BOLD, uni=True)

    def header(self):
        self.set_fill_color(15, 23, 42)
        self.rect(0, 0, 210, 40, 'F')
        self.set_y(12)
        self.set_text_color(0, 255, 127)
        self.set_font("DejaVu", "B", 22)
        self.cell(0, 10, "DIAGNOSTIC REPORT", ln=True, align="C")
        self.set_font("DejaVu", "", 10)
        self.set_text_color(180, 180, 180)
        self.cell(0, 8, "Advanced Vascular Pathology and Severity Analytics", ln=True, align="C")
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(128)
        self.cell(
            0,
            10,
            f"Page {self.page_no()} {datetime.now().strftime('%Y-%m-%d')}",
            0,
            0,
            "C"
        )


def generate_detailed_pdf(diseases, sev, level, bgr_img):
    pdf = DiagnosticReport()
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 14)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 10, "I. VISUAL DIAGNOSTIC SCAN", ln=True)

    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    try:
        img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        Image.fromarray(img_rgb).save(tmp_path)
        os.close(fd)

        start_y = pdf.get_y()
        pdf.image(tmp_path, x=10, y=start_y + 5, w=110)

        img_h = (110 * bgr_img.shape[0]) / bgr_img.shape[1]
        pdf.set_y(start_y + img_h + 15)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    pdf.set_y(160)
    pdf.set_font("DejaVu", "B", 13)
    pdf.set_text_color(0, 140, 80)
    pdf.cell(0, 10, f"II. System Telemetry | Severity Level: {level}", ln=True)

    pdf.set_font("DejaVu", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, f"Pathogen Load: {sev:.2f}%", ln=True)
    pdf.cell(0, 7, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(6)

    for d in diseases:
        info = disease_info.get(d, {})

        pdf.set_fill_color(235, 255, 235)
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0, 9, f"Detected Disease: {d.replace('_',' ')}", ln=True, fill=True)

        pdf.set_font("DejaVu", "", 10)
        pdf.set_x(10)
        pdf.multi_cell(0, 6, "Description: " + info.get("description", "N/A"))
        pdf.set_x(10)
        pdf.multi_cell(0, 6, "Cause: " + info.get("cause", "N/A"))
        pdf.set_x(10)
        pdf.multi_cell(0, 6, "Damage: " + info.get("impact", "N/A"))
        pdf.set_x(10)
        pdf.multi_cell(0, 6, "Treatment: " + info.get("treatment", "N/A"))
        pdf.set_x(10)
        pdf.multi_cell(0, 6, "Organic Option: " + info.get("organic", "N/A"))
        pdf.set_x(10)
        pdf.multi_cell(0, 6, "Prevention: " + info.get("prevention", "N/A"))

        pdf.ln(4)
        pdf.set_font("DejaVu", "B", 10)
        pdf.set_text_color(0, 80, 150)
        pdf.set_x(10)
        pdf.multi_cell(0, 7, "Future-ready: " + info.get("future_safety", "N/A"))

        pdf.set_text_color(0, 0, 0)
        pdf.ln(6)

    out = pdf.output(dest="S")
    return bytes(out)


# --- ADVANCED MAGNIFICENT 3D UI ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syncopate:wght@700&family=Inter:wght@400;600&display=swap');

.stApp {
    background: radial-gradient(circle at center, #1a2e1a 0%, #050505 100%);
    color: #e0e0e0;
}

.idle-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
        url('https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?auto=format&fit=crop&q=80&w=2000');
    background-size: cover;
    background-position: center;
    z-index: -1;
    animation: slow-zoom 20s infinite alternate linear;
}

@keyframes slow-zoom {
    0% { transform: scale(1.0); }
    100% { transform: scale(1.1); }
}

div[data-testid="stHorizontalBlock"] { perspective: 1500px; }

div[data-testid="stColumn"] {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(20px);
    border-radius: 25px;
    padding: 2rem;
    border: 1px solid rgba(0, 255, 127, 0.15);
    transition: transform 0.6s cubic-bezier(0.23, 1, 0.32, 1), border 0.4s ease;
    animation: float 5s ease-in-out infinite;
}

div[data-testid="stColumn"]:hover {
    transform: rotateY(7deg) rotateX(4deg) scale(1.02) translateZ(10px);
    border: 1px solid rgba(0, 255, 127, 0.5);
    box-shadow: 0 20px 40px rgba(0,0,0,0.6);
}

@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}

@keyframes scanner {
    0% { top: 0%; opacity: 0; }
    50% { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}

.scan-container {
    position: relative;
    overflow: hidden;
    border-radius: 20px;
    border: 1px solid rgba(0, 255, 127, 0.2);
}

.scan-line {
    position: absolute;
    width: 100%;
    height: 4px;
    background: #00ff7f;
    box-shadow: 0 0 15px #00ff7f;
    animation: scanner 3s linear infinite;
    z-index: 5;
}

.main-title {
    font-family: 'Syncopate', sans-serif;
    font-size: 3.5rem !important;
    background: linear-gradient(90deg, #00ff7f, #4ade80, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}

[data-testid="stMetricValue"] {
    font-family: 'Syncopate', sans-serif;
    color: #00ff7f !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Multi-Disease Detection and Severity Grading in Guava Leaves using Deep Learning</h1>', unsafe_allow_html=True)
st.markdown(
    "<p style='letter-spacing: 4px; color: #94a3b8; text-align: center;'>"
    "ULTRA-HD NEURAL PATHOLOGY ANALYSIS</p>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### 🛠️ MISSION CONTROL")
    mode = st.radio("SENSING MODE", ["Standard Diagnostic", "Enhanced XAI (Grad-CAM)"])
    st.markdown("---")
    files = st.file_uploader(
        "📥 DATA INGESTION",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

if files:
    for file in files:
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, 1)

        col_viz, col_anal = st.columns([1.6, 1])

        with st.spinner("⚡ CONDUCTING MOLECULAR SCAN..."):
            yolo_img, sev, level, diseases, results, boxes = run_yolo(img)

            if mode == "Enhanced XAI (Grad-CAM)":
                final_view = run_gradcam(yolo_img, boxes)
            else:
                final_view = yolo_img

        # ✅ CORRECT INDENTATION ENDS HERE

        with col_viz:
            st.markdown(
                "<h3 style='color: #00ff7f;'>🛰️ DIAGNOSTIC OVERLAY</h3>",
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="scan-container"><div class="scan-line"></div>',
                unsafe_allow_html=True
            )
            st.image(
                cv2.cvtColor(final_view, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_anal:
            st.markdown(
                "<h3 style='color: #00ff7f;'>📊 Damage </h3>",
                unsafe_allow_html=True
            )
            st.metric("PATHOGEN LOAD", f"{sev:.2f}%", level)

            pdf_data = generate_detailed_pdf(diseases, sev, level, final_view)
            st.download_button(
                label="📥 DOWNLOAD HD AUDIT REPORT",
                data=pdf_data,
                file_name=f"GuavaAudit_{datetime.now().strftime('%H%M%S')}.pdf",
                mime="application/pdf"
            )


            st.markdown("---")
            if diseases:
                for d in diseases:
                    with st.expander(f"🔴 DETECTION: {d.upper()}", expanded=True):
                        info = disease_info.get(d, {})
                        st.markdown(f"**What it is:** {info.get('description','N/A')}")
                        st.markdown(f"**Why it happened:** {info.get('cause','N/A')}")
                        st.markdown(f"**Damage caused:** {info.get('impact','N/A')}")
                        st.markdown(f"**Immediate treatment:** {info.get('treatment','N/A')}")
                        st.markdown(f"**Organic option:** {info.get('organic','N/A')}")
                        st.markdown(f"**Prevention:** {info.get('prevention','N/A')}")
                        st.markdown(
                            f"<span style='color:#00ff7f'><b>Future-ready:</b> "
                            f"{info.get('future_safety','N/A')}</span>",
                            unsafe_allow_html=True
                        )
        st.divider()
else:
    st.markdown('<div class="idle-bg"></div>', unsafe_allow_html=True)
    st.markdown(
        "<br><br><br><br>"
        "<div style='text-align:center;'>"
        "<h2 style='color:#00ff7f; font-family:Syncopate;'>NEURAL CORE IDLE</h2>"
        "<p style='color:#fff; font-size:1.2rem;'>"
        "Awaiting specimen injection for vascular analysis.</p></div>",
        unsafe_allow_html=True
    )
