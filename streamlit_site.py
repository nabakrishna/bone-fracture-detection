"""
Bone Fracture Detection Dashboard - Streamlit
=============================================
Medical-grade X-ray analysis dashboard with clean UI.

Install:
    pip install streamlit ultralytics opencv-python torch pillow

Run:
    streamlit run fracture_dashboard.py
"""

import streamlit as st
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import io
import time
import tempfile
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BoneVision AI",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f0f4f8;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1b2a !important;
    border-right: 1px solid #1e3a5f;
}

[data-testid="stSidebar"] * {
    color: #c8d8e8 !important;
}

[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
    margin-top: 8px;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

[data-testid="stSidebar"] .sidebar-brand {
    padding: 1.5rem 1rem 1rem;
    border-bottom: 1px solid #1e3a5f;
    margin-bottom: 1.5rem;
}

/* ── Main header ── */
.main-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a3a5c 60%, #1a6b8a 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    box-shadow: 0 8px 32px rgba(13,27,42,0.18);
}

.main-header .icon {
    font-size: 3rem;
    line-height: 1;
}

.main-header h1 {
    color: #ffffff !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    padding: 0 !important;
    letter-spacing: -0.5px;
}

.main-header p {
    color: #a8c8e8 !important;
    font-size: 0.95rem;
    margin: 0.25rem 0 0 !important;
}

/* ── Cards ── */
.card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(13,27,42,0.08);
    border: 1px solid #e2eaf2;
    margin-bottom: 1rem;
}

.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #7a9ab5;
    margin-bottom: 0.75rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #b8cfe0 !important;
    border-radius: 14px !important;
    background: #f7fafd !important;
    padding: 1rem !important;
    transition: all 0.2s;
}




[data-testid="stFileUploader"]:hover {
    border-color: #1a6b8a !important;
    background: #eef6fb !important;
}

/* ── Metric cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1rem 0;
}

.metric-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    border: 1px solid #e2eaf2;
    box-shadow: 0 2px 8px rgba(13,27,42,0.06);
    text-align: center;
}

.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #0d1b2a;
    line-height: 1;
    font-family: 'DM Mono', monospace;
}

.metric-card .label {
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #7a9ab5;
    margin-top: 0.35rem;
}

.metric-card.highlight .value { color: #c0392b; }
.metric-card.safe .value { color: #27ae60; }
.metric-card.info .value { color: #1a6b8a; }

/* ── Detection result boxes ── */
.detection-item {
    background: #f7fafd;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid #1a6b8a;
    border: 1px solid #e2eaf2;
    border-left: 4px solid #1a6b8a;
}

.detection-item.fracture { border-left-color: #e74c3c; background: #fff8f7; }
.detection-item.bone { border-left-color: #e67e22; background: #fffaf5; }
.detection-item.metal { border-left-color: #8e44ad; background: #fdf8ff; }
.detection-item.other { border-left-color: #1a6b8a; }

.detection-item .det-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.detection-item .det-class {
    font-weight: 600;
    font-size: 0.95rem;
    color: #0d1b2a;
}

.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
}

.badge-high { background: #d5f5e3; color: #1e8449; }
.badge-med  { background: #fef9e7; color: #b7950b; }
.badge-low  { background: #fce8e6; color: #c0392b; }
.badge-info { background: #e8f4fd; color: #1a6b8a; }

.det-coords {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #7a9ab5;
    margin-top: 0.4rem;
}

/* ── Status bar ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1rem;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 1rem;
}

.status-ready    { background: #d5f5e3; color: #1e8449; }
.status-warning  { background: #fef9e7; color: #9a7d0a; }
.status-error    { background: #fce8e6; color: #922b21; }
.status-info     { background: #e8f4fd; color: #154360; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0d1b2a, #1a6b8a) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px;
    transition: all 0.2s !important;
    width: 100%;
    box-shadow: 0 4px 14px rgba(26,107,138,0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(26,107,138,0.4) !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #1a6b8a !important;
    border-color: #1a6b8a !important;
}

/* ── Selectbox ── */
.stSelectbox [data-baseweb="select"] {
    border-radius: 8px !important;
}

/* ── Progress ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #1a3a5c, #1a6b8a) !important;
    border-radius: 4px !important;
}

/* ── Divider ── */
hr { border-color: #e2eaf2 !important; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar section labels ── */
.sidebar-section {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #4a7a9b !important;
    margin: 1.5rem 0 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e3a5f;
}

/* ── Image caption ── */
.img-caption {
    font-size: 0.72rem;
    color: #7a9ab5;
    text-align: center;
    margin-top: 0.4rem;
    font-style: italic;
}

/* ── No detection state ── */
.no-detection {
    text-align: center;
    padding: 2.5rem 1rem;
    color: #7a9ab5;
}
.no-detection .nd-icon { font-size: 3rem; margin-bottom: 0.75rem; }
.no-detection .nd-title { font-size: 1.1rem; font-weight: 600; color: #0d1b2a; }
.no-detection .nd-sub { font-size: 0.85rem; margin-top: 0.35rem; }

/* Column gap fix */
[data-testid="column"] { padding: 0 0.5rem !important; }

/* ── Sidebar toggle button ── */
#sidebar-toggle-btn {
    position: fixed;
    top: 50%;
    transform: translateY(-50%);
    z-index: 9999;
    background: #1a6b8a;
    color: white;
    border: none;
    border-radius: 0 8px 8px 0;
    width: 22px;
    height: 52px;
    cursor: pointer;
    font-size: 0.7rem;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 2px 0 8px rgba(0,0,0,0.18);
    transition: background 0.2s, left 0.3s;
    writing-mode: vertical-rl;
    letter-spacing: 1px;
    font-weight: 700;
}
#sidebar-toggle-btn:hover { background: #0d4f6b; }

/* ── Cap image heights so no scroll needed ── */
[data-testid="stImage"] img {
    max-height: 340px !important;
    width: auto !important;
    max-width: 100% !important;
    object-fit: contain !important;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar toggle button (JS injection) ────────────────────────────────────
st.markdown("""
<script>
(function() {
    function injectToggle() {
        const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        const existing = window.parent.document.getElementById('sidebar-toggle-btn');
        if (existing) return;

        const btn = window.parent.document.createElement('button');
        btn.id = 'sidebar-toggle-btn';
        btn.title = 'Toggle sidebar';

        let collapsed = false;

        function updateBtn() {
            btn.textContent = collapsed ? '▶' : '◀';
            btn.style.left = collapsed ? '0px' : (sidebar ? sidebar.offsetWidth - 1 + 'px' : '240px');
        }

        btn.addEventListener('click', function() {
            // Click the native Streamlit collapse button
            const nativeBtn = window.parent.document.querySelector('[data-testid="collapsedControl"]') ||
                               window.parent.document.querySelector('button[kind="headerNoPadding"]') ||
                               window.parent.document.querySelector('[data-testid="stSidebarNavCollapseButton"]');
            if (nativeBtn) {
                nativeBtn.click();
                collapsed = !collapsed;
                updateBtn();
            } else {
                // Fallback: toggle sidebar visibility directly
                if (sidebar) {
                    collapsed = !collapsed;
                    sidebar.style.transform = collapsed ? 'translateX(-100%)' : 'translateX(0)';
                    sidebar.style.transition = 'transform 0.3s ease';
                    updateBtn();
                }
            }
        });

        updateBtn();
        window.parent.document.body.appendChild(btn);
    }
    // Try immediately and after DOM is ready
    if (window.parent.document.readyState === 'complete') {
        injectToggle();
    } else {
        window.parent.document.addEventListener('DOMContentLoaded', injectToggle);
    }
    setTimeout(injectToggle, 800);
})();
</script>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "bone fracture":      (231, 76, 60),
    "bone injury":        (230, 126, 34),
    "bone abnormality":   (241, 196, 15),
    "periosteal reaction":(155, 89, 182),
    "foreign object":     (52, 152, 219),
    "metal":              (142, 68, 173),
    "pronator sign":      (26, 188, 156),
    "soft tissue":        (46, 204, 113),
    "text":               (149, 165, 166),
}

IGNORE_CLASSES = {"text"}
FRACTURE_CLASSES = {"bone fracture","bone injury","bone abnormality","bonelesion","fracture"}

def get_class_color(name):
    return CLASS_COLORS.get(name.lower(), (52, 152, 219))

def get_badge_class(conf):
    if conf >= 0.7: return "badge-high", "HIGH"
    if conf >= 0.5: return "badge-med",  "MED"
    return "badge-low", "LOW"

def get_card_class(name):
    n = name.lower()
    if any(k in n for k in ["fracture","injury","abnormality"]): return "fracture"
    if "bone" in n: return "bone"
    if "metal" in n: return "metal"
    return "other"

def draw_detections(img_bgr, detections, ignored_classes=IGNORE_CLASSES):
    """Draw bounding boxes on image."""
    img = img_bgr.copy()
    h, w = img.shape[:2]

    for det in detections:
        if det["class"].lower() in ignored_classes:
            continue
        x1, y1, x2, y2 = det["box"]
        col = get_class_color(det["class"])
        bgr = (col[2], col[1], col[0])
        conf = det["confidence"]

        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)

        # Label background
        label = f"{det['class'].title()}  {conf:.0%}"
        font_scale = max(0.45, min(0.7, w / 1200))
        thickness = 1 if w < 800 else 2
        (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        pad = 5
        y_label = max(y1 - lh - pad*2, 0)
        cv2.rectangle(img, (x1, y_label), (x1 + lw + pad*2, y_label + lh + pad*2), bgr, -1)
        cv2.putText(img, label, (x1 + pad, y_label + lh + pad),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img

@st.cache_resource(show_spinner=False)
def load_model(model_path_str):
    try:
        from ultralytics import YOLO
        model = YOLO(model_path_str)
        return model, None
    except Exception as e:
        return None, str(e)

def run_inference(model, image_pil, conf, iou):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arr = np.array(image_pil)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr)
        arr = arr.astype(np.uint8)
    img_np = Image.fromarray(arr).convert("RGB")
    img_np = np.array(img_np)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img_bgr)

    t0 = time.perf_counter()
    results = model(tmp_path, conf=conf, iou=iou, device=device, verbose=False)
    elapsed = time.perf_counter() - t0
    os.unlink(tmp_path)

    result = results[0]
    boxes = result.boxes

    detections = []
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id]
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "class": class_name,
                "confidence": conf_score,
                "box": (x1, y1, x2, y2),
                "center": ((x1+x2)//2, (y1+y2)//2),
                "width": x2 - x1,
                "height": y2 - y1,
            })

    return detections, elapsed, img_bgr


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div style="font-size:1.6rem; font-weight:700; color:#fff; letter-spacing:-0.5px;">
            🦴 BoneVision
        </div>
        <div style="font-size:0.78rem; color:#4a7a9b; margin-top:0.2rem;">
            AI-Powered Fracture Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Model Configuration</div>', unsafe_allow_html=True)

    # ── Option A: upload the .pt directly ──
    st.markdown("**Option A — Upload model file**")
    uploaded_model = st.file_uploader(
        "Drop your .pt file here", type=["pt"], key="model_uploader"
    )
    if uploaded_model is not None:
        model_tmp_dir = Path(tempfile.gettempdir()) / "bonevision_model"
        model_tmp_dir.mkdir(exist_ok=True)
        model_tmp_path = model_tmp_dir / uploaded_model.name
        model_tmp_path.write_bytes(uploaded_model.getvalue())
        st.session_state["resolved_model_path"] = str(model_tmp_path)
        st.markdown(
            f'<div style="font-size:0.75rem;color:#27ae60;margin-top:0.2rem;">' +
            f'✅ {uploaded_model.name} ready</div>',
            unsafe_allow_html=True
        )

    # ── Option B: type the full path ──
    st.markdown("**Option B — Paste full path**")
    typed_path = st.text_input(
        "Full path to .pt file",
        value=st.session_state.get("resolved_model_path", r"C:\Users\N. K. Hazarika\Downloads\best (3).pt"),
        placeholder=r"C:\Users\Name\Downloads\best (3).pt",
        label_visibility="collapsed",
        help=(
            "Example:  C:\\Users\\N. K. Hazarika\\Downloads\\best (3).pt\n"
            "Spaces and brackets in the filename are fine — paste the full path."
        )
    )
    if typed_path:
        st.session_state["resolved_model_path"] = typed_path

    model_path_input = st.session_state.get("resolved_model_path", r"C:\Users\N. K. Hazarika\Downloads\best (3).pt")

    model_variant = st.selectbox(
        "Model Variant",
        ["Custom (.pt file above)", "YOLOv8n (Nano)", "YOLOv8s (Small)", "YOLOv8m (Medium)"],
        index=0
    )

    st.markdown('<div class="sidebar-section">Detection Settings</div>', unsafe_allow_html=True)

    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.10, max_value=0.95,
        value=0.30, step=0.05,
        help="Lower = more detections (more false positives). Recommended: 0.25–0.40 for medical imaging."
    )

    iou_thresh = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.10, max_value=0.90,
        value=0.45, step=0.05,
        help="Controls overlap suppression. Higher = more overlapping boxes kept."
    )

    show_ignored = st.checkbox("Show 'Text' annotations", value=False)

    st.markdown('<div class="sidebar-section">Display</div>', unsafe_allow_html=True)

    box_thickness = st.slider("Box thickness", 1, 5, 2)
    show_all_tab = st.checkbox("Show all detection details", value=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#4a7a9b; line-height:1.7;">
        <b style="color:#7a9ab5;">Classes:</b><br>
        Bone Fracture · Bone Injury<br>
        Bone Abnormality · Metal<br>
        Periosteal Reaction · Soft Tissue<br>
        Foreign Object · Pronator Sign
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="icon">🩻</div>
    <div>
        <h1>Bone Fracture Detection</h1>
        <p>Upload an X-ray image and run YOLO-based fracture detection analysis</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Main Layout ───────────────────────────────────────────────────────────────
col_upload, col_results = st.columns([1.1, 1], gap="medium")

with col_upload:
    st.markdown('<div class="card-title">📂 Input Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        label_visibility="collapsed"
    )

    # Model status check
    model_loaded = Path(model_path_input).exists() if model_path_input else False

    if not model_loaded:
        st.markdown(f"""
        <div class="status-bar status-warning">
            ⚠️  Model not found at <code>{model_path_input}</code> — update path in sidebar
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-bar status-ready">✅ Model file found — ready to analyze</div>',
                    unsafe_allow_html=True)

    # Preview
    if uploaded_file:
        img_pil = Image.open(uploaded_file)
        # Normalize 16-bit (or other high-depth) images to 8-bit RGB
        if img_pil.mode not in ("RGB", "RGBA", "L"):
            import numpy as np
            arr = np.array(img_pil).astype(np.float32)
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
            else:
                arr = np.zeros_like(arr)
            img_pil = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        w, h = img_pil.size
        # Cap preview height to keep it compact
        MAX_H = 320
        if h > MAX_H:
            ratio = MAX_H / h
            img_preview = img_pil.resize((int(w*ratio), MAX_H), Image.LANCZOS)
        else:
            img_preview = img_pil
        st.image(img_preview, use_container_width=False, width=None, caption="")
        st.markdown(f'<div class="img-caption">{uploaded_file.name} &nbsp;·&nbsp; {w}×{h}px &nbsp;·&nbsp; {uploaded_file.size//1024} KB</div>',
                    unsafe_allow_html=True)

    st.markdown("")
    analyze_btn = st.button("🔍 Analyze Image", use_container_width=True,
                            disabled=(uploaded_file is None or not model_loaded))


with col_results:
    st.markdown('<div class="card-title">📊 Detection Results</div>', unsafe_allow_html=True)

    if "results_data" not in st.session_state:
        st.session_state.results_data = None

    # ── Run inference ─────────────────────────────────────────────────────
    if analyze_btn and uploaded_file and model_loaded:
        with st.spinner("Loading model and running inference…"):
            model, model_err = load_model(model_path_input)

        if model_err:
            st.markdown(f'<div class="status-bar status-error">❌ {model_err}</div>',
                        unsafe_allow_html=True)
        else:
            img_pil = Image.open(uploaded_file)
            prog = st.progress(0, text="Running YOLO inference…")
            time.sleep(0.1)
            prog.progress(40, text="Processing detections…")

            detections, elapsed, img_bgr = run_inference(model, img_pil, conf_thresh, iou_thresh)

            prog.progress(80, text="Rendering results…")
            time.sleep(0.1)
            prog.progress(100, text="Done!")
            time.sleep(0.3)
            prog.empty()

            st.session_state.results_data = {
                "detections": detections,
                "elapsed": elapsed,
                "img_bgr": img_bgr,
                "conf": conf_thresh,
                "iou": iou_thresh,
                "filename": uploaded_file.name,
            }

    # ── Display results ───────────────────────────────────────────────────
    rd = st.session_state.results_data

    if rd is None:
        st.markdown("""
        <div class="no-detection">
            <div class="nd-icon">🩻</div>
            <div class="nd-title">No analysis yet</div>
            <div class="nd-sub">Upload an X-ray and click <b>Analyze Image</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        dets = rd["detections"]
        active_dets = [d for d in dets if show_ignored or d["class"].lower() not in IGNORE_CLASSES]
        fracture_dets = [d for d in active_dets if d["class"].lower() in FRACTURE_CLASSES]
        other_dets   = [d for d in active_dets if d["class"].lower() not in FRACTURE_CLASSES
                        and d["class"].lower() not in IGNORE_CLASSES]
        ignored = [d for d in dets if d["class"].lower() in IGNORE_CLASSES]

        # ── Metric row ──
        avg_conf = (sum(d["confidence"] for d in active_dets) / len(active_dets)
                    if active_dets else 0)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card {'highlight' if fracture_dets else 'safe'}">
                <div class="value">{len(fracture_dets)}</div>
                <div class="label">Fractures</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card info">
                <div class="value">{len(other_dets)}</div>
                <div class="label">Other Findings</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card info">
                <div class="value">{avg_conf:.0%}</div>
                <div class="label">Avg Confidence</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{rd['elapsed']*1000:.0f}<span style="font-size:1rem">ms</span></div>
                <div class="label">Inference Time</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Annotated image ──
        img_ann = draw_detections(rd["img_bgr"], active_dets,
                                   ignored_classes=IGNORE_CLASSES if not show_ignored else set())
        img_rgb = cv2.cvtColor(img_ann, cv2.COLOR_BGR2RGB)
        # Cap result image height
        MAX_H = 320
        ah, aw = img_rgb.shape[:2]
        if ah > MAX_H:
            ratio = MAX_H / ah
            img_rgb_disp = cv2.resize(img_rgb, (int(aw*ratio), MAX_H), interpolation=cv2.INTER_AREA)
        else:
            img_rgb_disp = img_rgb
        st.image(img_rgb_disp, use_container_width=False, width=None)

        # ── Download ──
        buf = io.BytesIO()
        Image.fromarray(img_rgb).save(buf, format="JPEG", quality=95)
        st.download_button("⬇️ Download Annotated Image", buf.getvalue(),
                           file_name=f"annotated_{rd['filename']}",
                           mime="image/jpeg", use_container_width=True)


# ── Detection details ─────────────────────────────────────────────────────────
if rd is not None and show_all_tab:
    st.markdown("---")
    st.markdown('<div class="card-title">🔬 Detection Details</div>', unsafe_allow_html=True)

    all_active = [d for d in rd["detections"]
                  if show_ignored or d["class"].lower() not in IGNORE_CLASSES]
    all_active.sort(key=lambda x: x["confidence"], reverse=True)

    if not all_active:
        st.markdown("""
        <div class="no-detection">
            <div class="nd-icon">✅</div>
            <div class="nd-title">No detections above threshold</div>
            <div class="nd-sub">Try lowering the confidence threshold in the sidebar</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        cols = st.columns(min(len(all_active), 3))
        for i, det in enumerate(all_active):
            col = cols[i % len(cols)]
            with col:
                badge_cls, badge_lbl = get_badge_class(det["confidence"])
                card_cls = get_card_class(det["class"])
                x1, y1, x2, y2 = det["box"]
                cx, cy = det["center"]
                st.markdown(f"""
                <div class="detection-item {card_cls}">
                    <div class="det-header">
                        <span class="det-class">{det['class'].title()}</span>
                        <span class="badge {badge_cls}">{badge_lbl} {det['confidence']:.1%}</span>
                    </div>
                    <div class="det-coords">
                        📍 TL ({x1}, {y1}) · BR ({x2}, {y2})<br>
                        ⊙ Center ({cx}, {cy}) &nbsp;·&nbsp; {det['width']}×{det['height']}px
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Ignored count
    ignored = [d for d in rd["detections"] if d["class"].lower() in IGNORE_CLASSES]
    if ignored and not show_ignored:
        st.markdown(f"""
        <div style="font-size:0.8rem; color:#7a9ab5; margin-top:0.5rem;">
            🚫 {len(ignored)} text annotation(s) hidden — enable in sidebar to show
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size:0.75rem; color:#7a9ab5; padding:0.5rem 0 1rem;">
    BoneVision AI &nbsp;·&nbsp; YOLOv8 Bone Fracture Detection
    &nbsp;·&nbsp; <em>For research use only — not a medical device</em>
</div>
""", unsafe_allow_html=True)


