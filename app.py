"""
Detection Testing Dashboard (local, presentation-only).
ROI occupancy logic + object detection. Runs on laptop, independent of the
production backend / edge device. Detection targets are referenced by numeric
class id only — no textual class labels are stored or displayed.
"""

import streamlit as st
import cv2
import numpy as np
import json
import time
import re
import tempfile
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ROI_DIR = BASE_DIR / "assets" / "roi"
SAMPLES_DIR = BASE_DIR / "assets" / "samples"

# ──────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIGURATION & CUSTOM THEME (PREMIUM DARK MODE)
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Detection Testing Dashboard",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background-color: #0b0c10;
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #c5c6c7;
    }

    header[data-testid="stHeader"] {
        background: rgba(11, 12, 16, 0.8);
        backdrop-filter: blur(8px);
    }

    section[data-testid="stSidebar"] {
        background-color: #12131c !important;
        border-right: 1px solid #1f2833;
    }

    .header-card {
        background: linear-gradient(135deg, #1f1f2e 0%, #151522 100%);
        border: 1px solid #2d2d44;
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }

    .header-title {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #6366f1 0%, #a5b4fc 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0 0 10px 0;
        letter-spacing: -0.5px;
    }

    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin: 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #181922 0%, #1f202e 100%);
        border: 1px solid #2d2d44;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        margin: 0px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.15);
        border-color: #4f46e5;
    }

    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.1;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-top: 6px;
    }

    .val-occupied { color: #f43f5e; text-shadow: 0 0 10px rgba(244, 63, 94, 0.2); }
    .val-empty { color: #10b981; text-shadow: 0 0 10px rgba(16, 185, 129, 0.2); }
    .val-total { color: #3b82f6; text-shadow: 0 0 10px rgba(59, 130, 246, 0.2); }
    .val-rate { color: #f59e0b; text-shadow: 0 0 10px rgba(245, 158, 11, 0.2); }

    .slot-card {
        border-radius: 10px;
        padding: 14px 10px;
        text-align: center;
        font-weight: 700;
        font-size: 0.9rem;
        transition: all 0.2s;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        border: 1px solid transparent;
        margin-bottom: 12px;
    }

    .slot-card-occupied {
        background: rgba(244, 63, 94, 0.08);
        border-color: rgba(244, 63, 94, 0.3);
        color: #f43f5e;
    }

    .slot-card-occupied:hover {
        background: rgba(244, 63, 94, 0.15);
        border-color: #f43f5e;
        box-shadow: 0 0 15px rgba(244, 63, 94, 0.3);
    }

    .slot-card-empty {
        background: rgba(16, 185, 129, 0.08);
        border-color: rgba(16, 185, 129, 0.3);
        color: #10b981;
    }

    .slot-card-empty:hover {
        background: rgba(16, 185, 129, 0.15);
        border-color: #10b981;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
    }

    .slot-num {
        font-size: 0.7rem;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 1px;
        margin-top: 4px;
        font-weight: 500;
    }

    .slot-icon {
        font-size: 1.3rem;
        margin-bottom: 2px;
        display: block;
    }

    .section-title {
        color: #f1f5f9;
        font-size: 1.1rem;
        font-weight: 700;
        border-left: 4px solid #6366f1;
        padding-left: 12px;
        margin: 24px 0 16px 0;
        font-family: 'Outfit', sans-serif;
    }

    .alert-box {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 10px;
        padding: 15px;
        color: #e2e8f0;
        margin: 10px 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #12131c;
        border: 1px solid #2d2d44;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        color: #94a3b8;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
        color: white !important;
        border-color: #4f46e5 !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# 2. UTILITY FUNCTIONS (ROI LOADING, MODEL SCANNING, IMAGE DRAWING)
# ──────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_yolo_model(model_path: str):
    """Load detection model with caching for fast reload."""
    try:
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


def scan_best_models():
    """Discover trained model weights bundled in the local models/ directory."""
    models = {}
    for ext in ("*.pt", "*.onnx"):
        for weight in sorted(MODELS_DIR.glob(ext)):
            models[f"{weight.name}"] = str(weight.resolve())
    if not models:
        models["No model found! Place best.pt in models/"] = str(MODELS_DIR / "best.pt")
    return models


def find_available_rois():
    """Discover ROI coordinate .txt files in assets/roi dynamically."""
    rois = {}
    if ROI_DIR.exists():
        files = sorted(ROI_DIR.iterdir(), key=lambda f: (0 if "new-json-parking" in f.name else 1, f.name))
        for file in files:
            if file.is_file() and file.suffix == ".txt":
                if "coordinates_numpyformat" in file.name:
                    label = f"Numpy Format ({file.name})"
                elif "json-dataset" in file.name:
                    label = f"JSON Dataset Format ({file.name})"
                elif "new-json-parking" in file.name:
                    label = f"JSON Parking Format ({file.name})"
                elif "json-point" in file.name:
                    label = f"JSON Point Format ({file.name})"
                else:
                    label = f"Custom ROI ({file.name})"
                rois[label] = str(file.resolve())
    if not rois:
        rois["JSON Parking Format (new-json-parking.txt)"] = str(ROI_DIR / "new-json-parking.txt")
    return rois


def load_roi_polygons(file_path: Path):
    """
    Robust ROI coordinate parser. Supports JSON-like and numpy-array formats.
    Falls back to a regex digit extractor so it never crashes on odd formats.
    """
    if not file_path.exists():
        return None

    with open(file_path, "r") as f:
        raw_content = f.read().strip()

    # Method 1: JSON-like parsing (curly braces -> brackets)
    try:
        processed = raw_content
        if processed.startswith("{") and processed.endswith("}"):
            processed = "[" + processed[1:-1] + "]"
        data = json.loads(processed)
        polygons = []
        for poly in data:
            pts = []
            for pt in poly:
                if isinstance(pt, dict):
                    pts.append([float(pt.get("x", pt.get("X", 0))), float(pt.get("y", pt.get("Y", 0)))])
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    pts.append([float(pt[0]), float(pt[1])])
            if len(pts) >= 3:
                polygons.append(pts)
        if len(polygons) > 0:
            return polygons
    except Exception:
        pass

    # Method 2: Unified regex digit extractor (fully robust)
    polygons = []
    try:
        lines = raw_content.split("\n")
        for line in lines:
            line = line.strip()
            digits = re.findall(r"\d+", line)
            if len(digits) >= 6:
                pts = []
                for i in range(0, len(digits), 2):
                    if i + 1 < len(digits):
                        pts.append([int(digits[i]), int(digits[i + 1])])
                if len(pts) >= 3:
                    polygons.append(pts)
        if len(polygons) > 0:
            return polygons
    except Exception as e:
        st.error(f"Coordinate regex parse error: {e}")

    return None


def save_upload_to_temp(uploaded_file, suffix=None):
    """Save an uploaded streamlit file to a temporary local path."""
    uploaded_file.seek(0)
    name = getattr(uploaded_file, "name", "upload")
    suffix = suffix or Path(name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = Path(tmp_file.name)
    uploaded_file.seek(0)
    return tmp_path


# Standard reference resolutions where pixel-absolute ROI are usually annotated.
# Ascending, so coordinates snap to the smallest resolution that contains them.
_STANDARD_REFERENCE_RES = [
    (716, 400),
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]


def get_roi_reference_dimensions(roi_polygons):
    max_roi_x = 0
    max_roi_y = 0
    for polygon in roi_polygons:
        for pt in polygon:
            if pt[0] > max_roi_x:
                max_roi_x = pt[0]
            if pt[1] > max_roi_y:
                max_roi_y = pt[1]

    if max_roi_x <= 1.0 and max_roi_y <= 1.0:
        return 1.0, 1.0

    for ref_w, ref_h in _STANDARD_REFERENCE_RES:
        if max_roi_x <= ref_w and max_roi_y <= ref_h:
            return float(ref_w), float(ref_h)

    return float(max_roi_x), float(max_roi_y)


def scale_roi_polygons_for_frame(roi_polygons, frame_w, frame_h):
    ref_w, ref_h = get_roi_reference_dimensions(roi_polygons)
    scale_x = frame_w / ref_w
    scale_y = frame_h / ref_h
    scaled_polygons = []
    for polygon in roi_polygons:
        scaled_polygons.append([[int(pt[0] * scale_x), int(pt[1] * scale_y)] for pt in polygon])
    return scaled_polygons


def process_video_file(input_path, output_path, model, roi_polygons, conf_threshold, imgsz_val, overlay_alpha, show_bbox, progress_bar=None):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        total_frames = 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    scaled_roi_polygons = scale_roi_polygons_for_frame(roi_polygons, width, height)

    processed_frames = 0
    occupied_counts = []
    empty_counts = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, imgsz=imgsz_val, conf=conf_threshold, verbose=False)
        slot_status = run_occupancy_check(results, scaled_roi_polygons, conf_threshold)
        overlay_frame = draw_visual_overlay(
            frame,
            scaled_roi_polygons,
            slot_status,
            show_bbox=show_bbox,
            results=results,
            overlay_alpha=overlay_alpha,
        )

        writer.write(overlay_frame)
        processed_frames += 1
        occupied_counts.append(slot_status.count("occupied"))
        empty_counts.append(slot_status.count("empty"))

        if progress_bar is not None:
            progress_bar.progress(min(processed_frames / total_frames, 1.0))

    writer.release()
    cap.release()

    if processed_frames == 0:
        return None, None

    stats = {
        "processed_frames": processed_frames,
        "fps": fps,
        "avg_occupied": sum(occupied_counts) / processed_frames,
        "avg_empty": sum(empty_counts) / processed_frames,
        "min_occupied": min(occupied_counts),
        "max_occupied": max(occupied_counts),
        "min_empty": min(empty_counts),
        "max_empty": max(empty_counts),
    }
    return output_path, stats


def _compute_overlap_ratio(poly_arr, x1, y1, x2, y2, img_h, img_w):
    """
    Intersection ratio between a bbox and an ROI polygon.
    Method: pixel mask over the bbox crop — efficient for high-res images.
    Returns intersection_pixels / bbox_pixels (0.0–1.0).
    """
    bx1, by1 = max(0, int(x1)), max(0, int(y1))
    bx2, by2 = min(img_w, int(x2)), min(img_h, int(y2))
    if bx2 <= bx1 or by2 <= by1:
        return 0.0
    crop_h, crop_w = by2 - by1, bx2 - bx1
    shifted = (poly_arr - np.array([[bx1, by1]])).astype(np.int32)
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted], 1)
    bbox_area = float(crop_h * crop_w)
    return float(mask.sum()) / bbox_area if bbox_area > 0 else 0.0


# ── Target class configuration (numeric ids only — no textual labels) ────────
# Ids that mark a slot as occupied. Model-specific; edit if weights differ.
_TARGET_IDS = frozenset({2, 5, 7, 8, 67})
# Minimum overlap (bbox ratio) for a detection to be assigned to a slot.
_MIN_OVERLAP = 0.10


def _resolve_class_partitions(class_names):
    """
    Derive (target_ids, empty_ids) from the loaded model's class map using
    numeric ids only. Never inspects textual class names.
    """
    if not class_names:
        return set(_TARGET_IDS), set()

    target_ids = {cid for cid in class_names if cid in _TARGET_IDS}
    empty_ids = set()

    # Two-class custom weights (index 0 = empty, index 1 = target).
    if not target_ids and len(class_names) == 2:
        empty_ids = {0}
        target_ids = {1}

    return target_ids, empty_ids


def run_occupancy_check(results, roi_polygons, conf_threshold):
    """
    Classify each ROI slot from detection output.

    1. Filter detections by numeric target class id.
    2. Exclusive assignment: each detection fills only ONE slot (largest overlap),
       so a single object cannot mark two adjacent slots.
    """
    status = ["empty"] * len(roi_polygons)
    bboxes = results[0].boxes

    if bboxes is None or len(bboxes) == 0:
        return status

    class_names = results[0].names
    img_h, img_w = results[0].orig_shape

    target_ids, empty_ids = _resolve_class_partitions(class_names)

    valid_boxes = []
    for box in bboxes:
        conf = float(box.conf.cpu().numpy()[0])
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls.cpu().numpy()[0])
        if cls_id in empty_ids:
            continue
        if target_ids and cls_id not in target_ids:
            continue
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
        valid_boxes.append((cls_id, conf, float(x1), float(y1), float(x2), float(y2)))

    if not valid_boxes:
        return status

    poly_arrays = [np.array(p, dtype=np.int32) for p in roi_polygons]

    for (cls_id, conf, x1, y1, x2, y2) in valid_boxes:
        best_slot = -1
        best_ratio = _MIN_OVERLAP

        for slot_idx, poly_arr in enumerate(poly_arrays):
            ratio = _compute_overlap_ratio(poly_arr, x1, y1, x2, y2, img_h, img_w)
            if ratio > best_ratio:
                best_ratio = ratio
                best_slot = slot_idx

        if best_slot >= 0:
            status[best_slot] = "occupied"

    return status


def draw_visual_overlay(image_bgr, roi_polygons, slot_status, show_bbox=True, results=None, overlay_alpha=0.3):
    """
    Draw visual overlay. Neon green = empty, neon red = occupied.
    Detection boxes are labelled generically (no class-name text).
    """
    overlay = image_bgr.copy()
    target_ids = set(_TARGET_IDS)
    if results is not None:
        target_ids, _ = _resolve_class_partitions(results[0].names)

    # 1. Detection bounding boxes (optional)
    if show_bbox and results is not None:
        bboxes = results[0].boxes
        if bboxes is not None:
            for box in bboxes:
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
                cls_id = int(box.cls.cpu().numpy()[0])

                # Color by numeric membership only: red = target, cyan = other.
                color = (50, 50, 240) if cls_id in target_ids else (0, 229, 255)

                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
                label = f"obj {conf:.2f}"
                cv2.putText(overlay, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 2. ROI polygon per slot
    for idx, (polygon, status) in enumerate(zip(roi_polygons, slot_status)):
        poly_arr = np.array(polygon, dtype=np.int32)
        color = (50, 50, 240) if status == "occupied" else (50, 200, 50)

        mask = overlay.copy()
        cv2.fillPoly(mask, [poly_arr], color)
        cv2.addWeighted(mask, overlay_alpha, overlay, 1 - overlay_alpha, 0, overlay)

        cv2.polylines(overlay, [poly_arr], isClosed=True, color=color, thickness=2)

        cx = int(poly_arr[:, 0].mean())
        cy = int(poly_arr[:, 1].mean())

        cv2.rectangle(overlay, (cx - 15, cy - 10), (cx + 15, cy + 8), (15, 15, 25), -1)
        cv2.rectangle(overlay, (cx - 15, cy - 10), (cx + 15, cy + 8), color, 1)

        label = f"S{idx + 1}"
        cv2.putText(overlay, label, (cx - 11, cy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


# ──────────────────────────────────────────────────────────────────────────
# 3. SIDEBAR CONTROLS
# ──────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("<h2 style='text-align: center; color: #6366f1; font-family: Outfit;'>⚙️ TEST PARAMETERS</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown("<div class='section-title'>1. SELECT MODEL</div>", unsafe_allow_html=True)
available_models = scan_best_models()
selected_model_name = st.sidebar.selectbox(
    "Model weights to test:",
    options=list(available_models.keys()),
    index=0
)
model_path = available_models[selected_model_name]

model, load_err = load_yolo_model(model_path)

if load_err:
    st.sidebar.error(f"❌ Failed to load model: {load_err}")
else:
    st.sidebar.success(f"✔️ Model Loaded: {Path(model_path).name}")

st.sidebar.markdown("<div class='section-title'>2. ROI COORDINATE FILE</div>", unsafe_allow_html=True)
available_rois = find_available_rois()
selected_roi_label = st.sidebar.selectbox(
    "Slot coordinate file:",
    options=list(available_rois.keys()),
    index=0
)
roi_file_path = Path(available_rois[selected_roi_label])

roi_polygons = load_roi_polygons(roi_file_path)

if roi_polygons:
    st.sidebar.success(f"✔️ Loaded {len(roi_polygons)} ROI slots")
else:
    st.sidebar.error("❌ Failed to load ROI coordinates. Check the file.")

st.sidebar.markdown("<div class='section-title'>3. INFERENCE SETTINGS</div>", unsafe_allow_html=True)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
imgsz_val = st.sidebar.selectbox(
    "Inference Resolution (imgsz)",
    options=[320, 640, 960, 1280, 1536, 1920],
    index=1,
    help="Input resolution fed to the model. Raise it (1280/1536) for high-res images to sharpen small-object detection."
)
overlay_alpha = st.sidebar.slider("ROI Overlay Transparency", 0.1, 0.8, 0.3, 0.05)
show_bbox = st.sidebar.checkbox("Show Detection Bounding Boxes", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align: center; font-size: 0.8rem; color: #64748b;'>"
    "Detection Testing Tool<br>ROI Logic + Object Detection<br><b>Local / Presentation</b>"
    "</div>",
    unsafe_allow_html=True
)


# ──────────────────────────────────────────────────────────────────────────
# 4. MAIN CONTENT
# ──────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-card">
    <h1 class="header-title">🅿️ DETECTION TESTING DASHBOARD</h1>
    <p class="header-subtitle">Upload an image or video to verify slot-occupancy detection using ROI perspective logic.</p>
</div>
""", unsafe_allow_html=True)

tab_detection, tab_roi_detail = st.tabs(["📊 DETECTION DASHBOARD", "📋 ROI COORDINATE DETAIL"])

# ==========================================================================
# TAB 1: DETECTION DASHBOARD
# ==========================================================================
with tab_detection:
    if model is None:
        st.error("⚠️ Model not loaded. Check the model weights in the sidebar.")
        st.stop()

    if not roi_polygons:
        st.error("⚠️ ROI slot coordinates not loaded. Check your ROI file.")
        st.stop()

    st.markdown("### 📷 Choose a test image or video")

    img_select_option = st.radio(
        "Use a bundled sample or upload your own file:",
        ["Sample A (assets/samples/image.png)",
         "Sample B (assets/samples/dataset.png)",
         "Upload Image (.png, .jpg, .jpeg)",
         "Upload Video (.mp4, .mov, .avi, .mkv)"],
        horizontal=True
    )

    test_img_path = None
    uploaded_image = None
    uploaded_video = None
    video_input_path = None
    video_output_path = None
    video_stats = None

    if img_select_option == "Sample A (assets/samples/image.png)":
        test_img_path = SAMPLES_DIR / "image.png"
    elif img_select_option == "Sample B (assets/samples/dataset.png)":
        test_img_path = SAMPLES_DIR / "dataset.png"
    elif img_select_option == "Upload Video (.mp4, .mov, .avi, .mkv)":
        uploaded_video = st.file_uploader("Choose a video:", type=["mp4", "mov", "avi", "mkv"])
    else:
        uploaded_image = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])

    img_bgr = None
    if uploaded_image is not None:
        file_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif uploaded_video is not None:
        if uploaded_video.size > 0:
            video_input_path = save_upload_to_temp(uploaded_video)
            st.markdown("### 🎬 Input Video")
            st.video(str(video_input_path))
            st.info("⏱️ Processing the full video frame by frame for slot detection.")
            with st.spinner("Running inference on every video frame..."):
                video_output_path = Path(tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name)
                video_output_path, video_stats = process_video_file(
                    video_input_path,
                    video_output_path,
                    model,
                    roi_polygons,
                    conf_threshold,
                    imgsz_val,
                    overlay_alpha,
                    show_bbox,
                    progress_bar=st.progress(0),
                )

            if video_output_path is not None:
                st.markdown("### 🎞️ Detection Result Video")
                st.video(str(video_output_path))
                st.markdown(
                    f"<div class='alert-box'>"
                    f"🎯 Processed <b>{video_stats['processed_frames']}</b> frames.<br>"
                    f"• Avg occupied slots: <b>{video_stats['avg_occupied']:.1f}</b><br>"
                    f"• Avg empty slots: <b>{video_stats['avg_empty']:.1f}</b><br>"
                    f"• Occupied range: <b>{video_stats['min_occupied']}–{video_stats['max_occupied']}</b> slots</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.error("⚠️ Failed to process the video. Make sure the file is valid and retry.")

        else:
            st.warning("⚠️ Video file missing or empty.")
    elif test_img_path and test_img_path.exists():
        img_bgr = cv2.imread(str(test_img_path))
    else:
        st.info("📂 Upload your own image or video to start testing.")

    if img_bgr is not None:
        h_curr, w_curr = img_bgr.shape[:2]

        max_roi_x = max((pt[0] for polygon in roi_polygons for pt in polygon), default=0)
        max_roi_y = max((pt[1] for polygon in roi_polygons for pt in polygon), default=0)
        ref_w, ref_h = get_roi_reference_dimensions(roi_polygons)

        scale_x = w_curr / ref_w
        scale_y = h_curr / ref_h

        scaled_roi_polygons = []
        for polygon in roi_polygons:
            scaled_poly = []
            for pt in polygon:
                scaled_x = int(pt[0] * scale_x)
                scaled_y = int(pt[1] * scale_y)
                scaled_poly.append([scaled_x, scaled_y])
            scaled_roi_polygons.append(scaled_poly)

        t_start = time.perf_counter()
        results = model(img_bgr, imgsz=imgsz_val, conf=conf_threshold, verbose=False)
        t_inf = (time.perf_counter() - t_start) * 1000

        slot_status = run_occupancy_check(results, scaled_roi_polygons, conf_threshold)

        n_occupied = slot_status.count("occupied")
        n_empty = slot_status.count("empty")
        n_total = len(slot_status)
        occupancy_rate = (n_occupied / n_total * 100) if n_total > 0 else 0

        overlay_bgr = draw_visual_overlay(
            img_bgr,
            scaled_roi_polygons,
            slot_status,
            show_bbox=show_bbox,
            results=results,
            overlay_alpha=overlay_alpha
        )

        img_original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<p class="metric-value val-occupied">{n_occupied}</p>'
                f'<p class="metric-label">Occupied Slots</p></div>',
                unsafe_allow_html=True
            )
        with m_col2:
            st.markdown(
                f'<div class="metric-card">'
                f'<p class="metric-value val-empty">{n_empty}</p>'
                f'<p class="metric-label">Empty Slots</p></div>',
                unsafe_allow_html=True
            )
        with m_col3:
            st.markdown(
                f'<div class="metric-card">'
                f'<p class="metric-value val-total">{n_total}</p>'
                f'<p class="metric-label">Total Capacity</p></div>',
                unsafe_allow_html=True
            )
        with m_col4:
            st.markdown(
                f'<div class="metric-card">'
                f'<p class="metric-value val-rate">{occupancy_rate:.0f}%</p>'
                f'<p class="metric-label">Occupancy Rate</p></div>',
                unsafe_allow_html=True
            )

        st.markdown("<div class='section-title'>🖼️ VISUAL DETECTION ANALYSIS</div>", unsafe_allow_html=True)
        col_img_left, col_img_right = st.columns(2)

        with col_img_left:
            st.markdown("<p style='text-align: center; font-weight:600; color:#94a3b8;'>Original Input</p>", unsafe_allow_html=True)
            st.image(img_original_rgb, use_container_width=True)

        with col_img_right:
            st.markdown("<p style='text-align: center; font-weight:600; color:#f1f5f9;'>Detection + ROI Perspective Overlay</p>", unsafe_allow_html=True)
            st.image(img_overlay_rgb, use_container_width=True)

        st.markdown("<div class='section-title'>🚦 SLOT STATUS GRID</div>", unsafe_allow_html=True)

        n_cols = 8
        grid_cols = st.columns(n_cols)

        for i, status in enumerate(slot_status):
            col_idx = i % n_cols
            with grid_cols[col_idx]:
                if status == "occupied":
                    card_class = "slot-card-occupied"
                    icon = "🔴"
                    badge_status = "OCCUPIED"
                else:
                    card_class = "slot-card-empty"
                    icon = "✔️"
                    badge_status = "EMPTY"

                st.markdown(
                    f'<div class="slot-card {card_class}">'
                    f'<span class="slot-icon">{icon}</span>'
                    f'{badge_status}'
                    f'<div class="slot-num">Slot {i+1}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        device_name = "GPU (CUDA)" if results[0].boxes is not None and next(model.parameters()).is_cuda else "CPU"

        st.markdown("---")
        st.markdown(
            f'<div class="alert-box">'
            f'⚡ <b>INFERENCE METADATA & PERFORMANCE:</b><br>'
            f'• Inference Latency: <b>{t_inf:.2f} ms</b> ({1000/t_inf:.1f} FPS)<br>'
            f'• Compute Device: <b>{device_name}</b><br>'
            f'• Active Model: <code>{Path(model_path).name}</code>'
            f'</div>',
            unsafe_allow_html=True
        )


# ==========================================================================
# TAB 2: ROI COORDINATE DETAIL
# ==========================================================================
with tab_roi_detail:
    st.markdown("### 📋 Loaded ROI Slot Coordinates")

    if roi_polygons:
        st.markdown(
            f"<div class='alert-box'>ROI polygons imported from: <code>{roi_file_path}</code>. "
            f"Total <b>{len(roi_polygons)}</b> slots configured.</div>",
            unsafe_allow_html=True
        )

        col_json, col_expl = st.columns([1, 1])

        with col_json:
            st.markdown("<div class='section-title'>Polygon Coordinate Structure</div>", unsafe_allow_html=True)
            formatted_coords = {}
            for idx, poly in enumerate(roi_polygons):
                formatted_coords[f"Slot_{idx + 1}"] = [{"x": pt[0], "y": pt[1]} for pt in poly]

            st.json(formatted_coords)

        with col_expl:
            st.markdown("<div class='section-title'>ROI Geometric Logic</div>", unsafe_allow_html=True)
            st.markdown("""
            #### 1. Detection Bounding Box
            The model detects objects and maps each bounding box as:
            $$B = [x_{min}, y_{min}, x_{max}, y_{max}]$$
            The centroid is:
            $$C(x, y) = \\left( \\frac{x_{min} + x_{max}}{2}, \\frac{y_{min} + y_{max}}{2} \\right)$$

            #### 2. Bbox ↔ Polygon Overlap Check
            Slot occupancy is resolved by measuring the intersection ratio between each
            detection bbox and the closed slot polygon $P$, then assigning the detection
            to the single slot with the largest overlap (exclusive assignment).

            #### 3. Method Advantages
            * **Perspective robust:** 4-point polygon logic maps skewed camera angles.
            * **Lightweight:** one global inference pass, no per-slot cropping.
            """)

    else:
        st.warning("⚠️ No ROI coordinates loaded. Check the sidebar settings.")
