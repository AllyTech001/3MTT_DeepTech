import streamlit as st
from PIL import Image, ExifTags
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from fpdf import FPDF
from recognizer import load_reference_encodings, recognize_faces

# --- Streamlit page config ---
st.set_page_config(page_title="🧠 Smart Face Recognition", layout="wide")

# --- Create necessary dirs ---
LOG_FILE = "logs/recognition_log.csv"
os.makedirs("logs", exist_ok=True)
os.makedirs("output", exist_ok=True)

TARGET_SIZE = 720  # Display size for images

# --- Dark sci-fi theme CSS ---
def set_dark_theme():
    st.markdown(
        """
        <style>
        body, .main {
            background-color: #0e1117;
            color: #c8d6e5;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stButton>button {
            background: linear-gradient(135deg, #00f0ff 0%, #0066ff 100%);
            color: white;
            border-radius: 10px;
            font-weight: bold;
            padding: 10px 20px;
            transition: background 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #00c3ff 0%, #004ecc 100%);
        }
        .stFileUploader > div {
            background-color: #121823;
            border-radius: 12px;
            padding: 10px;
        }
        .css-1d391kg {
            background-color: #121823 !important;
            border-radius: 12px;
        }
        .css-1d391kg, .css-1v0mbdj {
            color: #c8d6e5 !important;
        }
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background: #005eff;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
set_dark_theme()

def letterbox_image(image, size=TARGET_SIZE):
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas

def draw_futuristic_overlay(image, results):
    overlay = image.copy()
    glow = image.copy()
    for res in results:
        if "error" in res:
            continue
        top, right, bottom, left = [int(v * 2) for v in res["box"]]
        color = (0, 255, 0) if res["match"] else (0, 0, 255)
        thickness = 6 if res["match"] else 5
        cv2.rectangle(overlay, (left, top), (right, bottom), color, thickness)

        # Horizontal dashed lines
        num_lines = 8
        line_spacing = (bottom - top) // (num_lines + 1)
        dash_length = 10
        for i in range(1, num_lines + 1):
            y = top + i * line_spacing
            for x_start in range(left, right, dash_length * 2):
                x_end = min(x_start + dash_length, right)
                cv2.line(overlay, (x_start, y), (x_end, y), color, 6)

        # Vertical dashed lines
        num_vlines = 5
        vline_spacing = (right - left) // (num_vlines + 1)
        for i in range(1, num_vlines + 1):
            x = left + i * vline_spacing
            for y_start in range(top, bottom, dash_length * 2):
                y_end = min(y_start + dash_length, bottom)
                cv2.line(overlay, (x, y_start), (x, y_end), color, 1)

        # Label background
        label_text = f"{res['name']} | {res['confidence']}%"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        rect_tl = (left, top - text_h - baseline - 12)
        rect_br = (left + text_w + 12, top)
        cv2.rectangle(overlay, rect_tl, rect_br, color, cv2.FILLED)

        alpha = 0.7
        cv2.addWeighted(
            overlay[rect_tl[1]:rect_br[1], rect_tl[0]:rect_br[0]],
            alpha,
            image[rect_tl[1]:rect_br[1], rect_tl[0]:rect_br[0]],
            1 - alpha,
            0,
            image[rect_tl[1]:rect_br[1], rect_tl[0]:rect_br[0]],
        )

        cv2.putText(
            image,
            label_text,
            (left + 6, top - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

        cv2.rectangle(glow, (left, top), (right, bottom), color, thickness * 2)

    glow = cv2.GaussianBlur(glow, (21, 21), 0)
    combined = cv2.addWeighted(image, 0.7, glow, 0.3, 0)
    final = cv2.addWeighted(combined, 0.9, overlay, 0.5, 0)
    return final

def get_exif_location(img_pil):
    """Extract GPS coordinates from image EXIF, return (lat, lon) or None."""
    try:
        exif_data = img_pil._getexif()
        if not exif_data:
            return None
        gps_info = {}
        for tag, value in exif_data.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                for t in value:
                    sub_decoded = ExifTags.GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = value[t]
        if not gps_info:
            return None

        def convert_to_degrees(value):
            d = value[0][0] / value[0][1]
            m = value[1][0] / value[1][1]
            s = value[2][0] / value[2][1]
            return d + (m / 60.0) + (s / 3600.0)

        lat = convert_to_degrees(gps_info["GPSLatitude"])
        lon = convert_to_degrees(gps_info["GPSLongitude"])
        if gps_info.get("GPSLatitudeRef") != "N":
            lat = -lat
        if gps_info.get("GPSLongitudeRef") != "E":
            lon = -lon
        return lat, lon
    except Exception:
        return None

def gps_to_google_maps_url(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"

def log_results(results, location_url=None):
    logs = []
    for res in results:
        if "error" in res:
            continue
        logs.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Name": res["name"],
            "Confidence": res["confidence"],
            "Match": res["match"],
            "Location": location_url or ""
        })
    if logs:
        df = pd.DataFrame(logs)
        df.to_csv(LOG_FILE, mode='a', index=False, header=not os.path.exists(LOG_FILE))
    return logs

def load_all_logs():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=["Timestamp", "Name", "Confidence", "Match", "Location"])

def generate_pdf_report(logs_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Face Recognition Scan Report", 0, 1, "C")
    pdf.set_font("Arial", size=12)
    for _, row in logs_df.iterrows():
        line = f"{row['Timestamp']} - {row['Name']} - Confidence: {row['Confidence']}%"
        if row["Location"]:
            line += f" - Location: {row['Location']}"
        pdf.cell(0, 10, line, 0, 1)
    pdf_path = "output/scan_result.pdf"
    pdf.output(pdf_path)
    return pdf_path

def make_clickable(val):
    if val:
        return f'[Map]({val})'
    return ""

# --- Load known faces on app start ---
with st.spinner("🔄 Loading reference faces..."):
    known_encodings, known_names = load_reference_encodings()

# --- Sidebar navigation ---
st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio("Go to", ["Upload & Recognize", "Recognition Logs"])

if page == "Upload & Recognize":
    st.title("🧠 Smart Face Recognition System")
    st.markdown("---")

    input_mode = st.radio("Select Input Mode", ["Upload Image", "Use Webcam"])

    img_cv = None
    location_url = None
    results = None

    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("🖼️ Upload Image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img_pil = Image.open(uploaded_file)
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            # Extract location from EXIF
            gps = get_exif_location(img_pil)
            if gps:
                location_url = gps_to_google_maps_url(*gps)
    else:
        capture = st.camera_input("📷 Capture Image from Webcam")
        if capture:
            img_pil = Image.open(capture)
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            # No GPS info available from webcam captures
            location_url = None

    if img_cv is not None:
        col1, col2 = st.columns(2)
        img_input_display = letterbox_image(img_cv)
        col1.image(cv2.cvtColor(img_input_display, cv2.COLOR_BGR2RGB), caption="🔍 Input Image", use_container_width=True)

        with st.spinner("⏳ Scanning for matches..."):
            results = recognize_faces(img_cv, known_encodings, known_names)

        if not results or "error" in results[0]:
            col2.error("🚫 No face detected or recognition error.")
        else:
            img_annotated = draw_futuristic_overlay(img_cv.copy(), results)
            img_output_display = letterbox_image(img_annotated)
            col2.image(cv2.cvtColor(img_output_display, cv2.COLOR_BGR2RGB), caption="📡 Recognition Output", use_container_width=True)

            log_results(results, location_url)

elif page == "Recognition Logs":
    st.title("🧾 Recognition Logs")
    st.markdown("---")

    logs_df = load_all_logs()

    all_names = sorted(logs_df["Name"].unique())
    filter_name = st.multiselect("Filter by Name(s)", options=all_names, default=all_names)

    min_date = pd.to_datetime(logs_df["Timestamp"]).min() if not logs_df.empty else None
    max_date = pd.to_datetime(logs_df["Timestamp"]).max() if not logs_df.empty else None

    date_range = st.date_input(
        "Filter by Date Range",
        value=(min_date, max_date) if min_date and max_date else None,
        min_value=min_date,
        max_value=max_date,
    )

    filtered_df = logs_df.copy()
    if filter_name:
        filtered_df = filtered_df[filtered_df["Name"].isin(filter_name)]

    if date_range and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = filtered_df[
            (pd.to_datetime(filtered_df["Timestamp"]) >= start_date)
            & (pd.to_datetime(filtered_df["Timestamp"]) <= end_date + pd.Timedelta(days=1))
        ]

    filtered_df["Map Link"] = filtered_df["Location"].apply(make_clickable)

    st.dataframe(
        filtered_df.drop(columns=["Location"]),
        use_container_width=True,
    )

    st.markdown("### Location Links:")
    for idx, row in filtered_df.iterrows():
        if row["Location"]:
            st.markdown(f"- **{row['Name']}** at [{row['Location']}]({row['Location']})")

    if st.button("📄 Generate PDF Report"):
        if filtered_df.empty:
            st.warning("No logs to generate report.")
        else:
            pdf_path = generate_pdf_report(filtered_df)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=f,
                    file_name="recognition_report.pdf",
                    mime="application/pdf",
                )
