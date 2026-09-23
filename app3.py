import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }
)

st.set_page_config(page_title="Waste Detection", layout="wide")

model = YOLO("best.pt") 

st.title("Waste Detection")

option = st.radio("Pilih Mode:", ["Webcam", "Upload Gambar"])


kategori_sampah = {
    "plastik": "Anorganik",
    "botol": "Anorganik",
    "kaca": "Anorganik",
    "glass": "Anorganik",       
    "logam": "Anorganik",
    "metal": "Anorganik", 
    "kertas": "Organik",
    "paper": "Organik",         
    "daun": "Organik"
}


st.markdown("""
<style>
.card {
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 12px;
    color: white;
    font-size: 18px;
}
.organik {
    background-color: #2ecc71;
}
.anorganik {
    background-color: #e74c3c;
}
.icon {
    font-size: 28px;
    margin-right: 10px;
}
</style>
""", unsafe_allow_html=True)


def tampilkan_kategori(results):
    detected_labels = set()

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id].lower()
        detected_labels.add(label)

    st.subheader("📌 Hasil Klasifikasi Sampah")

    if not detected_labels:
        st.write("Tidak ada objek terdeteksi.")
        return

  
    st.write("""
             Kartu Kategori Sampah
             🟩 Organik 
             🟥 Anorganik """)
    for label in detected_labels:
        kategori = kategori_sampah.get(label, "Tidak diketahui")

        css_class = "organik" if kategori == "Organik" else "anorganik"
        icon = "🟩" if kategori == "Organik" else "🟥"

        st.markdown(
            f"""
            <div class="card {css_class}">
                <span class="icon">{icon}</span>
                <b>{label.upper()}</b> → {kategori}
            </div>
            """,
            unsafe_allow_html=True
        )


    for label in detected_labels:
        kategori = kategori_sampah.get(label, "Tidak diketahui")
        icon = "🟩🗑 Organik" if kategori == "Organik" else "🟥🗑 Anorganik"
        st.write(f"- **{label}** → {icon}")




def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model.predict(img_rgb, conf=0.5, verbose=False)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id].lower()
        kategori = kategori_sampah.get(label, "Tidak diketahui")
        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if kategori == "Organik":
            box_color = (113, 204, 46)  # Green in BGR
        elif kategori == "Anorganik":
            box_color = (60, 76, 231)   # Red in BGR
        else:
            box_color = (0, 165, 255)   # Amber in BGR

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

        # Label lines: object name with confidence, and waste category
        line1 = f"{label.upper()} ({conf:.2f})"
        line2 = f"{kategori}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
        (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, thickness)

        badge_w = max(w1, w2) + 12
        badge_h = h1 + h2 + 14

        badge_y1 = max(0, y1 - badge_h)
        badge_y2 = badge_y1 + badge_h
        badge_x1 = max(0, x1)
        badge_x2 = min(img.shape[1], badge_x1 + badge_w)

        cv2.rectangle(img, (badge_x1, badge_y1), (badge_x2, badge_y2), box_color, -1)

        cv2.putText(
            img,
            line1,
            (badge_x1 + 6, badge_y1 + h1 + 3),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            line2,
            (badge_x1 + 6, badge_y1 + h1 + h2 + 9),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return av.VideoFrame.from_ndarray(img, format="bgr24")


if option == "Webcam":
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.markdown(
            """
            <div style="display: flex; justify-content: center; gap: 12px; margin-bottom: 12px;">
                <span style="background-color: #2ecc71; color: white; padding: 5px 12px; border-radius: 6px; font-weight: 600; font-size: 13px;">
                    🟩 Organik
                </span>
                <span style="background-color: #e74c3c; color: white; padding: 5px 12px; border-radius: 6px; font-weight: 600; font-size: 13px;">
                    🟥 Anorganik
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        webrtc_streamer(
            key="waste-detection-webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640, "max": 640},
                    "height": {"ideal": 480, "max": 480},
                    "frameRate": {"ideal": 20, "max": 30},
                },
                "audio": False,
            },
            async_processing=True,
        )

        st.caption("ℹ️ Klik tombol **START** di atas untuk mengaktifkan webcam browser dan memulai deteksi real-time.")


else:
    uploaded = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.image(img_rgb, caption="Gambar Asli")

        results = model.predict(img_rgb, conf=0.5)
        annotated = results[0].plot()

        st.image(annotated, caption="Hasil Deteksi")

        tampilkan_kategori(results)