# AI-Powered Waste Type Detection

An interactive computer vision web application built with Streamlit and YOLO (Ultralytics) that detects waste objects in real time and classifies them into **Organic (_Organik_)** and **Inorganic (_Anorganik_)** categories.

---

## 🚀 Live Demo

Access the live application deployed on Streamlit Community Cloud:

👉 **[https://waste-type-detection-oybbjtnv7yzsejug3degew.streamlit.app/](https://waste-type-detection-oybbjtnv7yzsejug3degew.streamlit.app/)**

---

## 📄 Project Purpose

Proper waste sorting is an essential step in modern recycling systems and environmental management. Manual sorting can be prone to human error and inconsistency. This project demonstrates how computer vision and deep learning can automate waste identification and categorize discarded items into organic and inorganic streams at the point of disposal.

---

## 📌 Features

- **Real-Time Webcam Detection**: Low-latency video streaming directly through the web browser powered by `streamlit-webrtc` and Google STUN servers.
- **Image Upload Detection**: Support for uploading static image files (`.jpg`, `.jpeg`, `.png`) for instant analysis.
- **YOLO Object Detection**: Uses custom trained weights (`best.pt`) to locate and detect waste items.
- **Bounding Boxes & Confidence Scores**: Displays localized bounding boxes with detection probabilities for each object.
- **Organic / Inorganic Categorization**: Automatically maps detected object labels to their respective waste categories (_Organik_ vs _Anorganik_) with distinct color-coded badges (Green for Organic, Red for Inorganic).
- **Streamlit Cloud Compatible**: Built using browser-side camera access, eliminating dependencies on server-side video hardware.

---

## 🧠 How It Works

```text
User Input (Webcam Stream / Image Upload)
                   │
                   ▼
    Frame Preprocessing (OpenCV)
  - Color space conversion (BGR to RGB)
  - Resolution management (640x480)
                   │
                   ▼
       YOLO Inference (best.pt)
  - Object localization & bounding boxes
  - Class prediction & confidence score
                   │
                   ▼
       Category Classification
  - Maps detected label via kategori_sampah
  - Assigns "Organik" (🟩) or "Anorganik" (🟥)
                   │
                   ▼
             Display Output
  - Real-time video overlay or annotated image
  - Formatted classification summary cards
```

1. **Input Handling**: In **Webcam** mode, video frames are captured through the user's browser via WebRTC. In **Upload Gambar** mode, the user uploads an image file decoded via OpenCV.
2. **Model Inference**: Each frame or image is passed to the YOLO model (`best.pt`). The model identifies objects matching trained waste classes with a confidence threshold (`conf=0.5`).
3. **Category Mapping**: Detected labels (such as `botol`, `kertas`, `plastik`) are checked against the `kategori_sampah` mapping dictionary to determine whether the object is organic or inorganic.
4. **Rendering**:
   - For the webcam stream, bounding boxes and two-line category badges (`LABEL (CONF)` and `Kategori`) are drawn directly onto the video frames.
   - For uploaded images, the annotated image is presented alongside classification cards and category details.

---

## 🛠️ Tech Stack

| Technology                            | Purpose                                                       |
| ------------------------------------- | ------------------------------------------------------------- |
| **Python**                            | Core programming language                                     |
| **Streamlit**                         | Web application framework and user interface                  |
| **Ultralytics YOLO**                  | Object detection architecture and model inference (`best.pt`) |
| **OpenCV (`opencv-python-headless`)** | Image decoding, color conversion, and canvas annotation       |
| **Streamlit-WebRTC & PyAV**           | Real-time browser-to-server video stream handling             |
| **NumPy**                             | Array transformations and image buffer manipulation           |
| **Pandas**                            | Tabular data support                                          |

---

## 📊 Waste Categories

Detected waste labels are classified into two primary categories:

| Detected Label     | Waste Category            | Indicator | Notes                             |
| ------------------ | ------------------------- | :-------: | --------------------------------- |
| `kertas` / `paper` | **Organik** (Organic)     |    🟩     | Paper products, degradable fibers |
| `daun`             | **Organik** (Organic)     |    🟩     | Leaves, plant-based matter        |
| `plastik`          | **Anorganik** (Inorganic) |    🟥     | Plastic packaging, containers     |
| `botol`            | **Anorganik** (Inorganic) |    🟥     | Bottles (plastic/glass)           |
| `kaca` / `glass`   | **Anorganik** (Inorganic) |    🟥     | Glass containers, shards          |
| `logam` / `metal`  | **Anorganik** (Inorganic) |    🟥     | Cans, metal hardware              |

---

## 📂 Project Structure

```text
waste-type-detection/
├── app3.py              # Main Streamlit web application
├── best.pt              # Trained YOLO weights for waste detection
├── requirements.txt     # Python dependencies for deployment
├── README.md            # Project documentation
├── main.py              # CLI entry point for training and offline detection
├── scripts/             # Supporting scripts (train.py, webcam.py, webcam_detect.py)
├── dataset/             # Dataset configuration (data.yaml) and training splits
├── yolo11n.pt           # YOLO11 nano base model
└── yolov8n.pt           # YOLOv8 nano base model
```

---

## 🎮 How to Use

### 1. Real-Time Webcam Mode

1. Open the application.
2. Select **Webcam** from the mode selector.
3. Grant camera permission in your web browser when prompted.
4. Click the **START** button in the video player.
5. Hold waste items in front of your camera.
6. Observe live bounding boxes, object names, confidence levels, and Organic/Inorganic tags.
7. Click **STOP** when you are done.

### 2. Upload Image Mode

1. Select **Upload Gambar** from the mode selector.
2. Click **Browse files** and select a `.jpg`, `.jpeg`, or `.png` file.
3. View the original image alongside the detected bounding boxes.
4. Check the **Hasil Klasifikasi Sampah** section below to review identified items and category cards.

---

## ⚙️ Installation & Setup

To run this application locally on your machine, follow these steps:

### Prerequisites

- Python 3.8 to 3.12 installed
- A working webcam (for real-time detection)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Vinn673/waste-type-detection.git
   cd waste-type-detection
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit App**

   ```bash
   streamlit run app3.py
   ```

5. Open your web browser and navigate to `http://localhost:8501`.

---

## 🌐 Deployment

The application is hosted on **Streamlit Community Cloud**:

- **Live URL**: [https://waste-type-detection-oybbjtnv7yzsejug3degew.streamlit.app/](https://waste-type-detection-oybbjtnv7yzsejug3degew.streamlit.app/)
- The deployment utilizes `opencv-python-headless` to ensure headless Linux server compatibility without missing shared graphics libraries (`libGL.so.1`).
- `streamlit-webrtc` operates with public STUN servers (`stun.l.google.com:19302`) for reliable NAT traversal in cloud container environments.

---

## 👥 Team Members

**Group 03**

| Name                           | Student ID |
| ------------------------------ | ---------- |
| **Bryan Carlos Matruti**       | 2802491204 |
| **Marvin Adriano Rusdianto**   | 2802402275 |
| **I Gede Aryaputra Maheswara** | 2802488922 |
| **Ernest Angelo Winoto**       | 2802494755 |
