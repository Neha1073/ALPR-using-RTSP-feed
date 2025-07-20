# ALPR-using-RTSP-feed
A multithreaded license plate detection system with RTSP camera/video input. Logs unique plates with confidence and position data to a CSV file.

---

## 📌 Features

- 📹 RTSP or video file input
- 🧠 Vehicle and license plate detection using YOLOv8
- 🔤 OCR with Tesseract for license plate text
- 🧵 Multithreaded processing (capture, detection, logging)
- 🗂 Duplicate filtering (time, location, similarity)
- 📄 CSV logging with timestamp, frame number, plate, confidence, and coordinates

---

## 🛠️ Requirements

- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and added to PATH
