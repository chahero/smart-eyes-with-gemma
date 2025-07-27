# Smart Gemma Stick

An intelligent assistive system for the visually impaired, combining real-time object detection (YOLO), AI-powered situational analysis (Gemma), and high-quality voice guidance (Piper TTS).

This smart stick application provides real-time environmental awareness through webcam-based object detection, AI-driven safety analysis, and natural voice guidance to help visually impaired users navigate safely.

This project is submitted as a demo source for **The Gemma 3n Impact Challenge** on Kaggle: https://www.kaggle.com/competitions/google-gemma-3n-hackathon/overview

## ✨ Key Features

- **👁️ Real-time Object Detection**: Uses YOLOv11 to detect people, vehicles, obstacles, and important objects with visual overlays and confidence scores.
- **🧠 AI-Powered Navigation Assistance**: Gemma AI analyzes detected objects and provides concise, safety-focused navigation guidance (max 15 words).
- **🗣️ High-Quality Voice Guidance**: Piper TTS delivers natural, human-like voice instructions with configurable intervals.
- **⚡ Smart Processing**: Intelligent frame skipping and object filtering to optimize performance and reduce unnecessary alerts.
- **🎛️ Real-time Controls**: Live adjustment of confidence thresholds, AI models, and TTS settings.
- **🔊 Audio Management**: Toggle TTS on/off with queue-based audio processing.
- **🌐 Web-based Interface**: Accessible via a web browser, supporting various devices.

## ⚙️ Technology Stack

| **Component** | **Technology** | **Description** |
| --- | --- | --- |
| Object Detection | YOLOv11 | Fast, accurate real-time object detection |
| AI Analysis | Google Gemma (via Ollama) | Local situational analysis and navigation guidance |
| Text-to-Speech | Piper TTS | High-quality, local speech synthesis |
| Backend | FastAPI | High-performance web framework for API endpoints |
| Frontend | HTML, CSS (Tailwind CSS), JavaScript | Interactive web interface with responsive design |
| Video Processing | OpenCV | Real-time camera stream handling and image manipulation |
| Audio Processing | `io`, `wave` (Python standard libraries) | Audio data handling and WAV conversion |

## 🔧 Installation Guide

### Prerequisites

- Python 3.10+
- Git
- Webcam/Camera
- Ollama installed and running locally
- Piper TTS executable and model files

### 1. Clone Repository

```
git clone https://github.com/your-username/smart-gemma-stick.git # Repository name updated
cd smart-gemma-stick
```

### 2. Install Python Dependencies

```
pip install -r requirements.txt
```

*(This will install all necessary packages including `fastapi`, `uvicorn`, `ultralytics`, `python-dotenv`, `requests`, `numpy`, `opencv-python`, etc.)*

### 3. Set up Ollama and Gemma

1. Install [Ollama](https://ollama.ai/) by following their official instructions.
2. Pull the Gemma model (e.g., `gemma3n:e2b` and `gemma3:4b` for vision):
    
    ```
    ollama pull gemma3n:e2b
    ollama pull gemma3:4b
    ```
    

### 4. Set up Piper TTS

1. Download the appropriate Piper executable for your OS (e.g., `piper_windows_x64.zip` for Windows) from [Piper Releases](https://github.com/rhasspy/piper/releases).
2. Create a `piper/` folder in your project root directory.
3. Extract the Piper executable (e.g., `piper.exe`) into the `piper/` folder.
4. Download your desired voice model (e.g., `en_US-lessac-medium.onnx` and its `.onnx.json` config file) from [Piper Voice Models](https://github.com/rhasspy/piper/blob/master/VOICES.md).
5. Place the voice model files (both `.onnx` and `.onnx.json`) into the `piper/` folder.

### 5. Configure Environment Variables

Create a `.env` file in the `backend/app` directory (or adjust `config.py` to load from the project root if preferred) with the following content:

```
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_TEXT_MODEL=gemma3n:e2b
OLLAMA_VISION_MODEL=gemma3:4b

# Camera Settings
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720

# YOLO Model Configuration
YOLO_MODEL_PATH=models/yolov11n.pt
YOLO_CONFIDENCE=0.5
YOLO_SKIP_FRAMES=3

# Piper TTS Configuration
PIPER_EXE_PATH=piper/piper.exe
PIPER_MODEL_PATH=piper/en_US-lessac-medium.onnx
```

*(Note: Ensure `YOLO_MODEL_PATH` points to your downloaded YOLOv11 model file. You might need to download `yolov11n.pt` or a similar model.)*

## 🚀 Usage

### Start the Backend Server

Navigate to the `backend` directory and run the FastAPI application using Uvicorn:
```
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

This will start the backend server, typically accessible at `http://localhost:8000`.

### Access the Frontend

Open your web browser and navigate to the backend server's address (e.g., `http://localhost:8000`). The `index.html` file will be served automatically.

### Using the Interface

1. Click the **"Start"** button to activate your webcam and begin real-time analysis.
2. The video feed will show bounding boxes around detected objects.
3. **Voice Guidance**: The system will provide real-time voice alerts based on approaching important objects.
4. **"Describe Scene" button (👁️)**: Click this to get a detailed AI-generated description of your current surroundings.
5. **"Microphone" button (🎤)**: Click and speak your question. The AI will answer based on the visual context or general knowledge.
6. **"Language Switch" button (EN/KO)**: Click this button to toggle between English and Korean for speech recognition.
7. **"Stop" button**: Deactivates the camera and stops the analysis.
8. **"Switch Camera" button (🔄)**: Toggles between front and rear cameras (if available).
9. **"Help" button (❓)**: Displays information about important objects and alert levels.

## 🎯 How It Works

1. **Video Capture**: The frontend captures real-time video frames from the webcam.
2. **WebSocket Streaming**: Frames are streamed to the FastAPI backend via WebSocket.
3. **Object Detection (YOLOv11)**: The backend processes frames using YOLOv11 to detect objects, track them, and analyze their movement (e.g., approaching speed, direction).
4. **AI Analysis (Google Gemma via Ollama)**:
    - **Situational Awareness**: Detected objects and their statuses are fed to Gemma AI (Text model) to generate concise, safety-focused navigation guidance.
    - **Scene Description**: Upon request, Gemma AI (Vision model) describes the visual content of the current frame.
    - **Question Answering**: Gemma AI (Vision or Text model, depending on the question type) answers user questions based on the current visual context or general knowledge.
5. **Voice Output (Piper TTS)**: AI-generated guidance and answers are converted into natural-sounding speech using Piper TTS.
6. **Real-time Display**: The frontend overlays bounding boxes and status messages on the video feed, providing visual feedback.

## 📊 System Requirements

- **CPU**: Modern multi-core processor (for YOLO and Ollama inference)
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ for models and dependencies
- **Camera**: USB webcam or integrated camera
- **OS**: Windows 10/11, macOS, or Linux (ensure Piper TTS executable is compatible)

## 🤝 Contributing

We welcome contributions to the Smart Gemma Stick project! Please follow these steps:

1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLO
- [Google](https://ai.google.dev/gemma) for Gemma
- [Ollama](https://ollama.ai/) for local LLM serving
- [Piper TTS](https://github.com/rhasspy/piper) for high-quality speech synthesis
- [FastAPI](https://fastapi.tiangolo.com/) for the backend web framework
- [Tailwind CSS](https://tailwindcss.com/) for rapid UI development