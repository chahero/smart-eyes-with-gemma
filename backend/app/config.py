# 파일: backend/app/config.py
# 역할: .env 파일에서 환경 변수를 로드하고 관리합니다.

import os
from dotenv import load_dotenv

# .env 파일의 위치를 명시적으로 지정합니다.
# 이 파일은 backend 폴더 안에 있어야 합니다.
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

class Settings:
    # YOLO 설정
    YOLO_MODEL_PATH: str = os.getenv('YOLO_MODEL_PATH', 'models/yolov11n.pt')
    YOLO_CONFIDENCE: float = float(os.getenv('YOLO_CONFIDENCE', 0.5))

    # Ollama (Gemma) 설정
    OLLAMA_HOST: str = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    OLLAMA_TEXT_MODEL: str = os.getenv('OLLAMA_TEXT_MODEL', 'gemma3n:e2b')
    OLLAMA_VISION_MODEL: str = os.getenv('OLLAMA_VISION_MODEL', 'gemma3:4b')

    # Piper TTS 설정
    PIPER_EXE_PATH: str = os.getenv('PIPER_EXE_PATH', 'piper/piper.exe')
    PIPER_MODEL_PATH: str = os.getenv('PIPER_MODEL_PATH', 'piper/en_US-lessac-medium.onnx')

    # 중요 객체 목록
    IMPORTANT_OBJECTS: list = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'chair', 'stairs']

    # 위험 등급 분석을 위한 설정
    FAST_APPROACH_THRESHOLD: float = 2.0
    SLOW_APPROACH_THRESHOLD: float = 1.4

settings = Settings()
