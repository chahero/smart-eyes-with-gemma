# 파일: backend/app/services.py
# 역할: YOLO, Ollama, Piper TTS 등 핵심 비즈니스 로직을 수행합니다.
# 작업에 따라 텍스트/비전 모델을 분리하여 사용

import cv2
import numpy as np
import requests
import subprocess
import io
import wave
import base64
from ultralytics import YOLO
from typing import List, Optional, Dict, Any
from collections import deque

from .config import settings
from .models import DetectedObject, BoundingBox, AlertLevel

# --- 이미지 인코딩 헬퍼 함수 ---
def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode('utf-8')

# --- 모델 로드 및 전역 변수 ---
object_history: Dict[int, Dict[str, Any]] = {}
HISTORY_LENGTH = 15
MIN_HISTORY_FOR_ANALYSIS = 5
IMAGE_WIDTH = 640

try:
    # GPU 사용을 시도합니다.
    yolo_model = YOLO(settings.YOLO_MODEL_PATH)
    yolo_model.to('cuda')
    print(f"✅ YOLO model loaded and moved to GPU successfully: {settings.YOLO_MODEL_PATH}")
except Exception as e:
    # GPU 사용이 실패하면 CPU로 폴백합니다.
    print(f"⚠️ Warning: Failed to load YOLO model on GPU ({e}). Falling back to CPU.")
    yolo_model = YOLO(settings.YOLO_MODEL_PATH)
    yolo_model.to('cpu')

# --- 실시간 영상 분석 관련 함수 ---
def analyze_image_with_yolo(image_bytes: bytes) -> List[DetectedObject]:    
    global object_history
    if yolo_model is None: return []
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return []
        results = yolo_model.track(img, conf=settings.YOLO_CONFIDENCE, persist=True, verbose=False)
        current_track_ids = set()
        detected_objects: List[DetectedObject] = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            for box, track_id_tensor in zip(results[0].boxes, results[0].boxes.id):
                track_id = int(track_id_tensor.item())
                current_track_ids.add(track_id)
                bbox = box.xyxy[0].cpu().numpy()
                if track_id not in object_history:
                    object_history[track_id] = {"label": yolo_model.names[int(box.cls[0])], "is_important": yolo_model.names[int(box.cls[0])] in settings.IMPORTANT_OBJECTS, "areas": deque(maxlen=HISTORY_LENGTH)}
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                object_history[track_id]["areas"].append(area)
                status_info = analyze_object_status(track_id)
                direction = analyze_object_direction(bbox)
                detected_objects.append(DetectedObject(track_id=track_id, label=object_history[track_id]["label"], confidence=float(box.conf[0]), box=BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]), is_important=object_history[track_id]["is_important"], status=status_info["status"], size_ratio=status_info["ratio"], direction=direction))
        stale_tracks = [tid for tid in object_history if tid not in current_track_ids]
        for tid in stale_tracks:
            del object_history[tid]
        return detected_objects
    except Exception as e:
        print(f"Error during YOLO analysis: {e}")
        return []

def analyze_object_direction(bbox: np.ndarray) -> str:    
    center_x = (bbox[0] + bbox[2]) / 2
    if center_x < IMAGE_WIDTH / 3: return "left"
    elif center_x > IMAGE_WIDTH * 2 / 3: return "right"
    else: return "front"

def analyze_object_status(track_id: int) -> Dict[str, Any]:    
    history = object_history[track_id]["areas"]
    if len(history) < MIN_HISTORY_FOR_ANALYSIS: return {"status": "Detected", "ratio": 1.0}
    mid_point = len(history) // 2
    if mid_point == 0: return {"status": "Detected", "ratio": 1.0}
    avg_area_past = sum(list(history)[:mid_point]) / mid_point
    if avg_area_past == 0: return {"status": "Detected", "ratio": 1.0}
    avg_area_recent = sum(list(history)[mid_point:]) / (len(history) - mid_point)
    ratio = avg_area_recent / avg_area_past
    if ratio > settings.FAST_APPROACH_THRESHOLD: status = "Fast Approaching"
    elif ratio > settings.SLOW_APPROACH_THRESHOLD: status = "Approaching"
    else: status = "Stationary"
    return {"status": status, "ratio": ratio}

def get_risk_based_guide(objects: List[DetectedObject]) -> Dict[str, Any]:    
    critical_object = next((obj for obj in objects if obj.is_important and obj.status == "Fast Approaching"), None)
    warning_object = next((obj for obj in objects if obj.is_important and obj.status == "Approaching"), None)
    if critical_object:
        message = f"{critical_object.label} is approaching fast from {critical_object.direction}."
        return {"alert_level": AlertLevel.CRITICAL, "guide_message": message}
    if warning_object:
        message = f"{warning_object.label} is approaching from {warning_object.direction}."
        return {"alert_level": AlertLevel.WARNING, "guide_message": message}
    return {"alert_level": AlertLevel.NONE, "guide_message": None}


# --- Ollama 관련 함수 ---
def describe_scene_with_ollama(image_bytes: bytes) -> str:
    """
    Uses Ollama's Vision model to describe the surrounding scene from an image.
    """
    print(f"✅ Scene Description path activated: Ollama Vision Model ({settings.OLLAMA_VISION_MODEL})")
    
    try:
        image_b64 = image_to_base64(image_bytes)
        
        # [수정] 프롬프트 수정: 불필요한 안내 문구 제거 및 직접적인 묘사 유도
        prompt = "You are an AI assistant for a visually impaired person. Describe what you see in this image from a first-person perspective. Start the description immediately, without any introductory phrases. For example: 'A person is sitting at a desk in front of you.'"

        payload = {
            "model": settings.OLLAMA_VISION_MODEL, # 비전 모델 사용
            "messages": [
                {
                    "role": "user",
                    "content": prompt, # 수정된 프롬프트 적용
                    "images": [image_b64]
                }
            ],
            "stream": False
        }
        
        response = requests.post(f"{settings.OLLAMA_HOST}/api/chat", json=payload, timeout=45)
        
        if response.status_code == 200:
            return response.json().get('message', {}).get('content', '').strip()
        else:
            print(f"❌ Ollama API Error (Vision): {response.status_code} - {response.text}")
            return f"Error from AI model (Scene Description): {response.status_code}"
            
    except requests.RequestException as e:
        print(f"❌ Ollama Scene Description API request error: {e}")
        return "Sorry, I'm having trouble describing the scene."
    except Exception as e:
        print(f"❌ Error processing image (Scene Description): {e}")
        return "Sorry, there was an error processing the image for description."

def get_answer_for_question(question: str, image_bytes: bytes) -> str:
    """
    Analyzes the user's question and answers image-related questions with the Vision model,
    and general questions with the Text model.
    """
    # OCR_KEYWORDS에 영어 키워드 추가
    OCR_KEYWORDS = [
        "읽어줘", "써 있어", "글자", "표지판", "내용", "보여", "보이니",  # 한국어
        "read", "text", "sign", "what does it say", "label", "writing", "show me", "can you see", "what do you see" # 영어
    ]
    trigger_ocr = any(keyword in question.lower() for keyword in OCR_KEYWORDS) # 질문을 소문자로 변환하여 비교

    # --- 경로 1: 이미지 분석 요청 시 (Ollama Vision 모델 사용) ---
    if trigger_ocr:
        print(f"✅ Image question path activated: Ollama Vision Model ({settings.OLLAMA_VISION_MODEL})")
        try:
            image_b64 = image_to_base64(image_bytes)
            
            # 프롬프트 수정: 직접적인 답변 유도
            prompt = f"Answer the user's question based on the image. Be direct and concise. Question: {question}"

            payload = {
                "model": settings.OLLAMA_VISION_MODEL, # 비전 모델 사용
                "messages": [{"role": "user", "content": prompt, "images": [image_b64]}], # 수정된 프롬프트 적용
                "stream": False
            }
            response = requests.post(f"{settings.OLLAMA_HOST}/api/chat", json=payload, timeout=45)
            
            if response.status_code == 200:
                return response.json().get('message', {}).get('content', '').strip()
            else:
                print(f"❌ Ollama API Error (Vision): {response.status_code} - {response.text}")
                return f"Error from AI model (Vision): {response.status_code}"
                
        except requests.RequestException as e:
            print(f"❌ Ollama Vision API request error: {e}")
            return "Sorry, I'm having trouble analyzing the image."
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return "Sorry, there was an error processing the image."

    # --- 경로 2: 일반 질문 시 (Ollama Text 모델 사용) ---
    else:
        print(f"➡️ General question path activated: Ollama Text Model ({settings.OLLAMA_TEXT_MODEL})")
        try:
            detected_objects = analyze_image_with_yolo(image_bytes)
            objects_text = ", ".join([f"a {obj.label} on the {obj.direction}" for obj in detected_objects]) if detected_objects else "nothing detected"
        except Exception as e:
            print(f"Error during YOLO analysis (for question answering): {e}")
            objects_text = "an error in vision analysis"

        prompt = f"""
        You are a data-to-text reporting tool for a visually impaired user. Your function is to state visual facts based on the user's question.
        Follow these rules strictly:
        1. Answer only the user's direct question.
        2. Do not add any extra advice, suggestions, or conversational filler.
        3. Your response must be a factual description based on the visual context.
        4. Keep the answer as short as possible.
        ---
        **Visual Context:** "{objects_text}"
        **User's Question:** "{question}"
        ---
        Provide your answer now.
        """
        payload = {
            "model": settings.OLLAMA_TEXT_MODEL, # 텍스트 모델 사용
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        try:
            response = requests.post(f"{settings.OLLAMA_HOST}/api/chat", json=payload, timeout=20)
            if response.status_code == 200:
                return response.json().get('message', {}).get('content', '').strip().strip('"\'')
            else:
                print(f"Ollama API Error (Text): {response.status_code} - {response.text}")
                return f"Error from AI model: {response.status_code}"
        except requests.RequestException as e:
            print(f"Ollama question answering request error: {e}")
            return "Sorry, I'm having trouble connecting to the AI."

# --- Piper TTS 관련 함수 ---
def generate_tts_audio(text: str) -> Optional[bytes]:    
    if not text or not text.strip(): return None
    command = [settings.PIPER_EXE_PATH, '--model', settings.PIPER_MODEL_PATH, '--output-raw']
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = process.communicate(input=text.encode('utf-8'), timeout=30)
        return convert_raw_to_wav(stdout) if stdout else None
    except Exception as e:
        print(f"Error during TTS processing: {e}")
        return None

def convert_raw_to_wav(raw_data: bytes) -> bytes:    
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(raw_data)
    return wav_buffer.getvalue()
