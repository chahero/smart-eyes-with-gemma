# 파일: backend/app/api.py
# 역할: FastAPI의 API 엔드포인트(경로)를 정의합니다.
# [수정] Ollama용으로 함수명 변경

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import io

from .models import AnalysisResponse, QuestionResponse
from . import services

router = APIRouter()

# --- WebSocket 엔드포인트 ---
@router.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """
    Receives frames in real-time via WebSocket and sends analysis results to the client.
    """
    await websocket.accept()
    print("WebSocket client connected.")
    try:
        while True:
            image_bytes = await websocket.receive_bytes()
            
            detected_objects = services.analyze_image_with_yolo(image_bytes)
            guide_info = services.get_risk_based_guide(detected_objects)

            response = AnalysisResponse(
                objects=detected_objects, 
                guide_message=guide_info["guide_message"],
                alert_level=guide_info["alert_level"]
            )
            await websocket.send_json(response.dict())

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        services.object_history.clear()
        print("Object history cleared.")

# --- REST API 엔드포인트 ---

@router.post("/describe_scene", response_model=QuestionResponse, summary="Describe the surrounding scene")
async def describe_scene(
    file: UploadFile = File(..., description="Image frame at the time of scene description request")
):
    """
    Generates a description of the surrounding scene based on current visual information.
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file is empty for scene description.")

    # [수정] Ollama용 서비스 함수 호출
    description = services.describe_scene_with_ollama(image_bytes)
    
    if not description:
        description = "I'm sorry, I could not describe the scene."

    return QuestionResponse(answer=description)


@router.post("/ask_question", response_model=QuestionResponse, summary="Process user voice question")
async def ask_question(
    question: str = Form(..., description="User's voice question text"),
    file: UploadFile = File(..., description="Image frame at the time of question")
):
    """
    Generates an answer based on the user's question and current visual information.
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file is empty for the question.")

    # services.get_answer_for_question 함수는 내부적으로 Ollama를 사용하도록 수정되었음
    answer = services.get_answer_for_question(question, image_bytes)
    
    if not answer:
        answer = "I'm sorry, I could not find an answer."

    return QuestionResponse(answer=answer)


@router.post("/generate_tts", summary="Convert text to speech")
async def generate_tts(text: str = Form(...)):
    """
    Converts input text to WAV audio data and returns it as a stream.
    """
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty.")
    
    audio_bytes = services.generate_tts_audio(text)
    if audio_bytes is None:
        raise HTTPException(status_code=500, detail="Failed to generate TTS audio.")
    
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")
