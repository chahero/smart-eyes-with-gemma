# 파일: backend/app/main.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from .api import router
import os

app = FastAPI(
    title="Smart Gemma Stick API",
    description="Backend server for real-time video analysis and voice guidance",
    version="1.0.0"
)

# CORS 설정 추가 (프론트엔드와 백엔드 간 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 임시 이미지 파일 디렉토리
temp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "temp")
os.makedirs(temp_dir, exist_ok=True)

# 기존의 app.mount 제거하고 커스텀 엔드포인트 추가
@app.get("/temp/{filename}")
async def serve_temp_image(filename: str):
    """Serve temporary image files with CORP headers"""
    file_path = os.path.join(temp_dir, filename)
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        return {"error": "File not found"}, 404
    
    return FileResponse(
        file_path,
        media_type="image/jpeg",
        headers={
            "Cross-Origin-Resource-Policy": "cross-origin",
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "public, max-age=31536000",
            "X-Content-Type-Options": "nosniff"
        }
    )

# API 라우터를 먼저 등록합니다.
@app.get("/status", summary="Check server status")
def read_root():
    """Endpoint to check if the server is running correctly."""
    return {"status": "Smart Gemma Stick API is running!"}

app.include_router(router, prefix="/api")

# 시작 시 모델 로드 상태 확인
@app.on_event("startup")
async def startup_event():
    from .services import yolo_model
    if yolo_model is None:
        print("⚠️ Warning: YOLO model not loaded. Please check the model path.")
    else:
        print("✅ YOLO model loaded successfully.")

# --- HTML 프론트엔드를 제공하는 코드는 가장 마지막에 둡니다. ---
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")

# /assets 경로로 요청이 오면 frontend/assets 폴더의 파일을 제공
if os.path.exists(os.path.join(frontend_dir, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dir, "assets")), name="assets")

# 루트 경로('/')로 접속 시, frontend/index.html 파일을 반환
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(
            content="""
            <html>
                <body>
                    <h1>Smart Gemma Stick</h1>
                    <p>Frontend index.html not found.</p>
                    <p>API documentation available at <a href="/docs">/docs</a>.</p>
                </body>
            </html>
            """, 
            status_code=404
        )
    
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        return HTMLResponse(
            content=f"<html><body><h1>File Read Error</h1><p>{str(e)}</p></body></html>",
            status_code=500
        )
