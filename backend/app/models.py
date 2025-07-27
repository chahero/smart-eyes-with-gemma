# 파일: backend/app/models.py
# 역할: API의 입출력 데이터 형식을 Pydantic 모델로 정의합니다.

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class AlertLevel(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"
    NONE = "NONE"

class BoundingBox(BaseModel):
    x1: float = Field(..., description="x-coordinate of the bounding box start point")
    y1: float = Field(..., description="y-coordinate of the bounding box start point")
    x2: float = Field(..., description="x-coordinate of the bounding box end point")
    y2: float = Field(..., description="y-coordinate of the bounding box end point")

class DetectedObject(BaseModel):
    track_id: Optional[int] = Field(None, description="Unique ID for object tracking")
    label: str = Field(..., description="Name of the detected object")
    confidence: float = Field(..., description="Confidence score")
    box: BoundingBox = Field(..., description="Bounding box coordinates")
    is_important: bool = Field(False, description="Whether the object is important")
    status: Optional[str] = Field(None, description="Object status (e.g., 'Fast Approaching')")
    size_ratio: Optional[float] = Field(None, description="Current to past size ratio of the object (for debugging)")
    # [추가] 객체의 방향 정보를 담을 필드
    direction: Optional[str] = Field(None, description="Direction of the object (left, center, right)")

class AnalysisResponse(BaseModel):
    objects: List[DetectedObject] = Field(..., description="List of detected objects")
    guide_message: Optional[str] = Field(None, description="Situational analysis guide message")
    alert_level: AlertLevel = Field(AlertLevel.NONE, description="Risk level of the situation")

class QuestionResponse(BaseModel):
    answer: str = Field(..., description="Gemma's answer to the user's question")
