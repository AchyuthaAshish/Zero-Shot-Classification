"""Pydantic Schemas for Defect Report History API (GET /api/v1/history).

Conforms to Phase 3 Step 3.3:
- Versioned, structured response envelope for defect report history lists and details
- Strict exposure of persistent audit fields without disclosure of internal secrets
- Complete fidelity to existing Supabase defect_reports and defect_report_items tables
- Clear distinction between single-defect, multi-defect, and unknown report states
"""

from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class DefectReportSummarySchema(BaseModel):
    """Summary of a stored defect classification report."""
    id: str = Field(description="Unique UUID of the defect report")
    reporter_name: Optional[str] = Field(default=None, description="Name of employee who reported the defect")
    employee_id: Optional[str] = Field(default=None, description="Identifier of the reporting employee")
    defect_description: Optional[str] = Field(default=None, description="Original raw defect description text")
    category: Optional[str] = Field(default=None, description="Primary taxonomy category or Unknown")
    confidence: Optional[float] = Field(default=None, description="Report confidence score")
    confidence_level: Optional[str] = Field(default=None, description="Qualitative confidence tier: High, Medium, Low, Uncertain")
    confidence_range: Optional[str] = Field(default=None, description="Confidence range description (e.g. '~94%', 'Qualitative / N/A')")
    calibrated_score: Optional[float] = Field(default=None, description="Calibrated statistical probability if available")
    raw_score: Optional[float] = Field(default=None, description="Raw uncalibrated model score")
    top2_margin: Optional[float] = Field(default=None, description="Margin between top-1 and top-2 class probabilities")
    reliability: Optional[str] = Field(default=None, description="Reliability level (High, Medium, Low)")
    explanation: Optional[str] = Field(default=None, description="Classification rationale / natural language explanation")
    classification_mode: Optional[str] = Field(default=None, description="Classification mode: local, gemini, or hybrid")
    provider: Optional[str] = Field(default=None, description="Inference provider used")
    model: Optional[str] = Field(default=None, description="Inference model identifier")
    is_multi_defect: bool = Field(default=False, description="True if report contains 2 or more co-occurring defects; False otherwise")
    defect_count: int = Field(default=1, description="Number of distinct defect segments identified (0 for Unknown)")
    validation_status: Optional[str] = Field(default=None, description="Structural validation status: VALID, PARTIAL, INVALID, UNKNOWN")
    created_at: Optional[str] = Field(default=None, description="ISO-8601 UTC creation timestamp")

    model_config = ConfigDict(from_attributes=True)


class HistoryListData(BaseModel):
    """Data payload for history list query."""
    items: List[DefectReportSummarySchema] = Field(default_factory=list, description="List of defect report summaries")
    total: int = Field(description="Total count of reports returned in the current page")
    limit: int = Field(description="Requested page size limit")
    offset: int = Field(description="Requested pagination offset")


class HistoryListResponse(BaseModel):
    """Standardized API envelope for GET /api/v1/history."""
    success: bool = Field(default=True, description="Indicates whether the history query succeeded")
    data: HistoryListData = Field(description="History records data payload")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {
                    "items": [
                        {
                            "id": "a0000000-0000-0000-0000-000000000001",
                            "reporter_name": "Jane Doe",
                            "employee_id": "EMP-1001",
                            "defect_description": "Hydraulic line pressure drop in sector 4.",
                            "category": "Hydraulic System Defect",
                            "confidence": 0.952,
                            "confidence_level": "High",
                            "confidence_range": "~95%",
                            "calibrated_score": 0.952,
                            "raw_score": 0.952,
                            "top2_margin": 0.71,
                            "reliability": "High",
                            "explanation": "Pressure drop in hydraulic line indicates hydraulic failure.",
                            "classification_mode": "local",
                            "provider": "local",
                            "model": "Local ML (TF-IDF / MiniLM)",
                            "is_multi_defect": False,
                            "defect_count": 1,
                            "validation_status": "VALID",
                            "created_at": "2026-10-05T10:00:00Z"
                        }
                    ],
                    "total": 1,
                    "limit": 50,
                    "offset": 0
                }
            }
        }
    )


class DefectReportItemSchema(BaseModel):
    """Individual child defect segment record for multi-defect reports."""
    id: Optional[str] = Field(default=None, description="Child defect segment UUID")
    report_id: str = Field(description="Parent defect report UUID")
    defect_index: int = Field(description="1-based defect index within report")
    defect_id: Optional[int] = Field(default=None, description="Domain defect identifier")
    segment_id: Optional[int] = Field(default=None, description="Domain segment identifier")
    defect_text: str = Field(description="Isolated text of the specific defect segment")
    start_char: Optional[int] = Field(default=None, description="Start character offset in original text")
    end_char: Optional[int] = Field(default=None, description="End character offset in original text")
    category: str = Field(description="Classified taxonomy category for this specific defect")
    confidence: Optional[float] = Field(default=None, description="Defect confidence score")
    confidence_level: Optional[str] = Field(default=None, description="Confidence tier: High, Medium, Low, Uncertain")
    confidence_range: Optional[str] = Field(default=None, description="Confidence range description")
    raw_score: Optional[float] = Field(default=None, description="Raw model score")
    calibrated_score: Optional[float] = Field(default=None, description="Calibrated probability")
    top2_margin: Optional[float] = Field(default=None, description="Separation margin between top-2 classes")
    reliability: Optional[str] = Field(default=None, description="Reliability tier")
    explanation: Optional[str] = Field(default=None, description="Explanation specific to this defect segment")
    classification_mode: Optional[str] = Field(default=None, description="Classification mode applied")
    provider: Optional[str] = Field(default=None, description="Inference provider")
    model: Optional[str] = Field(default=None, description="Inference model")
    status: Optional[str] = Field(default="success", description="Classification status")
    is_ambiguous: bool = Field(default=False, description="Whether ambiguity was detected for this segment")
    ambiguity_reason: Optional[str] = Field(default=None, description="Ambiguity explanation if ambiguous")
    created_at: Optional[str] = Field(default=None, description="ISO-8601 UTC creation timestamp")

    model_config = ConfigDict(from_attributes=True)


class DefectReportDetailData(DefectReportSummarySchema):
    """Detailed defect report data payload including all child defect segments."""
    defects: List[DefectReportItemSchema] = Field(
        default_factory=list,
        description="Child defect items (populated for multi-defect reports; empty or single item for single-defect)"
    )


class HistoryDetailResponse(BaseModel):
    """Standardized API envelope for GET /api/v1/history/{report_id}."""
    success: bool = Field(default=True, description="Indicates whether report retrieval succeeded")
    data: DefectReportDetailData = Field(description="Detailed defect report payload with child defect items")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {
                    "id": "a0000000-0000-0000-0000-000000000002",
                    "reporter_name": "John Smith",
                    "employee_id": "EMP-1002",
                    "defect_description": "Conveyor belt slipping and motor overheating.",
                    "category": None,
                    "confidence": None,
                    "confidence_level": None,
                    "confidence_range": None,
                    "calibrated_score": None,
                    "raw_score": None,
                    "top2_margin": None,
                    "reliability": "High",
                    "explanation": "Identified 2 co-occurring defect signals.",
                    "classification_mode": "local",
                    "provider": "local",
                    "model": "Local ML (TF-IDF / MiniLM)",
                    "is_multi_defect": True,
                    "defect_count": 2,
                    "validation_status": "VALID",
                    "created_at": "2026-10-05T10:15:00Z",
                    "defects": [
                        {
                            "id": "b0000000-0000-0000-0000-000000000001",
                            "report_id": "a0000000-0000-0000-0000-000000000002",
                            "defect_index": 1,
                            "defect_id": 1,
                            "segment_id": 1,
                            "defect_text": "Conveyor belt slipping",
                            "start_char": 0,
                            "end_char": 22,
                            "category": "Mechanical Fault",
                            "confidence": 0.935,
                            "confidence_level": "High",
                            "confidence_range": "~94%",
                            "raw_score": 0.935,
                            "calibrated_score": 0.935,
                            "top2_margin": 0.62,
                            "reliability": "High",
                            "explanation": "Belt slipping symptom indicates mechanical traction wear.",
                            "classification_mode": "local",
                            "provider": "local",
                            "model": "Local ML (TF-IDF / MiniLM)",
                            "status": "success",
                            "is_ambiguous": False,
                            "ambiguity_reason": None,
                            "created_at": "2026-10-05T10:15:00Z"
                        },
                        {
                            "id": "b0000000-0000-0000-0000-000000000002",
                            "report_id": "a0000000-0000-0000-0000-000000000002",
                            "defect_index": 2,
                            "defect_id": 2,
                            "segment_id": 2,
                            "defect_text": "motor overheating",
                            "start_char": 27,
                            "end_char": 44,
                            "category": "Thermal Defect",
                            "confidence": 0.918,
                            "confidence_level": "High",
                            "confidence_range": "~92%",
                            "raw_score": 0.918,
                            "calibrated_score": 0.918,
                            "top2_margin": 0.58,
                            "reliability": "High",
                            "explanation": "Overheating symptom indicates thermal issue.",
                            "classification_mode": "local",
                            "provider": "local",
                            "model": "Local ML (TF-IDF / MiniLM)",
                            "status": "success",
                            "is_ambiguous": False,
                            "ambiguity_reason": None,
                            "created_at": "2026-10-05T10:15:00Z"
                        }
                    ]
                }
            }
        }
    )
