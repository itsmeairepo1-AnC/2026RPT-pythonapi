"""
Pydantic models for request/response schemas.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class POPredictRequest(BaseModel):
    """Request body for a single PO delay prediction."""
    vendor_id: str = Field(..., examples=["VENDOR_C"])
    vendor_country: str = Field(..., examples=["Vietnam"])
    vendor_otif_percent: float = Field(..., ge=0, le=100, examples=[82.0])
    vendor_avg_past_delay: float = Field(..., ge=0, examples=[3.0])
    material_id: str = Field(..., examples=["MAT-1002"])
    material_group: str = Field(..., examples=["Control Chip"])
    criticality_flag: str = Field(..., pattern="^(Yes|No)$", examples=["Yes"])
    plant_id: str = Field(..., examples=["PLANT_1000"])
    order_quantity: int = Field(..., gt=0, examples=[2200])
    net_price: float = Field(..., gt=0, examples=[195.5])
    planned_lead_time_days: int = Field(..., gt=0, examples=[9])
    order_month: int = Field(..., ge=1, le=12, examples=[11])
    incoterms: str = Field(..., examples=["FOB"])


class AlternativeSupplier(BaseModel):
    vendor: str
    country: str
    lead_time_days: int
    unit_price: float
    stock_available: int
    can_fulfill: bool


class FinancialImpact(BaseModel):
    price_premium: float
    avoided_loss: float
    net_benefit: float


class MitigationProposal(BaseModel):
    status: str
    recommended_alternative: Optional[AlternativeSupplier] = None
    financial_impact: Optional[FinancialImpact] = None
    all_alternatives: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None


class RiskSummary(BaseModel):
    predicted_delay_days: float
    risk_tier: str
    action_level: str
    recommendation: str
    is_critical: bool
    avoided_loss_usd: int


class POPredictResponse(BaseModel):
    po_context: Dict[str, Any]
    risk_summary: RiskSummary
    mitigation: Optional[MitigationProposal] = None


class VendorStats(BaseModel):
    vendor_id: str
    country: str
    avg_delay_days: float
    max_delay_days: float
    po_count: int
    otif_percent: float


class HistoricalSummary(BaseModel):
    total_pos: int
    green_count: int
    amber_count: int
    red_count: int
    green_pct: float
    amber_pct: float
    red_pct: float
    avg_delay_days: float
    vendor_stats: List[VendorStats]


class HealthResponse(BaseModel):
    status: str
    model: str
    data_loaded: bool


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: Optional[List[ChatMessage]] = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
