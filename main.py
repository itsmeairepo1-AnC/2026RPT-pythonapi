"""
JIT Prediction API
==================
FastAPI application exposing SAP-RPT-1 delay prediction as REST endpoints.

Endpoints:
  GET  /health                     – liveness / readiness check
  POST /api/v1/predict             – predict delay for a single PO
  GET  /api/v1/historical/summary  – historical risk distribution & vendor stats
  GET  /api/v1/vendors             – list of known vendors with their profiles
  GET  /api/v1/materials           – list of materials with alternative suppliers
  POST /api/v1/chat                – LLM-powered chat via SAP Gen AI Hub

Run locally:
  uvicorn main:app --reload --port 8000
"""
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from aicore import AICoreClient
from models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HistoricalSummary,
    MitigationProposal,
    POPredictRequest,
    POPredictResponse,
    RiskSummary,
    VendorStats,
)
from prediction import (
    apply_business_policy,
    build_mitigation,
    call_rpt1,
    derive_risk_tier,
    estimate_avoided_loss,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

AICORE_AUTH_URL = os.environ["AICORE_AUTH_URL"]
AICORE_CLIENT_ID = os.environ["AICORE_CLIENT_ID"]
AICORE_CLIENT_SECRET = os.environ["AICORE_CLIENT_SECRET"]
AICORE_BASE_URL = os.environ["AICORE_BASE_URL"]
AICORE_RESOURCE_GROUP = os.environ["AICORE_RESOURCE_GROUP"]
RPT1_DEPLOYMENT_URL = os.environ["RPT1_DEPLOYMENT_URL"]
ORCH_DEPLOYMENT_URL = os.getenv("ORCH_DEPLOYMENT_URL", "")  # optional — enables LLM chat

# Placeholder token — indicates URL has not been configured yet
_ORCH_PLACEHOLDER = "<your-orch-deployment-id>"

# Fixed: data/ is at repo root, not inside backend-api
DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# App state (loaded once at startup)
# ---------------------------------------------------------------------------
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["aicore"] = AICoreClient(
        auth_url=AICORE_AUTH_URL,
        client_id=AICORE_CLIENT_ID,
        client_secret=AICORE_CLIENT_SECRET,
        base_url=AICORE_BASE_URL,
        resource_group=AICORE_RESOURCE_GROUP,
    )
    app_state["historical_df"] = pd.read_csv(DATA_DIR / "historical_po_data.csv")
    app_state["alt_supplier_df"] = pd.read_csv(DATA_DIR / "alt_supplier_table.csv")
    yield
    app_state.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="JIT Supply Chain Risk API",
    description=(
        "Predict purchase-order delay risk using SAP-RPT-1 on SAP AI Core "
        "and surface mitigation proposals for CAP or any frontend consumer."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health():
    """Liveness + readiness check. Returns 200 when data and config are loaded."""
    return HealthResponse(
        status="ok",
        model="sap-rpt-1-small",
        data_loaded="historical_df" in app_state,
    )


@app.post("/api/v1/predict", response_model=POPredictResponse, tags=["Prediction"])
def predict(body: POPredictRequest):
    """
    Predict delivery delay for a new Purchase Order.

    Calls SAP-RPT-1 on SAP AI Core with historical context,
    applies the Green / Amber / Red business policy,
    and optionally returns an alternative-supplier mitigation proposal.
    """
    historical_df: pd.DataFrame = app_state.get("historical_df")
    alt_df: pd.DataFrame = app_state.get("alt_supplier_df")
    aicore: AICoreClient = app_state.get("aicore")

    if historical_df is None or aicore is None:
        raise HTTPException(503, "Service not ready – data not loaded.")

    row_dict = {
        "Vendor_ID": body.vendor_id,
        "Vendor_Country": body.vendor_country,
        "Vendor_OTIF_Percent": body.vendor_otif_percent,
        "Vendor_Avg_Past_Delay": body.vendor_avg_past_delay,
        "Material_ID": body.material_id,
        "Material_Group": body.material_group,
        "Criticality_Flag": body.criticality_flag,
        "Plant_ID": body.plant_id,
        "Order_Quantity": body.order_quantity,
        "Net_Price": body.net_price,
        "Planned_Lead_Time_Days": body.planned_lead_time_days,
        "Order_Month": body.order_month,
        "Incoterms": body.incoterms,
    }

    try:
        predicted_delay = call_rpt1(row_dict, historical_df, aicore, RPT1_DEPLOYMENT_URL)
    except RuntimeError as exc:
        raise HTTPException(502, detail=str(exc)) from exc

    risk_tier = derive_risk_tier(predicted_delay)
    is_critical = body.criticality_flag == "Yes"
    action_level, recommendation = apply_business_policy(risk_tier, is_critical)
    avoided_loss = estimate_avoided_loss(predicted_delay, is_critical)

    # Build mitigation proposal when risk is HIGH or CRITICAL
    # Fixed: convert to snake_case keys for build_mitigation
    request_dict_snake = {
        "vendor_id": body.vendor_id,
        "material_id": body.material_id,
        "order_quantity": body.order_quantity,
        "net_price": body.net_price,
        "criticality_flag": body.criticality_flag,
    }
    raw_mitigation = build_mitigation(request_dict_snake, predicted_delay, risk_tier, alt_df)
    mitigation = MitigationProposal(**raw_mitigation) if raw_mitigation else None

    return POPredictResponse(
        po_context=row_dict,
        risk_summary=RiskSummary(
            predicted_delay_days=predicted_delay,
            risk_tier=risk_tier,
            action_level=action_level,
            recommendation=recommendation,
            is_critical=is_critical,
            avoided_loss_usd=avoided_loss,
        ),
        mitigation=mitigation,
    )


@app.get("/api/v1/historical/summary", response_model=HistoricalSummary, tags=["Data"])
def historical_summary():
    """Return risk distribution and per-vendor stats derived from historical PO data."""
    df: pd.DataFrame = app_state.get("historical_df")
    if df is None:
        raise HTTPException(503, "Service not ready.")

    df = df.copy()
    df["Risk_Tier"] = df["Actual_Delay_Days"].apply(derive_risk_tier)
    total = len(df)
    counts = df["Risk_Tier"].value_counts().to_dict()

    vendor_stats = []
    for vendor_id, grp in df.groupby("Vendor_ID"):
        vendor_stats.append(
            VendorStats(
                vendor_id=vendor_id,
                country=grp["Vendor_Country"].iloc[0],
                avg_delay_days=round(grp["Actual_Delay_Days"].mean(), 2),
                max_delay_days=round(grp["Actual_Delay_Days"].max(), 2),
                po_count=len(grp),
                otif_percent=round(grp["Vendor_OTIF_Percent"].iloc[0], 1),
            )
        )

    return HistoricalSummary(
        total_pos=total,
        green_count=counts.get("Green", 0),
        amber_count=counts.get("Amber", 0),
        red_count=counts.get("Red", 0),
        green_pct=round(counts.get("Green", 0) / total * 100, 1),
        amber_pct=round(counts.get("Amber", 0) / total * 100, 1),
        red_pct=round(counts.get("Red", 0) / total * 100, 1),
        avg_delay_days=round(df["Actual_Delay_Days"].mean(), 2),
        vendor_stats=vendor_stats,
    )


@app.get("/api/v1/vendors", tags=["Data"])
def list_vendors():
    """Return the list of known vendors with their risk profiles."""
    df: pd.DataFrame = app_state.get("historical_df")
    if df is None:
        raise HTTPException(503, "Service not ready.")

    result = (
        df.groupby("Vendor_ID")
        .agg(
            country=("Vendor_Country", "first"),
            otif_percent=("Vendor_OTIF_Percent", "first"),
            avg_delay_days=("Actual_Delay_Days", "mean"),
        )
        .round(2)
        .reset_index()
        .rename(columns={"Vendor_ID": "vendor_id"})
        .to_dict("records")
    )
    return {"vendors": result}


@app.get("/api/v1/materials", tags=["Data"])
def list_materials():
    """Return materials and their available alternative suppliers."""
    alt_df: pd.DataFrame = app_state.get("alt_supplier_df")
    hist_df: pd.DataFrame = app_state.get("historical_df")
    if alt_df is None or hist_df is None:
        raise HTTPException(503, "Service not ready.")

    materials = (
        hist_df[["Material_ID", "Material_Group", "Criticality_Flag"]]
        .drop_duplicates("Material_ID")
        .set_index("Material_ID")
    )

    result = []
    for material_id, row in materials.iterrows():
        alts = alt_df[alt_df["Material_ID"] == material_id].to_dict("records")
        result.append(
            {
                "material_id": material_id,
                "material_group": row["Material_Group"],
                "criticality_flag": row["Criticality_Flag"],
                "alternative_suppliers": alts,
            }
        )

    return {"materials": result}


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
def chat(body: ChatRequest):
    """
    LLM-powered chat endpoint via SAP Gen AI Hub orchestration.
    Falls back to a helpful message when ORCH_DEPLOYMENT_URL is not configured.
    """
    aicore: AICoreClient = app_state.get("aicore")
    historical_df: pd.DataFrame = app_state.get("historical_df")

    session_id = body.session_id or str(uuid.uuid4())

    # Check if LLM is configured
    if not ORCH_DEPLOYMENT_URL or _ORCH_PLACEHOLDER in ORCH_DEPLOYMENT_URL:
        return ChatResponse(
            reply=(
                "⚠️ LLM chat is not yet configured. "
                "To enable it, add your SAP Gen AI Hub orchestration deployment URL "
                "to <code>backend-api/.env</code> as <code>ORCH_DEPLOYMENT_URL</code> "
                "and restart the server."
            ),
            session_id=session_id,
        )

    if aicore is None:
        raise HTTPException(503, "Service not ready.")

    # Build system prompt with live data context
    vendor_list = ""
    material_list = ""
    if historical_df is not None:
        vendors = (
            historical_df.groupby("Vendor_ID")
            .agg(country=("Vendor_Country", "first"), otif=("Vendor_OTIF_Percent", "first"))
            .reset_index()
        )
        vendor_list = ", ".join(
            f"{r.Vendor_ID} ({r.country}, OTIF {r.otif}%)" for r in vendors.itertuples()
        )
        materials = (
            historical_df[["Material_ID", "Material_Group", "Criticality_Flag"]]
            .drop_duplicates("Material_ID")
        )
        material_list = ", ".join(
            f"{r.Material_ID} ({r.Material_Group}, critical={r.Criticality_Flag})"
            for r in materials.itertuples()
        )

    system_prompt = f"""You are a JIT Supply Chain Risk Assistant for an SAP system.
You help users assess purchase order delay risks using the SAP-RPT-1 prediction model.

Available vendors: {vendor_list or 'VENDOR_A, VENDOR_B, VENDOR_C'}
Available materials: {material_list or 'MAT-1001 through MAT-1006'}
Plants: PLANT_1000
Incoterms: FOB, CIF, EXW

When a user wants to predict risk for a PO, extract: vendor_id, material_id, order_quantity,
net_price, planned_lead_time_days, order_month (1-12), plant_id, incoterms.
If any are missing, ask for them clearly.

Risk tiers: Green (<1 day delay) = safe, Amber (1-3 days) = monitor, Red (>3 days) = mitigate.
Keep responses concise and helpful. Use plain text, avoid markdown headers."""

    # Convert history to the format aicore expects
    history = [{"role": m.role, "content": m.content} for m in (body.history or [])]

    try:
        reply = aicore.chat_complete(
            deployment_url=ORCH_DEPLOYMENT_URL,
            system_prompt=system_prompt,
            history=history,
            user_message=body.message,
            max_tokens=500,
        )
    except Exception as exc:
        raise HTTPException(502, detail=f"LLM call failed: {exc}") from exc

    return ChatResponse(reply=reply, session_id=session_id)
