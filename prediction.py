"""
Core prediction logic: SAP-RPT-1 calls, risk tiers, business policy, mitigation.
"""
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import requests

from aicore import AICoreClient

# --- Business Policy Constants ---
RISK_THRESHOLD_AMBER = 1.0   # days
RISK_THRESHOLD_RED = 3.0     # days
LINE_DOWN_COST_PER_HOUR = 15_000   # USD
TYPICAL_DISRUPTION_HOURS = 8

CONTEXT_COLUMNS = [
    "Vendor_ID", "Vendor_Country", "Vendor_OTIF_Percent", "Vendor_Avg_Past_Delay",
    "Material_ID", "Material_Group", "Criticality_Flag", "Plant_ID",
    "Order_Quantity", "Net_Price", "Planned_Lead_Time_Days", "Order_Month",
    "Incoterms", "Actual_Delay_Days",
]


def derive_risk_tier(delay_days: float) -> str:
    if delay_days < RISK_THRESHOLD_AMBER:
        return "Green"
    if delay_days <= RISK_THRESHOLD_RED:
        return "Amber"
    return "Red"


def apply_business_policy(risk_tier: str, is_critical: bool) -> Tuple[str, str]:
    if risk_tier == "Red" and is_critical:
        return "CRITICAL", "Immediate mitigation required. Initiate alternative sourcing."
    if risk_tier == "Red":
        return "HIGH", "High risk detected. Review alternative suppliers."
    if risk_tier == "Amber" or is_critical:
        return "ELEVATED", "Increased monitoring. Prepare contingency plan."
    return "NORMAL", "Standard supplier monitoring. No action required."


def estimate_avoided_loss(delay_days: float, is_critical: bool) -> int:
    if not is_critical or delay_days < RISK_THRESHOLD_RED:
        return 0
    disruption_hours = min(delay_days * 4, TYPICAL_DISRUPTION_HOURS * 2)
    return int(disruption_hours * LINE_DOWN_COST_PER_HOUR)


def call_rpt1(row_dict: dict, historical_df: pd.DataFrame,
              aicore_client: AICoreClient, deployment_url: str) -> float:
    """
    Call SAP-RPT-1 inference endpoint and return the predicted delay in days.
    row_dict must contain all CONTEXT_COLUMNS except Actual_Delay_Days.
    """
    context_sample = historical_df[CONTEXT_COLUMNS].sample(
        n=min(200, len(historical_df)), random_state=42
    )
    pred_row = {**row_dict, "Actual_Delay_Days": "[PREDICT]"}
    rows = pd.concat(
        [context_sample, pd.DataFrame([pred_row])], ignore_index=True
    ).to_dict("records")

    payload = {
        "rows": rows,
        "prediction_config": {
            "target_columns": [
                {"name": "Actual_Delay_Days", "prediction_placeholder": "[PREDICT]"}
            ]
        },
    }

    url = deployment_url.rstrip("/")
    if not url.endswith("/predict"):
        url = f"{url}/predict"

    try:
        response = aicore_client.predict(url, payload)
    except requests.HTTPError as e:
        detail = ""
        if e.response is not None:
            detail = f" status={e.response.status_code} body={e.response.text[:300]}"
        raise RuntimeError(f"SAP-RPT-1 inference failed.{detail}") from e

    return float(response["predictions"][0]["Actual_Delay_Days"][0]["prediction"])


def find_alternatives(material_id: str, current_vendor: str,
                      alt_df: pd.DataFrame) -> pd.DataFrame:
    candidates = alt_df[
        (alt_df["Material_ID"] == material_id) &
        (alt_df["Alt_Vendor"] != current_vendor)
    ].copy()
    return candidates.sort_values(
        ["Lead_Time_Days", "Indicative_Unit_Price"], ascending=[True, True]
    )


def build_mitigation(request_data: dict, predicted_delay: float,
                     risk_tier: str, alt_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Return a mitigation proposal dict or None if not applicable.
    Fixed: expects snake_case keys (material_id, vendor_id, etc.).
    """
    action_level, _ = apply_business_policy(
        risk_tier, request_data.get("criticality_flag") == "Yes"
    )
    if action_level not in ("CRITICAL", "HIGH"):
        return None

    alternatives = find_alternatives(
        request_data["material_id"], request_data["vendor_id"], alt_df
    )

    if alternatives.empty:
        return {
            "status": "NO_ALTERNATIVES",
            "message": f"No alternative suppliers found for {request_data['material_id']}",
        }

    best = alternatives.iloc[0]
    qty = request_data["order_quantity"]
    price_premium = (float(best["Indicative_Unit_Price"]) - request_data["net_price"]) * qty
    avoided = estimate_avoided_loss(predicted_delay, request_data.get("criticality_flag") == "Yes")

    return {
        "status": "PROPOSAL_READY",
        "recommended_alternative": {
            "vendor": best["Alt_Vendor"],
            "country": best["Alt_Vendor_Country"],
            "lead_time_days": int(best["Lead_Time_Days"]),
            "unit_price": float(best["Indicative_Unit_Price"]),
            "stock_available": int(best["Current_Available_Stock"]),
            "can_fulfill": int(best["Current_Available_Stock"]) >= qty,
        },
        "financial_impact": {
            "price_premium": round(price_premium, 2),
            "avoided_loss": avoided,
            "net_benefit": round(avoided - max(price_premium, 0), 2),
        },
        "all_alternatives": alternatives.to_dict("records"),
    }
