import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
DATABRICKS_ENDPOINT_URL = os.environ.get(
    "DATABRICKS_ENDPOINT_URL",
    "https://<your-workspace>.cloud.databricks.com/serving-endpoints/health-food-risk-endpoint/invocations",
)

if not DATABRICKS_TOKEN:
    raise ValueError("DATABRICKS_TOKEN not found in .env file")

LAB_COLS = [
    "lab_URXUMA", "lab_URXUMS", "lab_URXUCR_x", "lab_URXCRS", "lab_URDACT",
    "lab_LBXSAL", "lab_LBDSALSI", "lab_LBXSAPSI", "lab_LBXSASSI", "lab_LBXSATSI",
    "lab_LBXSBU", "lab_LBDSBUSI", "lab_LBXSC3SI", "lab_LBXSCA", "lab_LBDSCASI",
    "lab_LBXSCH", "lab_LBDSCHSI", "lab_LBXSCK", "lab_LBXSCLSI", "lab_LBXSCR",
    "lab_LBDSCRSI", "lab_LBXSGB", "lab_LBDSGBSI", "lab_LBXSGL", "lab_LBDSGLSI",
    "lab_LBXSGTSI", "lab_LBXSIR", "lab_LBDSIRSI", "lab_LBXSKSI", "lab_LBXSLDSI",
    "lab_LBXSNASI", "lab_LBXSOSSI", "lab_LBXSPH", "lab_LBDSPHSI", "lab_LBXSTB",
    "lab_LBDSTBSI", "lab_LBXSTP", "lab_LBDSTPSI", "lab_LBXSTR", "lab_LBDSTRSI",
    "lab_LBXSUA", "lab_LBDSUASI", "lab_LBXWBCSI", "lab_LBXLYPCT", "lab_LBXMOPCT",
    "lab_LBXNEPCT", "lab_LBXEOPCT", "lab_LBXBAPCT", "lab_LBDLYMNO", "lab_LBDMONO",
    "lab_LBDNENO", "lab_LBDEONO", "lab_LBDBANO", "lab_LBXRBCSI", "lab_LBXHGB",
    "lab_LBXHCT", "lab_LBXMCVSI", "lab_LBXMCHSI", "lab_LBXMC", "lab_LBXRDW",
    "lab_LBXPLTSI", "lab_LBXMPSI", "lab_PHQ020", "lab_PHQ030", "lab_PHQ040",
    "lab_PHQ050", "lab_PHQ060", "lab_PHAFSTHR_x", "lab_PHAFSTMN_x", "lab_PHDSESN",
    "lab_LBDHDD", "lab_LBDHDDSI", "lab_LBXHA", "lab_LBXHBS", "lab_LBXHBC",
    "lab_LBDHBG", "lab_LBDHD", "lab_LBDHEG", "lab_LBDHEM", "lab_LBXGH",
    "lab_WTSH2YR_x", "lab_LBXTC", "lab_LBDTCSI", "lab_LBXTTG", "lab_WTSH2YR_y",
    "lab_URXVOL1", "lab_URDFLOW1",
]

record = {
    "gender": 1,
    "age": 54.0,
    "race_ethnicity": 3,
    "education_level": 3,
    "medication_count": 0,
    "weight_kg": 89.5,
    "height_cm": 176.8,
    "calories": None,
    "protein_g": 95.2,
    "carbs_g": 240.1,
    "sugar_g": 88.4,
    "fiber_g": 18.0,
    "total_fat_g": 70.5,
    "saturated_fat_g": 24.3,
    "cholesterol_mg": 310.0,
    "sodium_mg": 3400.0,
    "income_to_poverty_ratio": None,
    "waist_cm": None,
    "caffeine_mg": None,
    **{col: None for col in LAB_COLS},
}

print("Sending request to Databricks...")
response = requests.post(
    DATABRICKS_ENDPOINT_URL,
    headers={
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    },
    json={"dataframe_records": [record]},
    timeout=120,
)

print(f"Status: {response.status_code}")
print(json.dumps(response.json(), indent=2))
