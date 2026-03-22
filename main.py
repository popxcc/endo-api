from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import onnxruntime as rt
from pathlib import Path

app = FastAPI(title="EndoFertility Predictor API", version="1.0")

# 允许所有来源调用（前端在 Vercel 上）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
LR_PATH = BASE_DIR / "models" / "lr_pregnancy.onnx"
RF_PATH = BASE_DIR / "models" / "rf_livebirth.onnx"

# 冷启动时加载模型
lr_sess = rt.InferenceSession(str(LR_PATH))
rf_sess = rt.InferenceSession(str(RF_PATH))

# 根据你训练时的特征设计输入模型
class PatientInput(BaseModel):
    age: float
    inf_dur: float
    inf_type: int
    dysm: float
    amh: float
    afc: float
    endo_size: float
    bilateral: int
    treatment: int

def _predict(features: list) -> dict:
    x = np.array([features], dtype=np.float32)
    lr_prob = lr_sess.run(None, {"float_input": x})[1][0][1]
    rf_prob = rf_sess.run(None, {"float_input": x})[1][0][1]
    return {
        "pregnancy_prob": float(lr_prob),
        "livebirth_prob": float(rf_prob)
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(patient: PatientInput):
    features = [
        patient.age,
        patient.inf_dur,
        patient.inf_type,
        patient.dysm,
        patient.amh,
        patient.afc,
        patient.endo_size,
        patient.bilateral,
        patient.treatment
    ]
    result = _predict(features)
    result["note"] = "Proof-of-concept only. Not for clinical use."
    return result
