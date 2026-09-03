"""Inference server cho relation_classifier_v1 — phân loại quan hệ 6 lớp giữa
hai atomic claim (xem phase-2/README.md).

Suy luận đối xứng (symmetric-tta): cộng logit của cả hai thứ tự (left, right) và
(right, left) rồi mới argmax — đúng cách model được đánh giá lúc train, xem
predict() trong src/train/train.py. Bật/tắt qua env SYMMETRIC_TTA.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "relation_classifier_v1"))
MAX_LEN = int(os.environ.get("MAX_LEN", "160"))
SYMMETRIC_TTA = os.environ.get("SYMMETRIC_TTA", "true").lower() != "false"

LABELS = ["AGREEMENT", "PARTIAL_AGREEMENT", "COMPLEMENTARY",
          "PARTIAL_CONTRADICTION", "CONTRADICTION", "UNRELATED"]

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state["device"] = device
    state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_DIR)
    state["model"] = (AutoModelForSequenceClassification
                       .from_pretrained(MODEL_DIR).to(device).eval())
    yield
    state.clear()


app = FastAPI(title="Relation Classifier", lifespan=lifespan)


class PredictRequest(BaseModel):
    left: str = Field(..., min_length=1)
    right: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    label: str
    scores: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
@torch.no_grad()
def predict(req: PredictRequest) -> PredictResponse:
    tok, model, device = state["tokenizer"], state["model"], state["device"]
    orders = [(req.left, req.right)] + ([(req.right, req.left)] if SYMMETRIC_TTA else [])
    total = None
    for a, b in orders:
        enc = tok(a, b, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        logits = model(**enc).logits.float().cpu()
        total = logits if total is None else total + logits
    probs = torch.softmax(total, dim=-1)[0]
    return PredictResponse(
        label=LABELS[int(probs.argmax())],
        scores={l: round(float(p), 4) for l, p in zip(LABELS, probs)},
    )
