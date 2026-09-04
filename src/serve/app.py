"""Inference server cho relation_classifier_v1 — phân loại quan hệ 6 lớp giữa
hai atomic claim (xem phase-2/README.md).

Suy luận đối xứng (symmetric-tta): cộng logit của cả hai thứ tự (left, right) và
(right, left) rồi mới argmax — đúng cách model được đánh giá lúc train, xem
predict() trong src/train/train.py. Bật/tắt qua env SYMMETRIC_TTA.

``/predict`` (1 cặp, {label,scores}) là API gốc của service này. ``/v1/relations:predict``
là endpoint thêm sau để nói đúng batch contract mà AI core / benchmark harness của team
dùng (specs/relation-classifier-contract.md ở repo team-Edtecher) — cùng model/tokenizer
đã nạp, chỉ khác lớp request/response.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "relation_classifier_v1"))
MAX_LEN = int(os.environ.get("MAX_LEN", "160"))
SYMMETRIC_TTA = os.environ.get("SYMMETRIC_TTA", "true").lower() != "false"
MODEL_ID = os.environ.get("RELATION_MODEL_ID", "reviewsynth-relation-v2-mdeberta-xnli")
MODEL_VERSION = os.environ.get("RELATION_MODEL_VERSION", "2026-09-04")
MAX_BATCH_PAIRS = 128

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


@torch.no_grad()
def _predict_pair(left_text: str, right_text: str) -> tuple[str, dict[str, float]]:
    """Core single-pair inference shared by /predict and /v1/relations:predict."""
    tok, model, device = state["tokenizer"], state["model"], state["device"]
    orders = [(left_text, right_text)] + ([(right_text, left_text)] if SYMMETRIC_TTA else [])
    total = None
    for a, b in orders:
        enc = tok(a, b, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        logits = model(**enc).logits.float().cpu()
        total = logits if total is None else total + logits
    probs = torch.softmax(total, dim=-1)[0]
    label = LABELS[int(probs.argmax())]
    scores = {l: round(float(p), 4) for l, p in zip(LABELS, probs)}
    return label, scores


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    label, scores = _predict_pair(req.left, req.right)
    return PredictResponse(label=label, scores=scores)


def _relation_error(message: str, status_code: int = 422, retryable: bool = False) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": "relation_model_error", "message": message, "retryable": retryable},
    )


@app.post("/v1/relations:predict")
def predict_relations(body: dict) -> dict:
    if body.get("schema_version") != "1.0":
        raise _relation_error("schema_version must be '1.0'")
    request_id = body.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise _relation_error("request_id must be a non-empty string")
    pairs = body.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise _relation_error("relation batch contains no pairs")
    if len(pairs) > MAX_BATCH_PAIRS:
        raise _relation_error(f"relation batch exceeds {MAX_BATCH_PAIRS} pairs: {len(pairs)}")

    predictions = []
    try:
        for pair in pairs:
            pair_id = pair["pair_id"]
            left_text = pair["left"]["text"]
            right_text = pair["right"]["text"]
            relation, scores = _predict_pair(left_text, right_text)
            predictions.append({
                "pair_id": pair_id,
                "relation": relation,
                "confidence": scores[relation],
                "explanation": None,
            })
    except (KeyError, TypeError) as exc:
        raise _relation_error(f"malformed pair in batch: {exc}") from exc

    return {
        "schema_version": "1.0",
        "request_id": request_id,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "predictions": predictions,
    }
