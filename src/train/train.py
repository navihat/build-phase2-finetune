"""Fine-tune encoder phân loại quan hệ 6 lớp trên trackB_silver.

Ba điểm không mặc định của script này:

1. ĐỐI XỨNG. Quan hệ giữa hai claim không phụ thuộc claim nào đứng trước. Toàn bộ
   sự cố order-flip của pipeline ensemble (94% số cặp phải đi debate chỉ vì có model
   đổi ý khi đảo A/B) đến từ chỗ này. Ở đây xử lý bằng hai cơ chế:
   - huấn luyện: mỗi cặp được nhân đôi theo hai thứ tự (--symmetric-aug)
   - suy luận:  cộng logit của hai thứ tự rồi mới argmax (--symmetric-tta)
   Nhờ vậy model đối xứng theo cấu trúc chứ không phải "hi vọng nó tự học được".

2. TRỌNG SỐ LỚP. CONTRADICTION chỉ chiếm ~2.6%. Không có trọng số thì model bỏ hẳn
   lớp này mà accuracy vẫn đẹp.

3. NHÓM THEO PAPER. Dùng split/fold do split_trackb.py sinh ra, không chia lại
   ngẫu nhiên ở đây.

Cài (một lần):
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers scikit-learn

Chạy:
    python train_relation_classifier.py                     # train/val/test một lần
    python train_relation_classifier.py --cv                # 5-fold, đọc metric ổn định
    python train_relation_classifier.py --model xlm-roberta-base   # khi có dữ liệu VI
"""
import argparse, json, collections
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)

ROOT = Path(__file__).resolve().parents[2] / "phase2_trackb"
SPLITS = ROOT / "processed" / "splits"
LABELS = ["AGREEMENT", "PARTIAL_AGREEMENT", "COMPLEMENTARY",
          "PARTIAL_CONTRADICTION", "CONTRADICTION", "UNRELATED"]
L2I = {l: i for i, l in enumerate(LABELS)}


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class PairSet(Dataset):
    def __init__(self, rows, tok, max_len, symmetric_aug=False):
        self.ex = []
        for r in rows:
            y = L2I[r["relation"]]
            a, b = r["left"]["text"], r["right"]["text"]
            self.ex.append((a, b, y))
            if symmetric_aug:
                self.ex.append((b, a, y))          # cùng nhãn, đảo thứ tự
        self.tok, self.max_len = tok, max_len

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        a, b, y = self.ex[i]
        enc = self.tok(a, b, truncation=True, max_length=self.max_len, padding=False)
        enc["labels"] = y
        return enc


def make_collate(tok):
    def collate(batch):
        labels = torch.tensor([b.pop("labels") for b in batch])
        out = tok.pad(batch, return_tensors="pt")
        out["labels"] = labels
        return out
    return collate


@torch.no_grad()
def predict(model, rows, tok, args, device):
    """Trả về logits. Với --symmetric-tta thì cộng logit của cả hai thứ tự."""
    model.eval()
    orders = [(0, 1)] + ([(1, 0)] if args.symmetric_tta else [])
    total = None
    for lo, ro in orders:
        logits = []
        for i in range(0, len(rows), args.eval_batch):
            chunk = rows[i:i + args.eval_batch]
            texts = [(r["left"]["text"], r["right"]["text"]) for r in chunk]
            a = [t[lo] for t in texts]
            b = [t[ro] for t in texts]
            enc = tok(a, b, truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(device)
            logits.append(model(**enc).logits.float().cpu())
        logits = torch.cat(logits)
        total = logits if total is None else total + logits
    return total.numpy()


def train_one(train_rows, eval_rows, args, device, tag=""):
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS)).to(device)

    ds = PairSet(train_rows, tok, args.max_len, args.symmetric_aug)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=make_collate(tok), drop_last=False)

    # trọng số lớp: inverse-frequency tính trên chính tập train của fold này
    cnt = collections.Counter(r["relation"] for r in train_rows)
    w = torch.tensor([len(train_rows) / (len(LABELS) * max(cnt.get(l, 0), 1))
                      for l in LABELS], dtype=torch.float, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    steps = len(dl) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")

    for ep in range(args.epochs):
        model.train()
        running = 0.0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.fp16 and device.type == "cuda"):
                logits = model(**batch).logits
                loss = loss_fn(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
        print(f"  {tag}epoch {ep+1}/{args.epochs}  loss={running/len(dl):.4f}")

    logits = predict(model, eval_rows, tok, args, device)
    y_true = np.array([L2I[r["relation"]] for r in eval_rows])
    y_pred = logits.argmax(1)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return y_true, y_pred


def show(y_true, y_pred, title):
    present = sorted(set(y_true) | set(y_pred))
    print(f"\n=== {title} ===")
    print(classification_report(y_true, y_pred, labels=present,
                                target_names=[LABELS[i] for i in present],
                                digits=3, zero_division=0))
    print(f"macro-F1 = {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}   "
          f"micro-F1 = {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    print("\nma trận nhầm lẫn (hàng = thật, cột = dự đoán):")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    print(" " * 24 + "".join(f"{l[:6]:>8}" for l in LABELS))
    for i, l in enumerate(LABELS):
        print(f"{l:<24}" + "".join(f"{v:>8}" for v in cm[i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base",
                    help="roberta-base / microsoft/deberta-v3-small (EN); "
                         "xlm-roberta-base (khi có dữ liệu VI)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--symmetric-aug", action="store_true", default=True)
    ap.add_argument("--no-symmetric-aug", dest="symmetric_aug", action="store_false")
    ap.add_argument("--symmetric-tta", action="store_true", default=True)
    ap.add_argument("--no-symmetric-tta", dest="symmetric_tta", action="store_false")
    ap.add_argument("--cv", action="store_true", help="chạy 5-fold thay vì train/test một lần")
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"thiết bị: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"model: {args.model} | symmetric-aug={args.symmetric_aug} "
          f"symmetric-tta={args.symmetric_tta} fp16={args.fp16}\n")

    if args.cv:
        rows = read_jsonl(ROOT / "processed" / "trackB_silver.jsonl")
        folds = json.load(open(SPLITS / "folds.json", encoding="utf-8"))
        fold_of = folds["fold_of_pair_id"]
        n = folds["n_folds"]
        all_true, all_pred = [], []
        for k in range(n):
            tr = [r for r in rows if fold_of[r["pair_id"]] != k]
            te = [r for r in rows if fold_of[r["pair_id"]] == k]
            print(f"\n--- fold {k}: train={len(tr)} test={len(te)} ---")
            yt, yp = train_one(tr, te, args, device, tag=f"[f{k}] ")
            all_true.append(yt)
            all_pred.append(yp)
            print(f"  fold {k} macro-F1 = "
                  f"{f1_score(yt, yp, average='macro', zero_division=0):.4f}")
        show(np.concatenate(all_true), np.concatenate(all_pred),
             f"{n}-FOLD GỘP (nhóm theo paper)")
    else:
        tr = read_jsonl(SPLITS / "trackB_train.jsonl")
        va = read_jsonl(SPLITS / "trackB_val.jsonl")
        te = read_jsonl(SPLITS / "trackB_test.jsonl")
        print(f"train={len(tr)} val={len(va)} test={len(te)}")
        yt, yp = train_one(tr + va, te, args, device)
        show(yt, yp, "TEST")
        print("\n⚠ test chỉ có ~110 cặp và rất ít CONTRADICTION -> con số per-class "
              "ở đây nhiễu nặng. Đọc kết quả từ --cv mới đáng tin.")


if __name__ == "__main__":
    main()
