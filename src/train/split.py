"""Chia trackB_silver.jsonl thành train/val/test theo NHÓM PAPER.

Vì sao không chia ngẫu nhiên: nhiều cặp dùng lại cùng một claim và cùng một paper
(1099 cặp / ~263 paper). Chia ngẫu nhiên sẽ để claim của cùng một paper rơi vào cả
train lẫn test -> metric ảo. Ở đây nhóm theo paper, và kiểm tra rò rỉ ở mức TEXT
của claim (chặt hơn mức paper) rồi báo cáo lại.

Chạy:
    python split_trackb.py                 # 80/10/10 + file 5-fold
    python split_trackb.py --test-size 0.2
"""
import argparse, json, random, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "phase2_trackb"
SILVER = ROOT / "processed" / "trackB_silver.jsonl"
OUTDIR = ROOT / "processed" / "splits"
LABELS = ["AGREEMENT", "PARTIAL_AGREEMENT", "COMPLEMENTARY",
          "PARTIAL_CONTRADICTION", "CONTRADICTION", "UNRELATED"]
SEED = 20260828


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def group_of(row):
    """Khoá nhóm = paper bên trái.

    Cặp UNRELATED chéo paper có paper_id dạng 'A|B'. Không dùng union-find gộp A với B:
    110 cạnh chéo trên 263 paper sẽ nối phần lớn paper thành một thành phần liên thông
    khổng lồ, làm việc nhóm mất tác dụng. Lấy paper trái là đủ, phần rủi ro còn lại
    (claim bên phải xuất hiện ở fold khác) được đo trực tiếp bằng kiểm tra text bên dưới.
    """
    return row["paper_id"].split("|")[0]


def assign_groups(rows, ratios, seed=SEED):
    """Rải nhóm vào các phần theo tỉ lệ, ưu tiên giữ cân bằng nhãn.

    Duyệt nhóm từ lớn đến nhỏ, mỗi bước bỏ nhóm vào phần đang 'thiếu' nhất so với
    hạn ngạch -> tránh việc một nhóm lớn làm lệch hẳn một phần (điều dễ xảy ra khi
    rải ngẫu nhiên với chỉ vài trăm nhóm).
    """
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[group_of(r)].append(r)
    groups = sorted(by_group, key=lambda g: (-len(by_group[g]), g))
    rnd = random.Random(seed)
    rnd.shuffle(groups)
    groups.sort(key=lambda g: -len(by_group[g]))

    n = len(rows)
    quota = [n * x for x in ratios]
    size = [0] * len(ratios)
    out = [[] for _ in ratios]
    for g in groups:
        deficit = [(size[i] - quota[i]) / max(quota[i], 1) for i in range(len(ratios))]
        i = min(range(len(ratios)), key=lambda k: deficit[k])
        out[i].extend(by_group[g])
        size[i] += len(by_group[g])
    return out


def claim_texts(rows):
    s = set()
    for r in rows:
        s.add(r["left"]["text"])
        s.add(r["right"]["text"])
    return s


def report(name, rows, total):
    d = collections.Counter(r["relation"] for r in rows)
    parts = "  ".join(f"{k[:4]}:{d.get(k, 0)}" for k in LABELS)
    print(f"  {name:<6}{len(rows):>5} ({len(rows)/total*100:4.1f}%)  "
          f"{len({group_of(r) for r in rows}):>4} paper  |  {parts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-size", type=float, default=0.10)
    ap.add_argument("--val-size", type=float, default=0.10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rows = read_jsonl(SILVER)
    print(f"đọc {len(rows)} cặp từ {SILVER.name}")
    print(f"nhóm (paper): {len({group_of(r) for r in rows})}\n")

    tr_size = 1 - args.test_size - args.val_size
    train, val, test = assign_groups(rows, [tr_size, args.val_size, args.test_size], args.seed)

    print("PHÂN CHIA")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        report(name, part, len(rows))

    # --- kiểm tra rò rỉ: không paper nào, không claim nào nằm ở hai phần ---
    gt, gv, gs = ({group_of(r) for r in p} for p in (train, val, test))
    assert not (gt & gv) and not (gt & gs) and not (gv & gs), "paper bị lọt sang phần khác"
    ct, cv, cs = (claim_texts(p) for p in (train, val, test))
    leak_v, leak_s = len(ct & cv), len(ct & cs)
    print(f"\nrò rỉ mức paper : 0 (đã assert)")
    print(f"rò rỉ mức claim : train∩val = {leak_v}, train∩test = {leak_s}"
          + ("  <- do cặp UNRELATED chéo paper, xem group_of()" if leak_v or leak_s else "  (sạch)"))

    for name, part in [("train", train), ("val", val), ("test", test)]:
        write_jsonl(OUTDIR / f"trackB_{name}.jsonl", part)

    # --- gán fold cho cross-validation, cũng theo nhóm paper ---
    parts = assign_groups(rows, [1 / args.folds] * args.folds, args.seed)
    fold_of = {}
    for i, part in enumerate(parts):
        for r in part:
            fold_of[r["pair_id"]] = i
    with open(OUTDIR / "folds.json", "w", encoding="utf-8") as f:
        json.dump({"n_folds": args.folds, "seed": args.seed, "fold_of_pair_id": fold_of},
                  f, ensure_ascii=False, indent=1)

    print(f"\n{args.folds}-fold (nhóm theo paper):")
    for i, part in enumerate(parts):
        report(f"f{i}", part, len(rows))

    # --- trọng số lớp cho loss, tính trên tập train ---
    d = collections.Counter(r["relation"] for r in train)
    w = {k: len(train) / (len(LABELS) * max(d.get(k, 0), 1)) for k in LABELS}
    with open(OUTDIR / "class_weights.json", "w", encoding="utf-8") as f:
        json.dump({"labels": LABELS, "weights": [round(w[k], 4) for k in LABELS],
                   "counts_train": {k: d.get(k, 0) for k in LABELS}}, f, indent=1)
    print("\ntrọng số lớp (inverse-frequency, tính trên train):")
    for k in LABELS:
        print(f"  {k:<24}n={d.get(k, 0):<5} w={w[k]:.2f}")

    print(f"\nđã ghi -> {OUTDIR}")


if __name__ == "__main__":
    main()
