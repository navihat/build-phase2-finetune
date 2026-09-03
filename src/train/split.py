"""Chia trackB_silver.jsonl thành train/val/test theo NHÓM PAPER, có stratified nhãn.

Vì sao không chia ngẫu nhiên: nhiều cặp dùng lại cùng một claim và cùng một paper
(1099 cặp / ~263 paper). Chia ngẫu nhiên sẽ để claim của cùng một paper rơi vào cả
train lẫn test -> metric ảo. Ở đây nhóm theo paper, và kiểm tra rò rỉ ở mức TEXT
của claim (chặt hơn mức paper) rồi báo cáo lại.

Vì sao phải stratified (sửa 2026-08-29): bản trước chỉ cân bằng TỔNG số cặp, không đọc
`relation`. Nhãn tương quan mạnh trong cùng paper nên nhãn hiếm dồn cục -> test từng chỉ
có 2 mẫu CONTRADICTION, fold lệch 13 lần. Xem docs/TRAIN.md muc 2.

Chạy:
    python src/train/split.py                 # 80/10/10 + file 5-fold
    python src/train/split.py --test-size 0.2
"""
import argparse, json, random, collections, sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):     # console Windows mặc định cp1252, không in nổi tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2] / "pipeline_data"
SILVER = ROOT / "processed" / "trackB_silver.jsonl"
OUTDIR = ROOT / "processed" / "splits"
REPORT = ROOT / "reports" / "split_report.md"
LABELS = ["AGREEMENT", "PARTIAL_AGREEMENT", "COMPLEMENTARY",
          "PARTIAL_CONTRADICTION", "CONTRADICTION", "UNRELATED"]
SEED = 20260828

# Số hạng tổng-kích-thước trong hàm chi phí. Mỗi nhãn đã đóng góp một số hạng chuẩn hoá
# theo hạn ngạch của chính nó, nên lớp hiếm tự động nặng hơn; hệ số này giữ cho việc chạy
# theo nhãn không kéo lệch hẳn kích thước các phần.
#
# 6.0 chứ không phải 3.0 (đổi 2026-08-29, sau khi gán lại nhãn theo contract gold): với phân
# bố mới, 3.0 đẩy CONTRADICTION thành 5/5/2/5/5 giữa các fold VÀ để kích thước fold lệch
# 172-201. 6.0 cho 4/5/4/5/4 và 186-193 — tốt hơn ở CẢ HAI mặt nên không phải đánh đổi.
SIZE_W = 6.0

# Nguồn buộc phải nằm nguyên trong train.
# 39 cặp `fewshot_human_pairs` CHÍNH LÀ 39 ví dụ trong processed/fewshot.jsonl đã dùng làm
# few-shot khi Claude gán nhãn 1060 cặp còn lại (đối chiếu 2026-08-29: trùng 39/39 cả text
# lẫn nhãn). Nhãn của toàn bộ phần còn lại được sinh ra CÓ ĐIỀU KIỆN trên chúng, nên để
# chúng vào val/test là chấm điểm model trên chính các ví dụ đã định nghĩa ra nhãn của
# tập kiểm tra -> con số đẹp giả. Trước bản này chúng rơi vào train do may (cùng một
# `paper_id` giả 'HUMAN_ANNOTATED' nên thành nhóm lớn nhất), không do ràng buộc nào.
# Tắt bằng --no-pin-fewshot nếu muốn đo thử.
PIN_TRAIN_SOURCES = {"fewshot_human_pairs"}


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


def _profile(items):
    """Vector đếm: [n mỗi nhãn..., tổng]."""
    v = [0.0] * (len(LABELS) + 1)
    idx = {l: i for i, l in enumerate(LABELS)}
    for r in items:
        v[idx[r["relation"]]] += 1
        v[-1] += 1
    return v


def _dev(cur, tgt):
    """Độ lệch của MỘT phần so với hạn ngạch của nó.

    Mỗi nhãn được chuẩn hoá theo hạn ngạch của chính nhãn đó, nên lệch 2 mẫu ở lớp
    29-mẫu bị phạt nặng hơn hẳn lệch 2 mẫu ở lớp 496-mẫu. Đây chính là điều bản cũ
    không làm, và là lý do CONTRADICTION từng dồn hết vào một fold.
    """
    K = len(LABELS)
    s = sum(((cur[c] - tgt[c]) / max(tgt[c], 1.0)) ** 2 for c in range(K))
    return s + SIZE_W * ((cur[K] - tgt[K]) / max(tgt[K], 1.0)) ** 2


def assign_groups(rows, ratios, seed=SEED, passes=8):
    """Rải nhóm paper vào các phần, cân bằng theo TỪNG NHÃN chứ không chỉ tổng số cặp.

    Hai bước:
      1. Greedy — duyệt nhóm từ lớn đến nhỏ, bỏ vào phần làm tổng độ lệch tăng ít nhất.
         Nhóm lớn đi trước vì chúng ràng buộc chặt nhất, để cuối thì không còn chỗ chữa.
      2. Tinh chỉnh — lặp lại việc thử chuyển từng nhóm sang phần khác, chỉ nhận khi
         tổng độ lệch giảm. Với chỉ ~222 nhóm, greedy một lượt còn khá xa tối ưu;
         bước này rẻ và kéo lại đáng kể, nhất là cho lớp hiếm.

    Toàn bộ quá trình tất định theo `seed` (chỉ dùng để phá thế hoà giữa các nhóm
    cùng kích thước), nên chạy lại cho ra đúng một kết quả.
    """
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[group_of(r)].append(r)

    groups = sorted(by_group)
    rnd = random.Random(seed)
    rnd.shuffle(groups)
    groups.sort(key=lambda g: -len(by_group[g]))     # sort ổn định: giữ thứ tự đã xáo khi cùng cỡ

    gprof = {g: _profile(by_group[g]) for g in groups}
    total = _profile(rows)
    P = len(ratios)
    target = [[total[c] * ratios[i] for c in range(len(total))] for i in range(P)]
    cur = [[0.0] * len(total) for _ in range(P)]
    where = {}

    def add(i, g, sign=1):
        for c, x in enumerate(gprof[g]):
            cur[i][c] += sign * x

    for g in groups:                                  # --- 1. greedy ---
        best, best_cost = 0, None
        for i in range(P):
            add(i, g)
            cost = _dev(cur[i], target[i])
            add(i, g, -1)
            if best_cost is None or cost < best_cost:
                best, best_cost = i, cost
        add(best, g)
        where[g] = best

    for _ in range(passes):                           # --- 2. tinh chỉnh ---
        moved = False
        for g in groups:
            src = where[g]
            add(src, g, -1)
            base = _dev(cur[src], target[src])
            best, best_gain = src, 0.0
            for i in range(P):
                if i == src:
                    continue
                before = _dev(cur[i], target[i])
                add(i, g)
                gain = (before + base) - (_dev(cur[i], target[i]) + _dev(cur[src], target[src]))
                add(i, g, -1)
                if gain > best_gain + 1e-12:
                    best, best_gain = i, gain
            add(best, g)
            where[g] = best
            moved |= best != src
        if not moved:
            break

    out = [[] for _ in ratios]
    for g in groups:
        out[where[g]].extend(by_group[g])
    return out


def claim_texts(rows):
    s = set()
    for r in rows:
        s.add(r["left"]["text"])
        s.add(r["right"]["text"])
    return s


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def label_table(named_parts, total_rows):
    """Bảng nhãn × phần, kèm CHÊNH LỆCH so với phân bố toàn cục.

    Cột lệch mới là cột đáng đọc: nó trả lời 'phần này có đại diện đúng cho dữ liệu
    gốc không', điều mà cột đếm thô không nói được khi các phần khác cỡ nhau.
    """
    g = collections.Counter(r["relation"] for r in total_rows)
    N = len(total_rows)
    header = ["Nhãn", "toàn bộ", "%"] + [x for nm, _ in named_parts for x in (nm, "%", "lệch")]
    body, worst = [], 0.0
    for k in LABELS:
        row = [k, g[k], f"{g[k]/N*100:.1f}%"]
        for _, p in named_parts:
            c = sum(1 for r in p if r["relation"] == k)
            pct = c / max(len(p), 1) * 100
            d = pct - g[k] / N * 100
            worst = max(worst, abs(d))
            row += [c, f"{pct:.1f}%", f"{d:+.1f}"]
        body.append(row)
    row = ["**tổng**", N, "100%"]
    for _, p in named_parts:
        row += [len(p), f"{len(p)/N*100:.1f}%", ""]
    body.append(row)
    return _md_table(header, body), worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-size", type=float, default=0.10)
    ap.add_argument("--val-size", type=float, default=0.10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-pin-fewshot", dest="pin", action="store_false",
                    help="cho phép cặp few-shot rơi vào val/test (xem PIN_TRAIN_SOURCES)")
    args = ap.parse_args()

    rows = read_jsonl(SILVER)
    n = len(rows)
    md = [f"# Báo cáo chia dữ liệu — trackB_silver\n",
          f"> Sinh tự động bởi `src/train/split.py` · seed `{args.seed}` · "
          f"{n} cặp / {len({group_of(r) for r in rows})} nhóm paper\n",
          "Chia theo **nhóm paper** (chống rò rỉ) **và** stratified theo nhãn "
          "(để lớp hiếm không dồn cục).\n"]
    print(f"đọc {n} cặp từ {SILVER.name} · {len({group_of(r) for r in rows})} nhóm paper")

    tr_size = 1 - args.test_size - args.val_size

    # Ghim few-shot vào train TRƯỚC khi chia, rồi chia phần còn lại với hạn ngạch đã trừ đi
    # số cặp bị ghim -> tỉ lệ cuối cùng vẫn đúng 80/10/10 trên tổng.
    pinned = [r for r in rows if args.pin and r["source"] in PIN_TRAIN_SOURCES]
    pin_ids = {r["pair_id"] for r in pinned}
    pool = [r for r in rows if r["pair_id"] not in pin_ids]
    quota = [max(tr_size * n - len(pinned), 0.0), args.val_size * n, args.test_size * n]
    ratios = [q / sum(quota) for q in quota]

    parts3 = assign_groups(pool, ratios, args.seed)
    train, val, test = pinned + parts3[0], parts3[1], parts3[2]
    named = [("train", train), ("val", val), ("test", test)]
    if pinned:
        src = ", ".join(sorted(PIN_TRAIN_SOURCES))
        md.append(f"**Ghim vào train:** {len(pinned)} cặp thuộc `{src}` — đó chính là các ví dụ "
                  "few-shot đã dùng để gán nhãn phần còn lại, xem `PIN_TRAIN_SOURCES` "
                  "trong `split.py`.\n")
        print(f"ghim {len(pinned)} cặp ({src}) vào train")

    # --- 1. phân bố nhãn ---
    tbl, worst = label_table(named, rows)
    md += ["\n## 1 · Phân bố nhãn\n", tbl,
           f"\nLệch lớn nhất so với phân bố toàn cục: **{worst:.1f} điểm %**.\n"]
    print(f"\nPHÂN BỐ NHÃN (lệch lớn nhất {worst:.1f} điểm %)")
    print(f"  {'nhãn':<24}{'toàn bộ':>8}" + "".join(f"{nm:>8}" for nm, _ in named))
    gcnt = collections.Counter(r["relation"] for r in rows)
    for k in LABELS:
        print(f"  {k:<24}{gcnt[k]:>8}" +
              "".join(f"{sum(1 for r in p if r['relation'] == k):>8}" for _, p in named))
    print(f"  {'tổng':<24}{n:>8}" + "".join(f"{len(p):>8}" for _, p in named))

    # --- 2. nguồn dữ liệu × phần (muc 4.6 của TRAIN.md: synthetic phải đo tách) ---
    srcs = sorted({r["source"] for r in rows})
    body = []
    for s in srcs:
        row = [f"`{s}`", sum(1 for r in rows if r["source"] == s)]
        for _, p in named:
            c = sum(1 for r in p if r["source"] == s)
            row += [c, f"{c/max(len(p),1)*100:.1f}%"]
        body.append(row)
    md += ["\n## 2 · Nguồn dữ liệu × phần\n",
           _md_table(["`source`", "toàn bộ"] + [x for nm, _ in named for x in (nm, "%")], body),
           "\nCần đọc cùng muc 4.6 của `docs/TRAIN.md`: `synthetic_cross_paper` sinh bằng luật, "
           "rất dễ đoán, nên phải đo tách chứ không để nó làm đẹp macro-F1 tổng.\n"]

    # --- 3. rò rỉ ---
    gt, gv, gs = ({group_of(r) for r in p} for p in (train, val, test))
    assert not (gt & gv) and not (gt & gs) and not (gv & gs), "paper bị lọt sang phần khác"
    ct, cv, cs = (claim_texts(p) for p in (train, val, test))
    leaks = [("train ∩ val", len(ct & cv)), ("train ∩ test", len(ct & cs)),
             ("val ∩ test", len(cv & cs))]
    md += ["\n## 3 · Kiểm tra rò rỉ\n",
           _md_table(["Mức", "Cặp phần", "Số trùng"],
                     [["paper (đã `assert`)", "mọi cặp", 0]] +
                     [["text của claim", a, b] for a, b in leaks]),
           "\nMức claim chặt hơn mức paper. Phần dư đến từ cặp UNRELATED chéo paper "
           "(`paper_id` dạng `A|B`), vì `group_of()` chỉ lấy paper bên trái — đánh đổi có ý thức, "
           "xem docstring của hàm.\n"]
    print("\nRÒ RỈ  paper: 0 (đã assert)  |  claim text: " +
          ", ".join(f"{a} = {b}" for a, b in leaks))

    for name, part in named:
        write_jsonl(OUTDIR / f"trackB_{name}.jsonl", part)

    # --- 4. gán fold cho cross-validation, cũng theo nhóm paper + stratified ---
    # Cặp bị ghim nhận fold = -1: `train.py` lọc bằng `fold_of[...] != k`, nên -1 rơi vào
    # train của MỌI fold và không bao giờ bị chấm điểm. Không phải sửa gì bên train.py.
    parts = assign_groups(pool, [1 / args.folds] * args.folds, args.seed)
    fold_of = {r["pair_id"]: -1 for r in pinned}
    for i, part in enumerate(parts):
        for r in part:
            fold_of[r["pair_id"]] = i
    with open(OUTDIR / "folds.json", "w", encoding="utf-8") as f:
        json.dump({"n_folds": args.folds, "seed": args.seed,
                   "pinned_to_train": sorted(PIN_TRAIN_SOURCES) if pinned else [],
                   "pinned_fold_id": -1,
                   "fold_of_pair_id": fold_of}, f, ensure_ascii=False, indent=1)

    pcnt = collections.Counter(r["relation"] for r in pool)
    header = ["Nhãn", "chấm điểm", "ghim→train"] + [f"f{i}" for i in range(args.folds)] + ["min–max"]
    body = []
    for k in LABELS:
        cs_ = [sum(1 for r in p if r["relation"] == k) for p in parts]
        body.append([k, pcnt[k], gcnt[k] - pcnt[k]] + cs_ + [f"{min(cs_)}–{max(cs_)}"])
    body.append(["**tổng**", len(pool), len(pinned)] + [len(p) for p in parts] +
                [f"{min(len(p) for p in parts)}–{max(len(p) for p in parts)}"])
    md += [f"\n## 4 · {args.folds}-fold (nhóm paper + stratified)\n", _md_table(header, body),
           "\nCột **ghim→train** nhận `fold = -1`: có mặt trong train của mọi fold, không bao giờ "
           "bị chấm điểm. Đây mới là bảng để đọc kết quả (muc 2 của `docs/TRAIN.md`) — "
           "test chỉ vài mẫu CONTRADICTION nên F1 lớp đó trên test vẫn không đọc được.\n"]
    print(f"\n{args.folds}-FOLD  kích thước {[len(p) for p in parts]}")
    for k in LABELS:
        cs_ = [sum(1 for r in p if r["relation"] == k) for p in parts]
        print(f"  {k:<24}" + "".join(f"{c:>6}" for c in cs_))

    # --- 5. trọng số lớp cho loss, tính trên tập train ---
    d = collections.Counter(r["relation"] for r in train)
    w = {k: len(train) / (len(LABELS) * max(d.get(k, 0), 1)) for k in LABELS}
    with open(OUTDIR / "class_weights.json", "w", encoding="utf-8") as f:
        json.dump({"labels": LABELS, "weights": [round(w[k], 4) for k in LABELS],
                   "counts_train": {k: d.get(k, 0) for k in LABELS}}, f, indent=1)
    md += ["\n## 5 · Trọng số lớp (inverse-frequency, trên train)\n",
           _md_table(["Nhãn", "n train", "w"],
                     [[k, d.get(k, 0), f"{w[k]:.2f}"] for k in LABELS]),
           "\n⚠ `train.py` phải tính lại trọng số trên train của **từng fold**, "
           "file này chỉ dùng cho lần train một-lượt.\n"]
    print("\nTRỌNG SỐ LỚP  " + "  ".join(f"{k[:4]}:{w[k]:.2f}" for k in LABELS))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"\nđã ghi -> {OUTDIR}\nbáo cáo -> {REPORT}")


if __name__ == "__main__":
    main()
