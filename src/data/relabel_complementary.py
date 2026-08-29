"""Trích 496 cặp COMPLEMENTARY của silver ra để gán lại theo contract GOLD.

Vì sao chỉ lớp này: đối chiếu trường `why` của silver với `annotation_note` của
golden_set cho thấy bốn lớp kia (AGREEMENT / PARTIAL_AGREEMENT / PARTIAL_CONTRADICTION /
CONTRADICTION) đã khớp contract gold. Riêng COMPLEMENTARY thì rubric cũ định nghĩa là
"cùng aspect nhưng KHÁC điểm cụ thể" (494/496 cặp ghi đúng câu đó trong `why`), mà gold
gọi tình huống đó là UNRELATED. Xem docs/RUBRIC_GOLD.md muc 3.

Phép thử để gán lại, đúng một câu:
    Có tồn tại MỘT vấn đề chung mà cả hai claim đều nhắm tới không?
      có    -> COMPLEMENTARY   (hai mặt của cùng một thiếu sót)
      không -> UNRELATED       (hai chủ đề rời, chỉ chung cái nhãn aspect)

Chạy:
    python src/data/relabel_complementary.py                  # xuất batch 50 cặp/file
    python src/data/relabel_complementary.py --batch-size 100
    python src/data/relabel_complementary.py --apply decisions.jsonl   # ghi ngược vào silver
"""
import argparse, json, collections, sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2] / "phase2_trackb"
SILVER = ROOT / "processed" / "trackB_silver.jsonl"
GOLD = ROOT / "golden_set" / "gold_test.jsonl"
OUTDIR = ROOT / "interim" / "relabel"
DROP_SOURCES = {"synthetic_cross_paper"}     # xem RUBRIC_GOLD.md muc 3


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fewshot_from_gold(n_each=3):
    """Lấy ví dụ COMPLEMENTARY và UNRELATED từ chính gold làm mỏ neo.

    Dùng gold chứ không tự viết ví dụ: contract nằm ở đó, không nằm ở trí nhớ của ai.
    """
    gold = read_jsonl(GOLD)
    out = []
    for lab in ("COMPLEMENTARY", "UNRELATED"):
        for r in [x for x in gold if x["expected_relation"] == lab][:n_each]:
            out.append({"label": lab,
                        "left": r["left"]["text"],
                        "right": r["right"]["text"],
                        "why": r.get("annotation_note", "")})
    return out


def emit(args):
    rows = read_jsonl(SILVER)
    todo = [r for r in rows
            if r["relation"] == "COMPLEMENTARY" and r["source"] not in DROP_SOURCES]
    todo.sort(key=lambda r: r["pair_id"])          # tất định, batch chạy lại vẫn như cũ

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(OUTDIR / "_rubric.json", "w", encoding="utf-8") as f:
        json.dump({"test": "Có tồn tại MỘT vấn đề chung mà cả hai claim đều nhắm tới không?",
                   "yes": "COMPLEMENTARY", "no": "UNRELATED",
                   "contract": "docs/RUBRIC_GOLD.md",
                   "fewshot_from_gold": fewshot_from_gold()}, f, ensure_ascii=False, indent=1)

    n = args.batch_size
    batches = [todo[i:i + n] for i in range(0, len(todo), n)]
    for i, b in enumerate(batches):
        write_jsonl(OUTDIR / f"batch{i:02d}.jsonl",
                    [{"pair_id": r["pair_id"],
                      "aspect": r["aspect"],
                      "left": r["left"]["text"],
                      "right": r["right"]["text"],
                      "why_cu": r.get("why", ""),
                      "relation_moi": None,          # điền COMPLEMENTARY hoặc UNRELATED
                      "why_moi": None} for r in b])

    print(f"{len(todo)} cặp COMPLEMENTARY cần gán lại -> {len(batches)} batch × {n}")
    print(f"ghi vào {OUTDIR}")
    print(f"  _rubric.json   phép thử + 6 ví dụ lấy từ gold")
    print(f"  batch00..{len(batches)-1:02d}   điền `relation_moi` rồi gộp lại thành decisions.jsonl")
    drop = sum(1 for r in rows if r["source"] in DROP_SOURCES)
    print(f"\nlưu ý: {drop} cặp {sorted(DROP_SOURCES)} sẽ bị BỎ khi --apply")


def apply(args):
    rows = read_jsonl(SILVER)
    dec = {d["pair_id"]: d for d in read_jsonl(Path(args.apply))}
    missing = [d for d in dec.values() if d.get("relation_moi") not in
               ("COMPLEMENTARY", "UNRELATED")]
    if missing:
        sys.exit(f"LỖI: {len(missing)} quyết định thiếu/sai `relation_moi`, "
                 f"ví dụ {missing[0]['pair_id']}")

    todo = {r["pair_id"] for r in rows
            if r["relation"] == "COMPLEMENTARY" and r["source"] not in DROP_SOURCES}
    if todo - set(dec):
        sys.exit(f"LỖI: còn {len(todo - set(dec))} cặp chưa có quyết định")

    out, changed, dropped = [], collections.Counter(), 0
    for r in rows:
        if r["source"] in DROP_SOURCES:
            dropped += 1
            continue
        if r["pair_id"] in dec:
            d = dec[r["pair_id"]]
            if d["relation_moi"] != r["relation"]:
                changed[f"{r['relation']} -> {d['relation_moi']}"] += 1
            r = {**r, "relation": d["relation_moi"],
                 "why": d.get("why_moi") or r.get("why", ""),
                 "provenance": "relabel_to_gold_contract_v1"}
        out.append(r)

    write_jsonl(SILVER, out)
    print(f"đã ghi {len(out)} cặp -> {SILVER.name}  (bỏ {dropped} cặp {sorted(DROP_SOURCES)})")
    for k, v in changed.most_common():
        print(f"  {k}: {v}")
    d = collections.Counter(r["relation"] for r in out)
    print("\nphân bố mới:")
    for k, v in d.most_common():
        print(f"  {k:<24}{v:>5}  ({v/len(out):.1%})")
    print("\n-> chạy lại: python src/train/split.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--apply", metavar="decisions.jsonl",
                    help="ghi quyết định ngược vào trackB_silver.jsonl")
    args = ap.parse_args()
    apply(args) if args.apply else emit(args)


if __name__ == "__main__":
    main()
