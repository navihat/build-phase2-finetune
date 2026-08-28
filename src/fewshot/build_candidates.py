"""Dựng tập ỨNG VIÊN few-shot từ nhãn người trong Human_Annotated_Data.json.

Chạy local, không cần GPU, không cần Colab:

    python phase-2/build_fewshot_candidates.py

Output: phase-2/fewshot_candidates.jsonl

QUAN TRỌNG — đây là ỨNG VIÊN, không phải nhãn cuối. Trường `label` chỉ là ĐỀ XUẤT
suy từ `intensity.score` của annotator, mà `intensity` đo ĐỘ TƯƠNG PHẢN chứ không
phải trục hedging của contract. Người phải đọc và sửa từng dòng trước khi dùng.
Few-shot là neo hiệu chuẩn người DUY NHẤT còn lại của pipeline (PHASE2_PLAN.md §1);
tự động hoá bước xác nhận này là tự tay tháo mất chính cái neo đó.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "raw" / "Human_Annotated_Data.json"
OUT = ROOT / "data" / "fewshot" / "candidates.jsonl"

# Aspect hợp lệ theo B2_SCHEMA trong track_b_pipeline.ipynb
VALID_ASPECTS = {
    "clarity", "motivation", "substance",
    "soundness", "originality", "meaningful comparison",
}

# intensity.score -> nhãn ĐỀ XUẤT. Chỉ là điểm khởi đầu cho người đọc.
#   3 = tương phản cao      -> nhiều khả năng CONTRADICTION
#   2 = tương phản vừa      -> ranh giới, cần đọc kỹ nhất
#   1 = tương phản thấp     -> nhiều khả năng PARTIAL_CONTRADICTION
SCORE_TO_LABEL = {
    3: "CONTRADICTION",
    2: "PARTIAL_CONTRADICTION",
    1: "PARTIAL_CONTRADICTION",
}

# Số ứng viên xuất ra mỗi (nhãn đề xuất, score) — dư ra để người còn chỗ loại.
PER_BUCKET = 12


def evidence_pair(ev):
    """evidence có nhiều dạng trong cùng một file: list[2], và dict với key đặt tên
    không thống nhất ('review1', 'Review 1', 'review_1'). Chuẩn hoá: bất kỳ dict
    đúng 2 key nào cũng lấy theo thứ tự review 1 rồi review 2."""
    if isinstance(ev, list) and len(ev) == 2:
        return ev[0], ev[1]
    if isinstance(ev, dict) and len(ev) == 2:
        ordered = sorted(ev.items(), key=lambda kv: kv[0].lower().replace(" ", "").replace("_", ""))
        return ordered[0][1], ordered[1][1]
    return None


def clean(t):
    return re.sub(r"\s+", " ", (t or "")).strip()


# Một số evidence quote bị nhúng sẵn tiền tố "Review 1: '...'" ngay trong text.
# Để nguyên thì few-shot dạy model rằng claim có tiền tố đó — trong khi claim thật
# từ B2 không bao giờ có. Phải bóc.
PREFIX = re.compile(r"^\s*Review\s*\d+\s*[:\-–]\s*", re.I)


def clean_quote(t):
    t = clean(t)
    t = PREFIX.sub("", t)
    if len(t) >= 2 and t[0] in "'\"" and t[-1] in "'\"":
        t = t[1:-1].strip()
    return t


# Giữ đồng bộ với REBUTTAL_MARKERS trong track_b_pipeline.ipynb (B1). B1 cắt bỏ
# mọi thứ từ marker đầu tiên trở đi, nên claim hậu-rebuttal không bao giờ tới được B4.
REBUTTAL_MARKERS = [
    "update after", "after reading the author", "post-rebuttal", "after rebuttal",
    "i have read the author", "-- i have read", "post revision update",
    "post-revision update", "after the rebuttal", "in the rebuttal",
    "the author addressed", "author response",
]


def offdistribution(text):
    """Quote không thể sinh ra từ B2 -> làm few-shot thì lệch phân phối.

    B2 đặt is_evaluative=false cho câu hỏi thuần, guard 2 tầng đòi claim đứng một
    mình đọc vẫn hiểu, và B1 đã cắt sạch nội dung hậu-rebuttal. Quote vi phạm ba
    điều đó dạy model một kiểu input mà pipeline không bao giờ sinh ra.
    """
    low = text.lower()
    if any(m in low for m in REBUTTAL_MARKERS):
        return "hau_rebuttal"
    if text.endswith("?") and not re.search(r"\b(but|however|although|though)\b", text, re.I):
        return "cau_hoi_thuan"
    if len(text.split()) < 6:
        return "qua_ngan_thieu_ngu_canh"
    return None


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))

    buckets = defaultdict(list)
    skipped = Counter()
    aspect_fixed = Counter()

    for paper_id, entry in data.items():
        for item in entry.get("analysis", []):
            pair = evidence_pair(item.get("evidence"))
            if not pair:
                skipped["evidence_shape"] += 1
                continue

            left, right = clean_quote(pair[0]), clean_quote(pair[1])
            if not left or not right:
                skipped["empty_quote"] += 1
                continue

            flags = [f for f in (offdistribution(left), offdistribution(right)) if f]

            intensity = item.get("intensity") or {}
            score = intensity.get("score")
            score = int(score) if isinstance(score, (int, float)) else None
            label = SCORE_TO_LABEL.get(score)
            if label is None:
                skipped["no_score"] += 1
                continue

            aspect = clean(item.get("aspect")).lower()
            if aspect not in VALID_ASPECTS:
                # 'Originality' -> 'originality' đã xử lý bằng .lower().
                # 'presentation' KHÔNG có trong enum của B2 -> đánh dấu để người quyết định.
                aspect_fixed[aspect] += 1
                aspect = f"REVIEW_ME:{aspect}"

            if flags:
                skipped["offdistribution:" + "+".join(sorted(set(flags)))] += 1
                continue

            buckets[(label, score)].append({
                "left": left,
                "right": right,
                "label": label,
                "why": clean(item.get("contradiction")),
                "_review": {
                    "paper_id": paper_id,
                    "aspect": aspect,
                    "intensity_score": score,
                    "intensity_why": clean(intensity.get("justification"))[:220],
                },
            })

    rows = []
    for key in sorted(buckets, key=lambda k: (k[0], -k[1])):
        chosen = buckets[key][:PER_BUCKET]
        rows.extend(chosen)

    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    total = sum(len(v) for v in buckets.values())
    print(f"Đọc {SRC.name}: {total} mục dùng được, bỏ {dict(skipped) or '0'}")
    if aspect_fixed:
        print(f"Aspect ngoài enum B2 (đã đánh dấu REVIEW_ME): {dict(aspect_fixed)}")
    print(f"\nGhi {len(rows)} ứng viên -> {OUT.relative_to(ROOT.parent)}")
    print("Phân bố theo (nhãn đề xuất, intensity):")
    for key in sorted(buckets, key=lambda k: (k[0], -k[1])):
        label, score = key
        print(f"  {label:<24} score={score}  có {len(buckets[key]):>3} → xuất {min(len(buckets[key]), PER_BUCKET)}")

    print("""
CÒN THIẾU — dataset này chỉ chứa mâu thuẫn, nên KHÔNG có ứng viên nào cho:
  AGREEMENT, PARTIAL_AGREEMENT, COMPLEMENTARY, UNRELATED
Bốn nhãn này phải lấy tay từ interim/pairs.jsonl sau khi chạy pilot B1-B3.

Bước tiếp: đọc và sửa từng dòng, xoá trường `_review`, giữ ~6-7 dòng/nhãn,
rồi upload thành {ROOT_DRIVE}/processed/fewshot.jsonl""")


if __name__ == "__main__":
    main()
