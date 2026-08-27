"""Dựng fewshot.jsonl cuối từ Human_Annotated_Data.json.

    python phase-2/build_fewshot_final.py

Output: phase-2/fewshot.jsonl  (upload thành {ROOT}/processed/fewshot.jsonl)

Chọn cặp và gán nhãn: Claude, đọc tay từng cặp theo RUBRIC trong track_b_pipeline.ipynb.
Nguyên liệu là câu thật trong corpus, và cặp nào cũng đã được một annotator NGƯỜI
xác định là "hai reviewer đang nói về cùng một chuyện" — phần khó nhất. Việc còn
lại là ánh xạ sang 6 nhãn của contract, và đó là phần Claude làm.

=> Đây KHÔNG phải phương án (a) trong PHASE2_PLAN.md §1 (người tự gán 40 cặp).
   Phải khai vào known_limitations của manifest. Xem README_FEWSHOT.md.

Chọn cặp theo hai ràng buộc của pipeline:
  - Text lấy nguyên văn từ evidence quote -> cùng phân phối với claim do B2 sinh.
  - Bỏ câu hỏi thuần / quote quá ngắn / nội dung hậu-rebuttal -> ba loại này B1/B2
    lọc sạch trước khi tới B4, làm few-shot thì dạy model kiểu input không tồn tại.

R-1 (few-shot không nằm trong tập train) được bảo đảm bằng cấu trúc: 75 paper của
Human_Annotated_Data.json KHÔNG trùng paper nào trong 263 paper của IMPACT. Chỉ cần
đừng upload Human_Annotated_Data.json vào {ROOT}/raw/.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fewshot_candidates import (  # noqa: E402
    SRC, VALID_ASPECTS, clean_quote, evidence_pair, offdistribution,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "fewshot.jsonl"

# (đầu vế A, đầu vế B, nhãn, why). Khớp theo tiền tố nên đọc được và không phụ
# thuộc thứ tự duyệt file. `why` viết lại cho khớp nhãn — why gốc của annotator
# luôn diễn đạt theo khung "mâu thuẫn", đưa nguyên vào prompt của một ví dụ
# COMPLEMENTARY là dạy model một nhãn kèm lời giải thích chống lại chính nó.
SELECTION = [
    # ---------------- AGREEMENT ----------------
    ("Highly relevant prior work was overlooked", "This is an innovative paper, proposing a new class",
     "AGREEMENT", "Hai bên cùng ghi nhận bài không có phân tích lý thuyết VÀ cùng kết luận bài vẫn có giá trị."),
    ("The proposed idea is not exceptional original", "Finally, I should say that TAGCN idea is interesting",
     "AGREEMENT", "Cùng cho rằng độ mới ở mức khiêm tốn (một mở rộng của GCN) nhưng vẫn đánh giá tích cực."),
    ("This is a good first step towards scalable", "The paper strikes me as a valuable contribution",
     "AGREEMENT", "Cùng kết luận đóng góp có giá trị, cùng mức dè dặt về độ mới."),
    ("Clarity of the submission is overall good", "Generally , although the paper is ok written",
     "AGREEMENT", "Cùng kết luận clarity ở mức chấp nhận được kèm vài chỗ chưa rõ; cùng chiều, cùng mức."),
    ("the full model archives a slight improvement", "However, I am skeptical that the performance",
     "AGREEMENT", "Cùng cho rằng mức cải thiện là nhỏ; 'slight' và 'không đáng kể về thống kê' cùng chiều."),
    ("The improvement over a DRN is apparent", "for the stock prediction problem, it looks like the RDRN",
     "AGREEMENT", "Cùng kết luận cải thiện không đồng đều; B đưa đúng một ca minh hoạ cho 'not always' của A."),

    # ---------------- PARTIAL_AGREEMENT ----------------
    ("The reported experiments cover reasonable ground", "we are all in agreement that the paper",
     "PARTIAL_AGREEMENT", "Cùng chiều đánh giá phạm vi thực nghiệm, chỉ khác mức: 'reasonable' và 'cần rộng hơn'."),
    ("The evaluation is a good start with comparing", "The experiments are extensively evaluated",
     "PARTIAL_AGREEMENT", "Cùng chiều tích cực về đánh giá thực nghiệm, khác mức 'good start' và 'extensively'."),
    ("In general, the novelty of this paper is ok", "Despite having good experimental results",
     "PARTIAL_AGREEMENT", "Cùng chiều 'phương pháp mang tính kế thừa', khác mức: 'novelty ok' so với 'not novel'."),
    ("the work looks solid to me", "Overall I found this paper very impressive",
     "PARTIAL_AGREEMENT", "Cùng chiều tích cực về chất lượng công trình, khác mức nhiệt tình rõ rệt."),
    ("The experiments on adversarial training are especially", "Improving speed of generation is nice",
     "PARTIAL_AGREEMENT", "Cùng thấy phần thí nghiệm thú vị, khác mức về việc nó có đủ ý nghĩa hay không."),
    ("the paper brings interesting insights", "to the best of my knowledge, it is original",
     "PARTIAL_AGREEMENT", "Cùng chiều ghi nhận tính nguyên bản, khác phạm vi: A thêm nghi ngờ về mức liên quan."),
    ("First off , this is a clearly written", "While providing interesting insights",
     "PARTIAL_AGREEMENT", "Cùng chiều khen bài viết rõ, khác phạm vi: B chỉ riêng phần lý thuyết cần gọn hơn."),

    # ---------------- COMPLEMENTARY ----------------
    ("Weaknesses : 1 . Weak novelty", "I think that this work will have a non-trivial impact",
     "COMPLEMENTARY", "Novelty và impact là hai trục khác nhau; một phương pháp không mới vẫn có thể hữu dụng."),
    ("A normalisation-layer based algorithm is proposed", "TranNorm is well motivated",
     "COMPLEMENTARY", "Thiếu phân tích lý thuyết và có động cơ thiết kế tốt là hai khía cạnh cùng đúng được."),
    ("The intuition behind the method that missing a key", "Its not clear why introducing this PR subnet",
     "COMPLEMENTARY", "A khen phần trực giác của cơ chế, B chê phần lý giải vì sao có lợi; khác mục tiêu."),
    ("Figure 1 I do not like", "The paper is well written in most parts",
     "COMPLEMENTARY", "A chính là một trong 'some exceptions' mà B thừa nhận; không bên nào phủ định bên kia."),
    ("I have missed however a more clear statement", "The motivation behind using Dirichlet",
     "COMPLEMENTARY", "Thiếu phát biểu rõ trong intro và động cơ hợp lý là hai khía cạnh cùng đúng được."),
    ("I 'd like to see more meaty details", "The paper is well-written and easily readable",
     "COMPLEMENTARY", "Độ sâu chi tiết kỹ thuật và độ dễ đọc là hai trục khác nhau — ranh giới #4 của rubric."),
    ("The authors do a good job of motivating multiview", "As such , it is difficult to grasp utility",
     "COMPLEMENTARY", "Động cơ của bài toán và độ dễ nắm bắt của mô hình là hai khía cạnh cùng đúng được."),
    ("The experiments in Section 4 (and appendix) yield", "In Table 2 it is not clear what is compared",
     "COMPLEMENTARY", "Kết quả thuyết phục ở Section 4 và một bảng khó đọc là hai mục tiêu khác nhau."),

    # ---------------- PARTIAL_CONTRADICTION ----------------
    ("The paper claims “parameter-free” as a strength", "The TranNorm layer is simple and free of parameters",
     "PARTIAL_CONTRADICTION", "Cùng một mệnh đề 'parameter-free', nhưng rào đón 'a bit misleading' không phủ định toàn bộ."),
    ("My main concern regarding this paper is whether", "The paper is clear, and makes an interesting",
     "PARTIAL_CONTRADICTION", "A rào đón bằng 'concern... whether'; B chỉ khen chung chung nên không phủ định trọn vẹn."),
    ("However, the ablation study and analysis on the model", "The evaluation is thorough across the board",
     "PARTIAL_CONTRADICTION", "Cùng trục độ kỹ của đánh giá, nhưng B rào đón bằng 'as far as I can tell'."),
    ("Despite this limitation, I'm inclined to say", "I don't think I can fully trust your conclusions",
     "PARTIAL_CONTRADICTION", "Cùng câu hỏi 'hạn chế này có phá kết luận không', hai bên đều rào đón."),
    ("On the whole, I think the paper is well written", "Equation ( 6 ) and C1 are presented as contributions",
     "PARTIAL_CONTRADICTION", "Claim tổng quát 'idea novel' đối với claim rất cụ thể về Eq.(6) — ranh giới #5 của rubric."),
    ("The lack of clarity makes it difficult", "The paper communicates the main messages clearly",
     "PARTIAL_CONTRADICTION", "Cùng trục clarity, đối lập, nhưng B tự nhượng bộ 'writing could be improved significantly'."),
    ("The paper employs a substantial amount of methods", "the complexity of the method",
     "PARTIAL_CONTRADICTION", "Cùng một sự thật (nhiều máy móc kỹ thuật) bị hai bên đánh giá ngược chiều."),
    ("Otherwise the presentation is fair", "I found the paper difficult to read",
     "PARTIAL_CONTRADICTION", "Cùng trục clarity, đối lập, nhưng A rào đón bằng 'fair' nên không phủ định trọn vẹn."),

    # ---------------- CONTRADICTION ----------------
    ("This proposed method is compared to multiple baselines", "the experiments provided are not sufficient",
     "CONTRADICTION", "Cùng trục đầy đủ thực nghiệm, hai kết luận loại trừ nhau, không bên nào rào đón."),
    ("The approach is novel and very interesting", "All in all, the originality of the paper is lacking",
     "CONTRADICTION", "Cùng trục originality, hai lập trường loại trừ nhau, cả hai đều không rào đón."),
    ("So I believe the significance of the paper is high", "It's just trivial once we have both techniques",
     "CONTRADICTION", "Cùng trục significance, 'high' và 'trivial' không thể cùng đúng."),
    ("the proposed bilinear transformation is clearly different", "because the idea and effectiveness of channel",
     "CONTRADICTION", "Cùng trục novelty, B quy thẳng về MA-CNN và khẳng định 'not novel', không rào đón."),
    ("Overall, I think the paper considers an important problem", "The perception-driven control formulation",
     "CONTRADICTION", "Đối lập trực tiếp cả về đóng góp lẫn chất lượng trình bày, không bên nào rào đón."),
    ("The empirical results are also narrow", "The experimental results are comprehensive and diverse",
     "CONTRADICTION", "Cùng trục độ rộng của kết quả thực nghiệm, 'narrow' và 'comprehensive' loại trừ nhau."),
    ("The submitted paper also needs major improvements", "The paper is clear written and easy to follow",
     "CONTRADICTION", "Cùng trục trình bày, 'needs major improvements' và 'clear written' loại trừ nhau."),

    # ---------------- UNRELATED ----------------
    # Không rút được từ các cặp gốc: mỗi cặp trong dataset này theo định nghĩa đã
    # là "cùng một issue". Ba cặp dưới ghép CHÉO — cùng paper, cùng aspect (đúng
    # ràng buộc của B3), nhưng lấy từ hai analysis item khác nhau nên khác issue.
    ("I 'd like to see more meaty details", "for the stock prediction problem, it looks like the RDRN",
     "UNRELATED", "A đòi chi tiết kiến trúc, B nói một kết quả cụ thể trên bài toán chứng khoán — không có issue chung."),
    ("2.2 .An experimental comparison to the full outer product", "1 .The paper presents new insights into element-wise",
     "UNRELATED", "A nói về một so sánh còn thiếu, B nói về đóng góp diễn giải phép nhân — hai issue khác nhau."),
    ("I found this paper interesting, but I have one clarification", "it looks like the proposed model is basically a standard VAE",
     "UNRELATED", "A chỉ là nhận xét chung kèm lời hẹn làm rõ, không đủ căn cứ xác định nó cùng issue với B."),
]


def universe():
    """Mọi cặp dùng được: cặp gốc, cộng cặp ghép chéo cùng paper + cùng aspect."""
    data = json.loads(SRC.read_text(encoding="utf-8"))
    items, by = [], defaultdict(list)

    for paper_id, entry in data.items():
        for it in entry.get("analysis", []):
            pr = evidence_pair(it.get("evidence"))
            if not pr:
                continue
            left, right = clean_quote(pr[0]), clean_quote(pr[1])
            if not left or not right:
                continue
            if offdistribution(left) or offdistribution(right):
                continue
            aspect = (it.get("aspect") or "").strip().lower()
            items.append((left, right, paper_id, aspect, "goc"))
            by[(paper_id, aspect)].append((left, right))

    for (paper_id, aspect), lst in by.items():
        for i in range(len(lst) - 1):
            left, right = lst[i][0], lst[i + 1][1]
            if offdistribution(left) or offdistribution(right):
                continue
            items.append((left, right, paper_id, aspect, "ghep_cheo"))
    return items


def main():
    items = universe()
    out, missing = [], []

    for lp, rp, label, why in SELECTION:
        hit = [x for x in items if x[0].startswith(lp) and x[1].startswith(rp)]
        if not hit:
            missing.append((lp[:45], rp[:45]))
            continue
        left, right, paper_id, aspect, kind = hit[0]
        if aspect not in VALID_ASPECTS:
            aspect = "none"
        out.append({
            "left": left,
            "right": right,
            "label": label,
            "why": why,
            "provenance": {"paper_id": paper_id, "aspect": aspect, "pairing": kind},
        })

    if missing:
        print("[LỖI] Không khớp được:")
        for lp, rp in missing:
            print(f"  A~{lp!r}  B~{rp!r}")
        return 1

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n",
                   encoding="utf-8")

    dist = Counter(r["label"] for r in out)
    papers = {r["provenance"]["paper_id"] for r in out}
    print(f"Ghi {len(out)} ví dụ few-shot -> {OUT.relative_to(ROOT.parent)}")
    print(f"Rút từ {len(papers)} paper, tất cả nằm ngoài 263 paper của IMPACT.\n")
    for lab in ["AGREEMENT", "PARTIAL_AGREEMENT", "COMPLEMENTARY",
                "PARTIAL_CONTRADICTION", "CONTRADICTION", "UNRELATED"]:
        print(f"  {lab:<24} {dist.get(lab, 0):>2}")
    print(f"\n  {'aspect':<24} {dict(Counter(r['provenance']['aspect'] for r in out))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
