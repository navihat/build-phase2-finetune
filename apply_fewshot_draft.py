"""Áp pass gán nhãn đầu (do Claude đọc thủ công) lên fewshot_candidates.jsonl.

    python phase-2/apply_fewshot_draft.py

Output: phase-2/fewshot_draft.jsonl

ĐÂY LÀ BẢN NHÁP, KHÔNG PHẢI NHÃN NGƯỜI. Mỗi dòng giữ `_machine_label` (nhãn suy
từ intensity) và `_label_source: claude-first-pass` để không thể nhầm là đã duyệt.
Người phải đọc lại từng dòng, sửa nhãn/why nếu khác ý, rồi xoá hết trường `_*`.

Ship thẳng file này = few-shot do LLM gán = pipeline không còn neo người nào.
Khi đó PHẢI ghi vào `known_limitations` của manifest, đừng để im.
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "fewshot_candidates.jsonl"
OUT = ROOT / "fewshot_draft.jsonl"

# index -> (nhãn theo contract, why viết lại cho khớp nhãn)
# `why` gốc trong Human_Annotated_Data.json luôn diễn đạt theo khung "mâu thuẫn",
# nên với mọi dòng KHÔNG phải CONTRADICTION nó phải được viết lại, nếu không
# few-shot dạy model một nhãn kèm lời giải thích chống lại chính nhãn đó.
LABELS = {
    0:  ("COMPLEMENTARY",         "Novelty và impact là hai trục khác nhau; một phương pháp không mới vẫn có thể hữu dụng."),
    1:  ("PARTIAL_CONTRADICTION", "Cùng một mệnh đề 'parameter-free', nhưng rào đón 'a bit misleading' không phủ định toàn bộ."),
    2:  ("CONTRADICTION",         "Cùng trục đầy đủ thực nghiệm, hai kết luận loại trừ nhau, không bên nào rào đón."),
    3:  ("PARTIAL_CONTRADICTION", "A rào đón bằng 'concern... whether'; B chỉ khen chung chung nên không phủ định trọn vẹn."),
    4:  ("PARTIAL_CONTRADICTION", "Cùng trục độ thoả đáng của so sánh, nhưng B rào đón bằng 'seems'."),
    5:  ("CONTRADICTION",         "Cùng trục originality, hai lập trường loại trừ nhau, cả hai đều không rào đón."),
    6:  ("PARTIAL_CONTRADICTION", "Giá trị đóng góp và độ sâu insight là hai khía cạnh lệch nhau, chỉ mâu thuẫn một phần."),
    7:  ("CONTRADICTION",         "Cùng trục significance, 'high' và 'trivial' không thể cùng đúng."),
    8:  ("PARTIAL_CONTRADICTION", "Cùng trục novelty nhưng B rào đón bằng 'seems rather incremental'."),
    9:  ("CONTRADICTION",         "Cùng trục novelty, B quy thẳng về MA-CNN và khẳng định 'not novel', không rào đón."),
    10: ("PARTIAL_CONTRADICTION", "Cùng trục độ kỹ của đánh giá, nhưng B rào đón bằng 'as far as I can tell'."),
    11: ("COMPLEMENTARY",         "A đòi thêm kiến trúc mới, B báo kết quả forward transfer; hai khía cạnh cùng đúng được."),
    12: ("PARTIAL_CONTRADICTION", "Cùng câu hỏi 'hạn chế này có phá kết luận không', hai bên đều rào đón."),
    13: ("PARTIAL_AGREEMENT",     "Cùng chiều đánh giá phạm vi thực nghiệm, chỉ khác mức: 'reasonable' và 'cần rộng hơn'."),
    14: ("PARTIAL_CONTRADICTION", "Cùng trục so sánh với AdaBN/AutoDIAL, nhưng A phát biểu dạng đề xuất nên không phủ định trọn."),
    15: ("PARTIAL_AGREEMENT",     "Cùng chiều tích cực về đánh giá thực nghiệm, chỉ khác mức 'good start' và 'extensively'."),
    16: ("COMPLEMENTARY",         "Thiếu phân tích lý thuyết và có động cơ thiết kế tốt là hai khía cạnh cùng đúng được."),
    17: ("CONTRADICTION",         "Đối lập trực tiếp cả về đóng góp lẫn chất lượng trình bày, không bên nào rào đón."),
    18: ("PARTIAL_CONTRADICTION", "B đề xuất giao thức khác, hàm ý thiết lập hiện tại chưa thuyết phục — mâu thuẫn một trục."),
    19: ("COMPLEMENTARY",         "A khen phần trực giác của cơ chế, B chê phần lý giải vì sao có lợi; khác mục tiêu."),
    20: ("PARTIAL_CONTRADICTION", "Claim tổng quát 'idea novel' đối với claim rất cụ thể về Eq.(6) — ranh giới #5 của rubric."),
    21: ("PARTIAL_CONTRADICTION", "Kết quả khả quan và thiết lập thực nghiệm hạn chế là hai trục sát nhau nhưng lệch."),
    22: ("AGREEMENT",             "Hai bên cùng ghi nhận không có phân tích lý thuyết VÀ cùng kết luận bài vẫn có giá trị."),
    23: ("PARTIAL_CONTRADICTION", "A chê một lỗi framing cụ thể, B khen văn phong tổng thể — cùng trục clarity, khác phạm vi."),
    24: ("PARTIAL_CONTRADICTION", "Cùng trục clarity, đối lập, nhưng B tự nhượng bộ 'writing could be improved significantly'."),
    25: ("UNRELATED",             "A không đứng một mình đọc được ('which is the case'), không xác định nổi issue chung."),
    26: ("COMPLEMENTARY",         "A chính là một trong 'some exceptions' mà B thừa nhận; không bên nào phủ định bên kia."),
    27: ("COMPLEMENTARY",         "Thiếu phát biểu rõ trong intro và động cơ hợp lý là hai khía cạnh cùng đúng được."),
    28: ("COMPLEMENTARY",         "Chất lượng phương pháp và mức quan trọng của cả setting là hai tầng khác nhau."),
    29: ("COMPLEMENTARY",         "Cùng cặp tầng như trên: setting ít giá trị thực tiễn vẫn đi cùng phương pháp làm tốt."),
    30: ("COMPLEMENTARY",         "Độ sâu chi tiết kỹ thuật và độ dễ đọc là hai trục khác nhau — ranh giới #4 của rubric."),
    31: ("PARTIAL_CONTRADICTION", "Cùng một sự thật (nhiều máy móc kỹ thuật) bị hai bên đánh giá ngược chiều."),
    32: ("PARTIAL_CONTRADICTION", "Phán quyết tổng thể đối với lời khen một ý cụ thể — ranh giới #5 của rubric."),
    33: ("COMPLEMENTARY",         "A khẳng định có novelty, B nói novelty chưa được giải thích rõ; hai chuyện khác nhau."),
    34: ("PARTIAL_CONTRADICTION", "Cùng trục độ đầy đủ kiểm chứng thực nghiệm, A chỉ ra một thiếu sót cụ thể."),
    35: ("PARTIAL_CONTRADICTION", "Cùng trục clarity, khác phạm vi: riêng phần lý thuyết so với toàn bài."),
}

# Dòng cần người soi kỹ nhất, kèm lý do.
FLAGS = {
    25: "A khong qua duoc B2-guard (khong dung mot minh doc duoc). Giu lam vi du UNRELATED thi "
        "dang day model mot kieu input ma pipeline khong sinh ra -> can nhac bo va lay UNRELATED tu pilot pairs.",
    17: "Quote B co dau '...' cat giua cau trong du lieu goc.",
    22: "Vi du AGREEMENT duy nhat rut ra duoc tu ca dataset nay.",
}


def main():
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    out, changed = [], 0

    for i, r in enumerate(rows):
        label, why = LABELS[i]
        machine = r["label"]
        if label != machine:
            changed += 1
        rec = {
            "left": r["left"],
            "right": r["right"],
            "label": label,
            "why": why,
            "_label_source": "claude-first-pass",
            "_machine_label": machine,
            "_agrees_with_machine": label == machine,
            "_review": r["_review"],
            "_human_why": r["why"],
        }
        if i in FLAGS:
            rec["_flag"] = FLAGS[i]
        out.append(rec)

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n",
                   encoding="utf-8")

    dist = Counter(r["label"] for r in out)
    print(f"Ghi {len(out)} dòng -> {OUT.relative_to(ROOT.parent)}\n")
    print("Phân bố nhãn sau khi đọc tay:")
    for lab in ["AGREEMENT", "PARTIAL_AGREEMENT", "COMPLEMENTARY",
                "PARTIAL_CONTRADICTION", "CONTRADICTION", "UNRELATED"]:
        n = dist.get(lab, 0)
        gap = "" if n >= 6 else f"   <-- thiếu {6-n} để đủ 6/nhãn"
        print(f"  {lab:<24} {n:>2}{gap}")
    print(f"\nLệch so với nhãn máy suy từ intensity: {changed}/{len(out)} dòng "
          f"({changed/len(out)*100:.0f}%)")


if __name__ == "__main__":
    main()
