# Contract nhãn — bản GOLD

> Rút ra từ `phase2_trackb/golden_set/gold_test.jsonl` (129 cặp, `relation-gold-1.0`,
> `HUMAN_VERIFIED`, annotator `NTH`) bằng cách đọc toàn bộ trường `annotation_note`.
> **Đây là contract chuẩn.** Rubric cũ dùng cho `trackB_silver.jsonl` lệch ở đúng một chỗ,
> xem mục 3. Bối cảnh: [`TRAIN.md`](TRAIN.md) muc 10.

---

## 1 · Contract là HAI CHIỀU, không phải một trục

Rubric cũ xếp 6 nhãn lên một trục AGREEMENT↔CONTRADICTION rồi treo UNRELATED ra ngoài.
Đọc 129 note của gold thì thấy annotator thực ra quyết định theo **hai câu hỏi độc lập**:

```
        Câu 1: Hai claim có nhắm tới CÙNG MỘT issue cụ thể không?
                          │
        ┌─────────────────┴─────────────────┐
        │ CÙNG issue                        │ KHÁC issue
        │                                   │
   Câu 2: quan hệ giữa hai            ┌─────┴──────┐
   phán xét về issue đó?              │            │
        │                        cùng hướng    không liên
   ┌────┼────┬─────────┐         đánh giá,     quan nội dung
   │    │    │         │         khác phạm vi
 đồng  bổ  mâu     đối lập           │            │
 thuận sung thuẫn   trực tiếp        │            │
  hoàn  hai  có                      │            │
  toàn  góc  điều kiện               │            │
   │    │     │        │             │            │
AGREE  COMP  PART_    CONTRA    PARTIAL_      UNRELATED
       LEM   CONTRA                AGREEMENT
```

**Câu 1 là câu quyết định.** Rubric cũ hỏi câu này rồi trả lời sai chỗ — xem mục 3.

---

## 2 · Sáu nhãn, kèm cách diễn đạt thật của annotator

| Nhãn | Điều kiện | Mẫu câu trong `annotation_note` | n |
|---|---|---|---:|
| `AGREEMENT` | Cùng issue, cùng kết luận, **không kèm dè dặt** | *"Cả hai cùng nhấn mạnh…"* · *"Cả hai đồng ý…"* | 20 |
| `COMPLEMENTARY` | Cùng issue, **cùng chỉ ra một thiếu sót**, mỗi bên soi một mặt, **bổ sung cho nhau** | *"Cả hai chỉ ra tổng quan thiếu phần 'sau cảnh báo' nhưng từ góc khác nhau… bổ sung cho nhau, không mâu thuẫn"* | 29 |
| `PARTIAL_AGREEMENT` | **Cùng hướng đánh giá**, nhưng mỗi bên nhấn mạnh phạm vi khác; không phủ định nhau | *"Cùng hướng đánh giá tổng quan ổn nhưng thiếu sót; PB-01 tập trung… PB-03 tập trung…"* | 24 |
| `PARTIAL_CONTRADICTION` | Cùng issue, mâu thuẫn về **mức độ / điều kiện** chứ không phải có-không | *"PB-01 khẳng định không điều kiện, còn PB-02 chỉ công nhận nếu…"* | 24 |
| `CONTRADICTION` | Cùng issue, **hai khẳng định không thể cùng đúng** | *"PB-01 khẳng định ba mục tiêu đã cụ thể/đo được; PB-02 khẳng định mục tiêu đang trộn lẫn và chưa đủ rõ — đối lập trực tiếp"* | 14 |
| `UNRELATED` | **Khác issue.** Hai bên bàn hai chuyện độc lập, dù cùng tiêu chí | *"PB-01 nói về lợi ích tiết kiệm thời gian; PB-03 nói về mã nguồn mở — không cùng một issue cụ thể dù cùng tiêu chí"* | 18 |

### Ranh giới khó nhất: COMPLEMENTARY vs UNRELATED

Cả hai đều là "mỗi người nói một thứ khác nhau". Phép thử tách chúng:

> **Có tồn tại MỘT vấn đề chung mà cả hai claim đều đang nhắm tới không?**
> - Có → `COMPLEMENTARY` (hai mặt của cùng một thiếu sót; ghép lại thành bức tranh đầy đủ hơn)
> - Không, chỉ là hai chủ đề rời → `UNRELATED`

```
COMPLEMENTARY: "Cả hai phê bình cách dùng SHAP hiện tại nhưng theo hướng khác nhau:
                PB-02 về độ an toàn thống kê, PB-03 về khả năng diễn giải."
                → vấn đề chung tồn tại: cách dùng SHAP.

UNRELATED:     "PB-01 bàn về thiết kế thử nghiệm A/B; PB-03 bàn về phân bổ ngân sách camera
                — hai issue khác nhau bị gộp nhầm vào cùng tiêu chí phương pháp."
                → không có vấn đề chung, chỉ chung cái nhãn tiêu chí.
```

> ⚠ **UNRELATED KHÔNG có nghĩa "khác paper/khác đề tài".** Toàn bộ 18 cặp UNRELATED của gold
> đều **cùng cohort, cùng criterion**, chỉ khác reviewer. Đây là điểm mà 110 cặp
> `synthetic_cross_paper` trong silver hiểu sai hoàn toàn — xem mục 3.

### Ranh giới phụ: COMPLEMENTARY vs PARTIAL_AGREEMENT

Đều "cùng hướng, khác góc". Khác nhau ở chỗ neo vào đâu:

- `COMPLEMENTARY` neo vào **một thiếu sót cụ thể** — hai bên mô tả hai mặt của chính nó.
- `PARTIAL_AGREEMENT` neo vào **một phán xét tổng thể** — hai bên đồng ý về kết luận chung
  rồi mỗi người nhấn mạnh phạm vi riêng.

---

## 3 · Chỗ rubric cũ lệch — chỉ MỘT lớp

Đếm trường `why` của toàn bộ `trackB_silver.jsonl` (bỏ 39 cặp few-shot):

| Lớp silver | `why` mở đầu bằng | Khớp gold? |
|---|---|---|
| `AGREEMENT` | "cùng điểm cụ thể… cùng chiều cùng mức độ" — 70/70 | ✅ |
| `PARTIAL_AGREEMENT` | "cùng chiều…, khác phạm vi" — 210/210, 0 ca "khác điểm" | ✅ |
| `PARTIAL_CONTRADICTION` | "cùng điểm…, A khen B chê có điều kiện" — 160/160 | ✅ |
| `CONTRADICTION` | "cùng điểm cụ thể… không thể cùng đúng" — 22/22 | ✅ |
| **`COMPLEMENTARY`** | **"khác điểm cụ thể: X vs Y" — 494/496** | 🔴 **phần lớn phải là `UNRELATED`** |

Rubric cũ định nghĩa `COMPLEMENTARY` = *"cùng aspect nhưng khác điểm cụ thể"*. Gold gọi đúng
tình huống đó là `UNRELATED`, và dành `COMPLEMENTARY` cho trường hợp hẹp hơn nhiều: cùng một
thiếu sót, hai góc bổ sung nhau.

```
silver: "khác điểm cụ thể: tính mới tổng thể vs một kết quả receptive field cụ thể"
        -> gán COMPLEMENTARY
gold:   cùng dạng tình huống đó
        -> gán UNRELATED
```

`COMPLEMENTARY` chiếm **45% dữ liệu train**. Model học từ silver sẽ đều đặn xuất
`COMPLEMENTARY` ở đúng chỗ gold chờ `UNRELATED`.

### Còn 110 cặp `synthetic_cross_paper`

Sinh bằng luật "ghép claim của hai paper khác nhau → UNRELATED". Sai theo hai cách:

1. **Không học được.** 91.8% số cặp đó không chung một từ nội dung nào — nhưng
   `COMPLEMENTARY` cùng paper cũng 83.9% như vậy. Khác biệt duy nhất là metadata `paper_id`,
   thứ model không nhận. Đo được: F1 0.130 (`TRAIN.md` muc 9.3).
2. **Sai định nghĩa.** Gold không hề coi "khác paper" là UNRELATED — mọi cặp UNRELATED của
   gold đều cùng criterion.

→ **Bỏ hẳn 110 cặp này.** UNRELATED thật sẽ đến từ việc gán lại 496 cặp COMPLEMENTARY.

---

## 4 · Việc phải làm

| | |
|---|---|
| ✅ | Gán lại **496 cặp `COMPLEMENTARY`** theo phép thử ở mục 2 — xong 2026-08-29, xem mục 6 |
| ✅ | Bỏ **110 cặp `synthetic_cross_paper`** |
| ✅ | Chạy lại `src/train/split.py` sau khi silver đổi |
| ✅ | Đổi backbone sang `xlm-roberta-base` — gold 129/129 tiếng Việt, silver tiếng Anh |
| ✅ | Thêm cell đánh giá trên gold vào notebook (muc 10), tách riêng khỏi CV trên silver |
| ⬜ | Rà phụ ranh giới `COMPLEMENTARY` ↔ `PARTIAL_AGREEMENT` trên 217 cặp PARTIAL_AGREEMENT |
| ⬜ | NTH xác nhận contract ở mục 2 — biến nó từ "Claude suy ra từ gold" thành "người xác nhận" |

Bốn lớp còn lại **không cần đụng tới** — đó là phần đỡ tốn công nhất của phát hiện này.

---

## 6 · Kết quả gán lại *(2026-08-29)*

496 cặp đọc từng cặp theo phép thử ở mục 2, mặc định `UNRELATED`, chỉ giữ `COMPLEMENTARY`
khi có **bằng chứng dương** về một đối tượng/thiếu sót chung. Mỗi quyết định kèm `why_moi`
nêu rõ đối tượng chung là gì — lưu ở `phase2_trackb/interim/relabel/decisions.jsonl`.

```
COMPLEMENTARY -> UNRELATED : 343  (69.2%)
giữ COMPLEMENTARY          : 153  (30.8%)
bỏ synthetic_cross_paper   : 110
1099 -> 989 cặp
```

### Phân bố nhãn mới, đối chiếu với gold

| Nhãn | silver n | silver % | **gold %** | chênh |
|---|---:|---:|---:|---:|
| UNRELATED | 346 | 35.0% | 14.0% | **+21.0** |
| PARTIAL_AGREEMENT | 217 | 21.9% | 18.6% | +3.3 |
| PARTIAL_CONTRADICTION | 168 | 17.0% | 18.6% | −1.6 |
| COMPLEMENTARY | 153 | 15.5% | 22.5% | −7.0 |
| AGREEMENT | 76 | 7.7% | 15.5% | −7.8 |
| CONTRADICTION | 29 | 2.9% | 10.9% | −8.0 |

Mất cân bằng cao/thấp giảm từ **17:1 xuống 11.9:1**.

> ⚠ **Vẫn còn lệch tiên nghiệm (prior shift).** Silver có 35% UNRELATED còn gold chỉ 14% —
> model sẽ thiên về đoán UNRELATED trên gold. Trọng số lớp bù một phần (`UNRELATED w=0.48`,
> `CONTRADICTION w=5.25`), nhưng nếu ma trận nhầm lẫn ở muc 10 của notebook cho thấy cột
> UNRELATED phình ra thì đây là nguyên nhân, **không phải** do gán lại sai.
>
> Nguyên nhân của lệch: B3 ghép cặp theo "cùng paper + cùng aspect", mà aspect của ICLR
> (`soundness` gộp 5014 claim) thô hơn nhiều so với `criterion_id` của gold (17 tiêu chí).
> Aspect thô ⇒ nhiều cặp cùng nhãn nhưng khác issue ⇒ nhiều UNRELATED. Đây là đặc tính của
> cách sinh cặp, không sửa được bằng gán nhãn.

## 5 · Hai cặp gold đáng rà lại

`expected_relation = UNRELATED` nhưng note mô tả quan hệ khác:

- `gold-*` (cohort `case-01-dropout-c4`, `C2.1`): *"Cả hai đồng ý mục tiêu kỹ thuật khá rõ ràng;
  PB-03 chỉ bổ sung yêu cầu thêm chỉ số vận hành mà không phủ nhận đánh giá của PB-01."*
  → đọc như `PARTIAL_AGREEMENT`.
- `gold-*` (cùng cohort, `C2.1`): *"Cùng cho rằng mục tiêu hiện tại thiếu chỉ số quan trọng,
  nhưng PB-02 cho là vấn đề định nghĩa outcome, còn PB-03 chấp nhận mục tiêu hiện tại là rõ."*
  → đọc như `PARTIAL_CONTRADICTION`.

16/18 cặp còn lại rất nhất quán với phép thử ở mục 2. Hai cặp này chiếm 1.6% gold — không đủ
để hoãn việc gì, nhưng nên xác nhận trước khi chốt con số cuối.
