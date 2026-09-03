# Giai đoạn Train — Relation Classifier

> Fine-tune bộ phân loại quan hệ 6 lớp giữa hai atomic claim của hai reviewer.
> Đầu vào: [`pipeline_data/processed/trackB_silver.jsonl`](../pipeline_data/processed/trackB_silver.jsonl) (1099 cặp).
> Kế hoạch tổng: [`PLAN.md`](PLAN.md) · Nguồn few-shot: [`FEWSHOT.md`](FEWSHOT.md)

> ✅ **Sẵn sàng train.** *(cập nhật 2026-08-29)*
>
> Silver đã gán lại theo contract của tập gold: 496 cặp `COMPLEMENTARY` đọc lại từng cặp,
> **343 chuyển thành `UNRELATED`**, 110 cặp `synthetic_cross_paper` bỏ hẳn. `1099 → 989` cặp.
> Split đã chạy lại. → [`RUBRIC_GOLD.md`](RUBRIC_GOLD.md) ·
> [mục 10](#10--tập-gold-và-việc-gán-lại-nhãn-2026-08-29)
>
> - 🔴 **Mọi số ở mục 9.1 (macro-F1 CV = 0.319) đã VÔ HIỆU** — chúng đến từ split cũ, còn
>   synthetic UNRELATED, và quan trọng nhất là **contract nhãn cũ**. Đừng so với số mới.
>   → [mục 9](#9--lần-train-đầu-tiên--kết-quả-và-ba-thứ-đã-sửa-2026-08-29) giữ lại làm hồ sơ.
> - ⚠ **Con số để báo cáo giờ là macro-F1 trên GOLD** (muc 10 của notebook), không phải CV
>   trên silver. CV chỉ còn là chỉ số phát triển.
> - ✅ **Split đã stratified theo nhãn** và ghim few-shot vào train. 5-fold giờ đều
>   (CONTRADICTION 4–5 mỗi fold, trước là 1–13). → [mục 2](#2--chia-dữ-liệu) ·
>   [`reports/split_report.md`](../pipeline_data/reports/split_report.md)
> - 🔴 **CONTRADICTION vẫn chỉ 29 mẫu, test có 2.** Đọc kết quả từ **5-fold CV**, đừng đọc F1
>   lớp này trên test. → [mục 8](#8--tăng-mẫu-contradiction-phân-tích-2026-08-28-chưa-thực-thi)
>   — nhưng làm sau mục 9, vì lợi ích nhỏ hơn mà tốn nhãn tay.

---

## 0 · Vì sao giai đoạn này tồn tại ở dạng hiện tại

Pipeline gốc (`notebooks/01_data_pipeline.ipynb`, các bước B4→B6) dùng **ensemble 3 model +
debate + judge** để sinh nhãn. Nó đã chạy và **thất bại**, có số liệu:

| Chỉ số | Kết quả pilot 100 cặp | Ngưỡng trong [`CHECKLIST.md`](CHECKLIST.md) |
|---|---|---|
| Đồng thuận 3/3 (unanimous) | **0%** | — |
| Tỉ lệ đi debate | **53%** | cảnh báo khi > 35% |
| `dropped_far` (bị loại) | **29–33%** | cảnh báo khi > 25% |
| `AGREEMENT` trong silver | **0 mẫu** | lớp rỗng = không học được |

Chẩn đoán, theo `debate_reason` trong `interim/routed.jsonl`:

```
order_flip:      50/53  (94%)   <- có model đổi nhãn khi đảo A/B
adjacent_split:   3/53  ( 6%)   <- ba model bất đồng thật
```

**94% số cặp phải đi debate không phải vì model bất đồng, mà vì một model tự mâu thuẫn
với chính nó khi đảo thứ tự hai claim.** Flip-rate đo được: Qwen 50%, Gemma 44%, SeaLLM 18%.

Đây là lý do toàn bộ thiết kế train dưới đây xoay quanh **tính đối xứng**.

Ngoài ra, đối chiếu nhãn ensemble với nhãn đọc tay trên 56 cặp nó xử lý xong: **chỉ trùng
48.2%**, trong đó riêng lỗi `UNRELATED → COMPLEMENTARY` chiếm 14/29 ca lệch. Con số này là
căn cứ định lượng cho việc bỏ ensemble, không phải cảm tính.

---

## 1 · Dữ liệu

`trackB_silver.jsonl` — 1099 cặp, 1099 `pair_id` duy nhất, gộp từ **4 nguồn**:

| `source` | n | Cách tạo | Độ tin cậy |
|---|---:|---|---|
| `full_batch{00..03}_manual` | 800 | Claude đọc từng cặp theo RUBRIC | Nhãn LLM, đọc tay |
| `mined_stance_opposition` | 150 | Đào cặp POSITIVE-vs-NEGATIVE rồi đọc tay | Nhãn LLM, đọc tay |
| `synthetic_cross_paper` | 110 | **Luật**: ghép claim của hai paper khác nhau → UNRELATED | Luật, không đọc tay |
| `fewshot_human_pairs` | 39 | Cặp do annotator người xác nhận, nhãn 6 lớp do LLM gán | Nền người, nhãn LLM |

### Phân bố nhãn

| Nhãn | n | % |
|---|---:|---:|
| COMPLEMENTARY | 496 | 45.1% |
| PARTIAL_AGREEMENT | 217 | 19.7% |
| PARTIAL_CONTRADICTION | 168 | 15.3% |
| UNRELATED | 113 | 10.3% |
| AGREEMENT | 76 | 6.9% |
| CONTRADICTION | 29 | 2.6% |

Mất cân bằng cao nhất/thấp nhất **17:1**. Trước khi bổ sung là 34:1 với UNRELATED rỗng hoàn toàn.

### Vì sao COMPLEMENTARY chiếm gần một nửa

Không phải lỗi gán nhãn. B3 ghép cặp theo **cùng paper + cùng aspect**, mà "cùng aspect
nhưng khác điểm cụ thể" chính là định nghĩa của COMPLEMENTARY (ranh giới #2 và #5 của RUBRIC).
Đây là phân bố thật của cách sinh dữ liệu.

### Vì sao CONTRADICTION chỉ 2.6%

Cũng là đặc tính của miền, đã ghi nhận trong [`FEWSHOT.md`](FEWSHOT.md): trong 36 cặp mà
annotator người dán nhãn "contradiction", đọc theo contract chỉ **5 là CONTRADICTION**,
còn **10 là COMPLEMENTARY**. Cái mà người quen gọi là "hai reviewer mâu thuẫn" phần lớn
là hai người soi hai khía cạnh khác nhau của cùng một vấn đề.

### ⚠ Giới hạn phải khai vào manifest

- Nhãn 6 lớp **do LLM gán**, không có người xác nhận lại. Không có annotator thứ hai
  → **không có trần (ceiling)**. Nếu macro-F1 = 0.62 thì không biết 0.62 là gần trần hay còn xa.
- 110 cặp `synthetic_cross_paper` là **luật sinh**, không phải đọc tay. Phải đo tách riêng
  (mục 4.6) chứ không được để nó làm đẹp macro-F1 tổng.
- Mọi đánh giá dưới đây đo **độ khớp với nhãn Claude**, không phải với sự thật.

---

## 2 · Chia dữ liệu

**Chia theo NHÓM PAPER, không chia ngẫu nhiên.** 1099 cặp chỉ đến từ 222 nhóm paper
(~5 cặp/paper), và nhiều cặp dùng chung claim. Chia ngẫu nhiên sẽ để claim của cùng một
paper rơi vào cả train lẫn test → metric ảo.

```bash
python src/train/split.py
```

Ghi ra `pipeline_data/processed/splits/`:

```
trackB_train.jsonl   876 cặp / 189 paper
trackB_val.jsonl     112 cặp /  15 paper
trackB_test.jsonl    111 cặp /  18 paper
folds.json           gán fold 0-4 cho từng pair_id (-1 = luôn ở train), dùng cho CV
class_weights.json   trọng số inverse-frequency tính trên train
```

Và `reports/split_report.md` — bảng đầy đủ để duyệt trước khi train.

Script tự `assert` không paper nào nằm ở hai phần, và **báo cáo rò rỉ ở mức text của claim**
(chặt hơn mức paper). Hiện tại: `train∩val = 1`, `train∩test = 0`, `val∩test = 0` claim —
phần dư đến từ cặp UNRELATED chéo paper, vì `group_of()` chỉ lấy paper bên trái.

<details>
<summary><b>Vì sao không dùng union-find gộp cả hai paper của cặp chéo</b></summary>

110 cạnh chéo trên 263 paper sẽ nối phần lớn paper thành **một thành phần liên thông
khổng lồ**, khiến việc nhóm mất hoàn toàn tác dụng (mọi thứ rơi vào cùng một fold).
Lấy paper trái là đánh đổi có ý thức; phần rủi ro còn lại được đo trực tiếp và báo cáo
thay vì giấu đi.
</details>

### ✅ Stratified theo nhãn — đã sửa 2026-08-29

Bản trước của `assign_groups()` chỉ cân bằng **tổng số cặp**, không đọc `relation` (docstring
viết *"ưu tiên giữ cân bằng nhãn"* nhưng code không làm). Nhãn tương quan mạnh trong cùng
paper nên nhãn hiếm dồn cục.

Bản mới chạy hai bước, xem docstring `assign_groups()`:

1. **Greedy** — duyệt nhóm paper từ lớn đến nhỏ, bỏ vào phần làm tổng độ lệch tăng ít nhất.
2. **Tinh chỉnh** — lặp thử chuyển từng nhóm sang phần khác, chỉ nhận khi tổng độ lệch giảm.

Điểm mấu chốt ở `_dev()`: mỗi nhãn được chuẩn hoá theo **hạn ngạch của chính nhãn đó**, nên
lệch 2 mẫu ở lớp 29-mẫu bị phạt nặng hơn hẳn lệch 2 mẫu ở lớp 496-mẫu.

**Kết quả 5-fold, CONTRADICTION mỗi fold:**

| | f0 | f1 | f2 | f3 | f4 | biên độ |
|---|---:|---:|---:|---:|---:|---:|
| trước | 13 | 4 | 1 | 6 | 5 | **13×** |
| sau | 4 | 5 | 4 | 5 | 4 | **1.25×** |

Lệch lớn nhất của train/val/test so với phân bố toàn cục: **2.2 điểm %** (trước: 8.8 —
PARTIAL_AGREEMENT ở test là 10.9% trong khi toàn cục 19.7%).
Số liệu đầy đủ: [`pipeline_data/reports/split_report.md`](../pipeline_data/reports/split_report.md),
sinh tự động mỗi lần chạy `split.py`.

### ✅ Ghim few-shot vào train — sửa cùng đợt

39 cặp `fewshot_human_pairs` **chính là** 39 ví dụ trong `processed/fewshot.jsonl` đã dùng làm
few-shot khi Claude gán nhãn 1060 cặp còn lại (đối chiếu: trùng **39/39** cả text lẫn nhãn).
Nhãn của phần còn lại được sinh ra *có điều kiện* trên chúng → để chúng vào val/test là chấm
điểm model trên chính các ví dụ đã định nghĩa ra nhãn của tập kiểm tra.

Trước bản này chúng rơi vào train **do may** (chung một `paper_id` giả `HUMAN_ANNOTATED` nên
thành nhóm lớn nhất), không do ràng buộc nào — đổi seed hoặc `--test-size` là lọt.
Nay ghim cứng qua `PIN_TRAIN_SOURCES`; trong `folds.json` chúng nhận `fold = -1`, mà `train.py`
lọc bằng `fold_of[...] != k` nên tự động vào train của mọi fold. **Không phải sửa `train.py`.**

Tắt để đo thử: `python src/train/split.py --no-pin-fewshot`.

### ⚠ Test chỉ có 2 mẫu CONTRADICTION — giới hạn chưa gỡ được

F1 của lớp đó trên test chỉ có thể là 0, 0.5 hoặc 1.0 — **con số ngẫu nhiên, không đọc được**.
Vì vậy **luôn đọc kết quả từ 5-fold CV**, test chỉ dùng cho lần chốt checkpoint cuối.

Ghim few-shot lấy mất 7/29 mẫu CONTRADICTION khỏi phần chấm điểm (test 3 → 2), nhưng đổi lại
CV chặt hơn hẳn (4–5 thay vì 4–7) — mà CV mới là chỗ đọc kết quả, nên đánh đổi này có lời.

Stratified đã kịch trần. Đây là giới hạn vật lý của 29 mẫu / 1099 cặp — muốn test đọc được thì
phải **tăng số mẫu CONTRADICTION**, xem mục 8.

---

## 3 · Ba lựa chọn thiết kế (không phải mặc định)

### 3.1 Đối xứng theo cấu trúc

Quan hệ giữa hai claim **không phụ thuộc claim nào đứng trước**. Đây chính là thứ đã giết
pipeline ensemble. Xử lý bằng ba cơ chế độc lập:

| Cơ chế | Cờ | Làm gì |
|---|---|---|
| Augment 2 chiều | `--symmetric-aug` | Mỗi cặp vào train 2 lần: `(A,B)` và `(B,A)`, cùng nhãn |
| TTA 2 chiều | `--symmetric-tta` | Lúc suy luận: cộng logit của cả hai thứ tự rồi mới argmax |
| **Đo flip-rate** | (mục 4.5) | Đo trên logit **thô, tắt TTA** để kiểm chứng |

> **Điểm then chốt:** phải đo flip-rate trên logit thô. Nếu chỉ đo khi bật TTA thì flip-rate = 0
> **theo định nghĩa** (cộng logit hai chiều là phép đối xứng), và bạn sẽ không phát hiện được
> model có thật sự học được tính đối xứng hay chỉ đang được TTA che cho.

### 3.2 Trọng số lớp

`CrossEntropyLoss(weight=inverse_frequency)`, tính lại trên tập train của **từng fold**.
CONTRADICTION được `w = 6.37`. Không có trọng số thì model bỏ hẳn lớp này mà accuracy vẫn đẹp.

### 3.3 Nhóm theo paper

Đọc fold từ `folds.json`, không tự chia lại ngẫu nhiên trong lúc train.

---

## 4 · Bảy phép đánh giá

Mỗi phép trả lời một câu hỏi khác nhau. **Đừng chỉ đọc accuracy** — đoán COMPLEMENTARY
cho tất cả đã được 45%.

| # | Phép đo | Trả lời câu hỏi |
|---|---|---|
| 4.1 | Per-class P/R/F1, **macro-F1** | Lớp nào bị bỏ rơi? |
| 4.2 | Ma trận nhầm lẫn + top ô nhầm | Sai theo **mẫu** nào? |
| 4.3 | Độ chính xác dung sai theo trục | Sai **nặng** hay **nhẹ**? |
| 4.4 | **QWK** (bình phương khoảng cách) | Hơn đoán mò bao nhiêu, có tính thứ tự? |
| 4.5 | **Flip-rate** (logit thô) | Model có **đối xứng** thật không? |
| 4.6 | Tách theo **nguồn dữ liệu** | Con số nào là **thật**? |
| 4.7 | Hiệu chuẩn + ECE | Đặt được ngưỡng ABSTAIN không? |

### 4.3 · Trục quan hệ

```
AGREEMENT — PARTIAL_AGREEMENT — COMPLEMENTARY — PARTIAL_CONTRADICTION — CONTRADICTION
                                      ⊥
                                  UNRELATED
```

Nhầm `PARTIAL_CONTRADICTION` ↔ `CONTRADICTION` (kề nhau, 1 bước) nhẹ hơn hẳn nhầm
`AGREEMENT` ↔ `CONTRADICTION` (4 bước). Accuracy coi hai lỗi này **như nhau** — đó là chỗ
nó che mất sự thật. Chỉ số dễ đọc nhất cho ứng dụng: **tỉ lệ đúng hoặc lệch ≤1 bước**.

### 4.5 · Đối chiếu flip-rate

| | Flip-rate |
|---|---|
| Qwen3-14B (ensemble cũ) | 50% |
| Gemma-2-9B (ensemble cũ) | 44% |
| SeaLLM-v3-7B (ensemble cũ) | 18% |
| **Model fine-tune (mục tiêu)** | **< 10%** |

Dưới ~10% nghĩa là đã khắc phục được vấn đề đã làm hỏng Track B.

### 4.6 · Vì sao phải tách theo nguồn

Nếu model đạt 99% trên `synthetic_cross_paper` (UNRELATED sinh bằng luật, rất dễ đoán)
nhưng kém ở nơi khác thì **con số UNRELATED trong báo cáo tổng là ảo**. Bảng này bắt đúng
loại tự lừa đó — kể cả tự lừa chính người viết ra tập dữ liệu.

---

## 5 · Baseline và so sánh ngoài

Không có mốc thì macro-F1 = 0.45 là tốt hay tệ đều không biết.

| Mức | Baseline | Trả lời | Chi phí |
|---|---|---|---|
| **Có sẵn** | `majority` (luôn đoán COMPLEMENTARY) | Model có học được gì không? | 0 |
| **Có sẵn** | `stance-rule` (stance đối nghịch → PARTIAL_CONTRADICTION) | Có hơn một luật 3 dòng không? | 0 |
| **Nên thêm** | NLI zero-shot (`mDeBERTa-v3-base-xnli`, đã dùng ở B2-guard) | **Fine-tune trên 1099 nhãn có hơn model NLI có sẵn không?** | vài phút |
| **Nên thêm** | Ablation: tắt `symmetric-aug` / `symmetric-tta` / trọng số lớp | Ba lựa chọn thiết kế có thật sự cần? | 3 lần train |
| Nên có | So backbone: `roberta-base` vs `deberta-v3-base` vs `xlm-roberta-base` | Chọn đúng backbone chưa? Mất gì khi chuyển đa ngữ? | 3× CV |
| Miễn phí | Ensemble cũ vs nhãn đọc tay: **48.2% trùng** | Căn cứ định lượng cho việc bỏ ensemble | đã có |

> **Phép thử gắt nhất là NLI zero-shot.** Nếu nó ngang ngửa model fine-tune thì toàn bộ công
> gán nhãn là vô ích — cần biết điều đó **trước** khi báo cáo, không phải sau.

### Thiếu sót không thể tự bù

Không có annotator thứ hai → không có trần. Cách rẻ nhất để có: bạn (hoặc Hiếu) gán độc lập
**50 cặp** rút ngẫu nhiên rồi tính κ với nhãn Claude. ~30 phút người, và nó biến mọi con số
từ *"khớp với Claude"* thành *"khớp với người, trong khoảng tin cậy X"*.

---

## 6 · Chạy

### 6.1 Colab (khuyên dùng để train)

```
https://colab.research.google.com/github/navihat/peer-review-claim-relations/blob/main/notebooks/02_train_relation_classifier.ipynb
```

Notebook `git clone` repo → cần **push `trackB_silver.jsonl` lên GitHub trước**.
Thứ tự cell: `Setup → Data → Baselines → Train → Đánh giá → 5-fold CV → Learning curve → Lưu Drive`.

### 6.2 Máy cá nhân

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers scikit-learn

python src/train/split.py          # chia dữ liệu (đã chạy, ra kết quả ở mục 2)
python src/train/train.py --cv     # benchmark — con số để báo cáo
python src/train/train.py          # lấy checkpoint cuối
```

### 6.3 Chọn model theo VRAM

| Máy | Model | Cấu hình |
|---|---|---|
| **GTX 1650 4GB** (train, chỉ EN) | `roberta-base` / `deberta-v3-small` | fp16, batch 8-16, max_len 128 |
| GTX 1650 4GB (train, EN+VI) | `distilbert-base-multilingual-cased` | fp16, batch 4 + grad-accum |
| Colab T4/A100 | `deberta-v3-base` / `xlm-roberta-base` | batch 16-32 |

GTX 16-series **không có tensor core** → fp16 tiết kiệm bộ nhớ nhưng không tăng tốc nhiều.
Dataset nhỏ nên cả 5-fold CV cũng chỉ vài phút.

### 6.4 Inference tại máy

Train ở Colab, inference ở máy — **hoàn toàn khả thi**:

| | VRAM cần |
|---|---|
| Training (grad + optimizer state + activation) | ~4-6× kích thước model |
| **Inference** (weights + activation nhỏ) | **~1.2× kích thước model** |

`xlm-roberta-base` fp16 ≈ 540MB → batch 32 vẫn dư trên 4GB. CPU-only cũng chạy được.

> ⚠ Khi inference **nhớ cộng logit cả hai thứ tự** `(A,B)` và `(B,A)` đúng như lúc đánh giá.
> Bỏ bước này là mất luôn tính đối xứng đã dày công xây.

---

## 7 · Trạng thái và việc còn lại

| | |
|---|---|
| ✅ | 1099 cặp; sau khi bỏ UNRELATED còn 986 cặp / 5 lớp (mục 9.3) |
| ✅ | Chia theo paper + 5-fold, **stratified theo nhãn**, có kiểm tra rò rỉ + báo cáo duyệt |
| ✅ | Few-shot ghim vào train (`fold = -1`), không lọt vào phần chấm điểm |
| ✅ | Notebook Colab đã chạy end-to-end, 7 phép đánh giá — kết quả ở mục 9.1 |
| 🔴 | **macro-F1 CV = 0.319, chưa dùng được cho hệ thống.** Chẩn đoán ở mục 9 |
| 🔴 | **Cần chạy lại notebook** — số hiện có là từ split cũ + còn lớp UNRELATED (mục 9.2, 9.3) |
| 🔴 | **CONTRADICTION 29 mẫu là nút thắt** — test chỉ 2 mẫu, F1 lớp đó không đọc được (mục 8) |
| ⚠ | **`src/train/train.py` chưa được sửa theo mục 9** — nó vẫn gộp val vào train, train cứng 6 epoch, chưa có `drop_unrelated`, và **không lưu checkpoint** (không có `torch.save` nào trong file). Notebook là đường chạy thật; script này đang lệch |
| ⬜ | Thêm baseline NLI zero-shot + ablation (mục 5) |
| ⬜ | 50 cặp gán độc lập để có trần κ |
| ⬜ | Dịch VI + test giữ nhãn (mục 8 của [`CHECKLIST.md`](CHECKLIST.md)) |
| ⬜ | Manifest cho Hiếu — **đang chặn người khác**, xem mục 9 của CHECKLIST |

Cell "Learning curve" trong notebook trả lời bằng số cho câu *"1099 đã đủ chưa"*: còn dốc
thì gán thêm còn lời, phẳng rồi thì nên đổ công vào chỗ khác.

---

## 8 · Tăng mẫu CONTRADICTION *(phân tích 2026-08-28, chưa thực thi)*

29 mẫu / 2.6% là thứ chặn cả test lẫn 5-fold. Dưới đây là số đo thật, không phải ước đoán.

### 8.1 · Mining hiện tại đã gần chạm trần

| `source` | n cặp | CONTRA | tỉ lệ |
|---|---:|---:|---:|
| `full_batch00-03_manual` (lấy ngẫu nhiên) | 800 | 12 | **1.5%** ← nền |
| `mined_stance_opposition` | 150 | 10 | **6.7%** ← lift 4.5× |
| `fewshot_human_pairs` (chọn tay) | 39 | 7 | 17.9% |
| `synthetic_cross_paper` | 110 | 0 | — |

Bộ lọc stance đã hiệu quả, nhưng kết quả chủ yếu rơi vào **PARTIAL_CONTRADICTION (54/150)**
và **COMPLEMENTARY (77/150)**, không phải CONTRADICTION.

> Đính chính con số cũ ghi ở mục này: tỉ lệ trúng "nhóm mâu thuẫn" (CONTRA + PARTIAL_CONTRA)
> của `mined_stance_opposition` là **42.7%** (64/150), không phải 48%. Tỉ lệ nền 15% thì đúng
> (118/800 = 14.75%).

### 8.2 · ⚠ Đừng lọc bằng độ trùng từ vựng — đã đo, phản tác dụng

TF-IDF cosine trên toàn bộ cặp đã gán nhãn (bỏ `synthetic`):

| Nhãn | n | median | p75 | p90 |
|---|---:|---:|---:|---:|
| **CONTRADICTION** | 29 | **0.000** | **0.000** | 0.141 |
| PARTIAL_CONTRADICTION | 168 | 0.000 | 0.020 | 0.146 |
| AGREEMENT | 76 | 0.078 | 0.141 | 0.266 |

Và trên 321 cặp đối-lập-stance đã có nhãn, chia theo bin similarity:

| bin sim | n | COMPLE | PARTIAL_CONTRA | **CONTRA** |
|---|---:|---:|---:|---:|
| ≥ 0.30 | 4 | 0% | **100%** | 0% |
| 0.20–0.30 | 9 | 0% | 67% | 11% |
| 0.12–0.20 | 17 | 24% | 41% | 12% |
| < 0.12 | 291 | 48% | 43% | 7% |

**Sim cao cho PARTIAL_CONTRADICTION, không cho CONTRADICTION.** Lý do nhìn rõ trong data:
contradiction thật thường là hai phán xét tổng thể diễn đạt hoàn toàn khác nhau về từ ngữ —
*"The motivation is straightforward and inspiring"* vs *"The paper doesn't do a good job of
convincing me why I should care"* (sim ≈ 0). Đây là quan hệ **ngữ nghĩa**, không phải từ vựng.

Heuristic bề mặt tốt nhất tôi test được (cả hai claim ≤ 25 từ + chứa từ đánh giá tổng thể +
không chứa số) nâng hit rate **6.9% → 15%**, nhưng PARTIAL_CONTRADICTION vẫn chiếm 70% kết quả
lọc. Đó là trần của heuristic bề mặt.

### 8.3 · Kho ứng viên còn lại

Đếm lại trên pool 10.263 claim / 263 paper, điều kiện: cùng paper + cùng aspect + khác
reviewer + chưa dùng trong silver.

| Bộ lọc stance | tổng | chưa dùng |
|---|---:|---:|
| POSITIVE vs NEGATIVE (chặt — bộ lọc cũ) | 2.457 | **2.285** |
| POSITIVE vs NEGATIVE+CONCERN (rộng) | 13.734 | **13.413** |

Con số "1.798" ghi trước đây là bộ lọc chặt; nới sang cả `CONCERN` (4.601 claim) mở ra kho lớn
hơn 5×. Nguyên liệu **không** phải nút thắt — xếp hạng cái nào đáng gán nhãn mới là nút thắt.

### 8.4 · Bốn hướng, theo thứ tự khuyến nghị

**① Swap augmentation — miễn phí, làm trước.**
Quan hệ đối xứng nên mỗi cặp hợp lệ ở cả hai chiều. Thêm bản `(B,A)` **chỉ cho lớp hiếm, chỉ
trong train**, giữ nguyên split: CONTRADICTION 23 → 46 mẫu train, AGREEMENT 65 → 130. Chi phí
gán nhãn = 0, và nó đánh thẳng vào order-flip (nguyên nhân gốc ở mục 0). Đây chính là
`--symmetric-aug` ở mục 3.1, chỉ khác là **áp dụng có chọn lọc theo lớp** thay vì cho tất cả.
⚠ Không cứu được val/test — chỉ cứu train.

**② Mine lại bằng NLI cross-encoder thay vì từ vựng — hướng chính.**
Xếp hạng 13.413 ứng viên theo xác suất `contradiction` của cross-encoder NLI (đã có
`mDeBERTa-v3-base-xnli` dùng ở B2-guard), gán nhãn top 300–400 bằng đúng đường
`manual_claude_rubric` hiện tại. Máy cá nhân chưa có torch, nhưng chạy Colab chỉ vài phút T4.
Kỳ vọng hit rate 15–25% → **+50÷100 CONTRADICTION**, đưa lớp này lên ~6–8%. Đây là hướng duy
nhất cho đủ số lượng để test và fold đọc được.

**③ Sinh counter-claim tổng hợp.**
Có tiền lệ trong pipeline (`synthetic_cross_paper` sinh 110 UNRELATED bằng luật). Cho Claude
viết claim reviewer đối lập về **cùng một điểm cụ thể**. Rẻ, kiểm soát được số lượng.
⚠ Rủi ro thật: model học lối tắt phủ định bề mặt. Nếu làm thì bắt buộc — diễn đạt lại chứ
không phủ định trực tiếp, gắn `source: synthetic_contradiction`, giữ ≤ 50% của lớp, và
**loại hoàn toàn khỏi val/test** (đúng nguyên tắc mục 4.6).

**④ Gộp lớp — phương án lui, khai báo trước khi train.**
Nếu ①+② vẫn không đủ: gộp CONTRADICTION + PARTIAL_CONTRADICTION thành một lớp (**251 mẫu,
22.8%**), báo cáo 5 lớp. Có cơ sở chứ không phải né tránh — chính ranh giới mờ giữa hai lớp đó
là thứ làm ensemble lật nhãn theo thứ tự (mục 0), và là thứ chiếm 70% kết quả mọi bộ lọc ở 8.2.

**Không khuyến nghị** mở rộng corpus (41/263 paper chưa đụng tới): base rate 1.5% nghĩa là chỉ
thêm ~2 mẫu, không đáng công.

### 8.5 · Việc cụ thể khi quay lại

| | |
|---|---|
| ✅ | ~~Sửa `assign_groups()` → stratified-group, chạy lại `split.py`~~ — xong 2026-08-29 |
| ⬜ | Viết `src/data/mine_contradiction.py` — xếp hạng ứng viên bằng NLI, xuất JSONL để gán nhãn |
| ⬜ | Gán nhãn 300–400 ứng viên top, ghi `source: mined_nli_contradiction` |
| ⬜ | Thêm swap-aug chọn lọc theo lớp vào `train.py` |
| ⬜ | Nếu sau đó CONTRADICTION vẫn < 60 mẫu → chốt phương án ④, sửa `LABELS` ở cả split và train |

---

## 9 · Lần train đầu tiên — kết quả và ba thứ đã sửa *(2026-08-29)*

### 9.1 · Kết quả

Đối chiếu công bằng, baseline tính lại trên đúng 1099 cặp (trong notebook baseline chạy trên
test còn model chạy trên CV, không so trực tiếp được):

| | macro-F1 | accuracy |
|---|---:|---:|
| majority (đoán COMPLEMENTARY hết) | 0.104 | 0.451 |
| stance-rule (3 dòng luật) | 0.191 | 0.420 |
| **roberta-base fine-tune, 5-fold CV** | **0.319** | 0.447 |

Fine-tune hơn luật 3 dòng **+0.13 macro-F1**. Accuracy thấp hơn majority là hệ quả cố ý của
trọng số lớp, không phải lỗi.

Per-class (CV, gộp 5 fold):

```
COMPLEMENTARY          0.584  (496)      AGREEMENT              0.292  ( 76)
PARTIAL_AGREEMENT      0.478  (217)      UNRELATED              0.130  (113)  <- bất thường
PARTIAL_CONTRADICTION  0.352  (168)      CONTRADICTION          0.078  ( 29)
```

Phụ trợ: flip-rate thô **0.155** (mục tiêu <10%, nhưng 16/17 ca lệch chỉ 1 bước — mềm hơn hẳn
kiểu 4 bước của ensemble cũ), QWK 0.240, đúng-hoặc-lệch-1-bước **0.858**, ECE 0.338.

> **0.319 là cao hay thấp?** Chưa trả lời được, và đó mới là vấn đề. Con số 8x–9x thường thấy
> đến từ bài toán 2–3 lớp, ranh giới rõ, hàng trăm nghìn mẫu. Ở đây là 6 lớp có thứ tự, ranh
> giới mờ, 1099 mẫu, nhãn LLM. Bỏ hai lớp gần-bằng-0 thì bốn lớp còn lại trung bình 0.43.
> Muốn biết thật thì phải có trần κ — xem "Thiếu sót không thể tự bù" ở mục 5.

### 9.2 · ⚠ Kết quả trên là từ split CŨ

Notebook có **bản sao riêng** của `assign_groups` và tự chia lại tại chỗ, không đọc
`processed/splits/`. Output của nó (`test AGRE:3 CONT:2`, fold 220×5) đúng là bộ split hỏng
đã thay ở mục 2. Nên `AGREEMENT F1 = 0.000` ở mục 7.1 của notebook là do test chỉ có 3 mẫu,
không phải model không học được lớp đó — CV cho thấy 0.292 trên 76 mẫu.

**Đã sửa:** notebook đọc `splits/` và `folds.json`, có `assert` chặn cả rò rỉ paper lẫn
few-shot lọt vào phần chấm điểm.

### 9.3 · 🔴 UNRELATED hỏng từ thiết kế — nguyên nhân lớn nhất

113 mẫu (không phải lớp hiếm) mà F1 chỉ 0.130. **Không phải lỗi model.**

110/113 cặp là `synthetic_cross_paper`: ghép claim của hai paper khác nhau. Đo trên dữ liệu:

| | n | không chung từ nội dung nào | có ≥1 từ chung |
|---|---:|---:|---:|
| UNRELATED (chéo paper) | 110 | **91.8%** | 8.2% |
| COMPLEMENTARY (cùng paper) | 496 | **83.9%** | 16.3% |

Hai phân bố gần trùng. Luật ngưỡng tốt nhất chỉ đạt F1 0.323, trong khi đoán bừa đã 0.307.

```
COMPLEMENTARY (cùng paper):
  A: "The other concern I had is w.r.t. detailed settings of the 'no generator' experiment..."
  B: "The paper is well-written and is easy to follow."

UNRELATED (khác paper):
  A: "The proposed approach is, for the most part, easy to follow and understand."
  B: "The paper seems to focus on the former and does not have any analysis for the latter."
```

Khác biệt duy nhất giữa hai lớp là **metadata "cùng paper hay không"** — thứ model không bao
giờ nhìn thấy, vì `PairSet` chỉ lấy `left.text` và `right.text`. Thông tin định nghĩa ra lớp
không nằm trong input, nên F1 0.130 là hành vi *đúng* của model.

**Và nặng hơn:** hệ thống thật luôn so hai claim của **cùng một paper** — bạn biết điều đó
trước khi gọi model. Cặp chéo paper không bao giờ xuất hiện lúc triển khai, tức 97% dữ liệu
UNRELATED nằm ngoài phân phối của bài toán thật.

Đúng lỗi này đã có ở ensemble cũ: mục 0 ghi *"riêng lỗi UNRELATED → COMPLEMENTARY chiếm 14/29
ca lệch"*. Cùng một nguyên nhân gốc, không phải trùng hợp.

*(Suýt có rò rỉ: 60% cặp chéo paper mang `aspect` ghép dạng `clarity|substance`, COMPLEMENTARY
thì 0%. May là `aspect` không được đưa vào model.)*

**Đã sửa:** `cfg.drop_unrelated = True` → train 5 lớp. Ước tính lợi ích:

| | macro-F1 CV |
|---|---:|
| hiện tại, 6 lớp | 0.319 |
| bỏ UNRELATED, 5 lớp | **0.357** |
| giữ 6 lớp nhưng UNRELATED định nghĩa lại và học được ~0.5 | 0.381 |

Chưa kể lợi ích lan toả: 6 ca COMPLEMENTARY→UNRELATED và 6 ca ngược lại biến mất.

⚠ **Phải khai vào manifest:** checkpoint không có lớp UNRELATED; bên gọi chịu trách nhiệm đảm
bảo hai claim cùng một paper. Nếu sau này cần UNRELATED thật thì phải định nghĩa lại là *cùng
paper nhưng hai claim nói về hai chuyện không liên quan* và gán nhãn lại — ranh giới với
COMPLEMENTARY sẽ rất mờ.

### 9.4 · Overfit và không hiệu chuẩn được

6 epoch cố định kéo train loss xuống **0.24** trên ~1000 mẫu. Hệ quả: confidence khi đúng
0.868 vs khi sai 0.840 — gần như không phân biệt, ECE 0.338. Cắt ngưỡng 0.9 giữ 58% dữ liệu
mà accuracy chỉ nhích 0.518 → 0.547. **Không đặt được ngưỡng ABSTAIN.**

Bằng chứng phụ: fold 4 train kém nhất (loss dừng ở 0.561 thay vì 0.19–0.33) lại đạt macro-F1
0.347, thuộc nhóm cao nhất. Fit ít hơn không mất gì.

Thêm nữa `train_model(train_rows + val_rows, cfg)` **gộp val thẳng vào train** — val không hề
làm nhiệm vụ của val.

**Đã sửa:** `epochs` thành trần (6) + `patience=2`, chọn epoch theo macro-F1 trên val, giữ
state tốt nhất. CV không có val riêng nên train cứng `BEST_EPOCH` mà val đã chọn.

### 9.5 · Learning curve chưa đọc được — CHƯA sửa

`0.226 → 0.169 → 0.220 → 0.268`. Mức 50% thấp hơn mức 25%. Biến động giữa các mức (±0.05) lớn
hơn cả độ lệch giữa các fold (±0.039), vì mỗi mức chỉ chạy một seed và chấm trên test 110 cặp.
Cần ≥3 seed mỗi mức và chấm bằng CV. Cell đã in cảnh báo này; chưa đổi cách chạy.

### 9.6 · Phụ: pin phiên bản không có tác dụng

`pip install transformers==4.44.2` **fail build wheel cho tokenizers**, notebook chạy tiếp
bằng bản Colab có sẵn → kết quả không tái lập. Đã bỏ pin và in bản thật để ghi vào manifest.

### 9.7 · Thứ tự việc còn lại

| | |
|---|---|
| ✅ | Notebook đọc `splits/` + `folds.json` thay vì tự chia |
| ✅ | Bỏ lớp UNRELATED (`cfg.drop_unrelated`) |
| ✅ | Val dùng đúng nghĩa + dừng sớm |
| ⬜ | **Chạy lại notebook** — mọi số ở 9.1 là từ cấu hình cũ |
| ⬜ | Learning curve ≥3 seed, chấm bằng CV (9.5) |
| ⬜ | 50 cặp gán độc lập để có trần κ — không có nó thì không kết luận được 0.319 tốt hay tệ |
| ⬜ | Baseline NLI zero-shot (mục 5) — phép thử gắt nhất, vẫn chưa chạy |
| ⬜ | Chỉ sau đó mới tới mục 8 (CONTRADICTION): kéo lớp này 0.078 → 0.35 chỉ được +0.045 macro-F1, ít hơn 9.3 mà tốn 300–400 nhãn tay |

---

## 10 · Tập gold và việc gán lại nhãn *(2026-08-29)*

`pipeline_data/golden_set/gold_test.jsonl` xuất hiện: **129 cặp, `relation-gold-1.0`,
`HUMAN_VERIFIED`, annotator `NTH`**, cân đều cả 6 lớp (COMP 29 · PART_CONTRA 24 ·
PART_AGREE 24 · AGREE 20 · UNREL 18 · CONTRA 14). Đây chính là thứ mục 5 đòi từ đầu —
nền người để biết 0.319 là gần trần hay còn xa.

**Quyết định: gold là contract chuẩn, silver phải gán lại theo nó.**

### 10.1 · Ba khoảng cách giữa train và gold

| | train (silver) | eval (gold) |
|---|---|---|
| Ngôn ngữ | 1099/1099 **tiếng Anh** | 129/129 **tiếng Việt** |
| Miền | review paper ICLR về ML | phản biện đề tài đại học VN (3 cohort, 17 tiêu chí) |
| Contract nhãn | rubric v1 | [`RUBRIC_GOLD.md`](RUBRIC_GOLD.md) |

Khoảng cách 1 → đổi backbone sang `xlm-roberta-base` (đã làm). Khoảng cách 3 → gán lại
(đang chờ). Khoảng cách 2 chưa xử lý; cell mục 10 của notebook tách theo cohort để đo nó.

### 10.2 · 🔴 Contract lệch ở đúng một lớp

Đối chiếu `why` của silver với `annotation_note` của gold:

| Lớp silver | `why` mở đầu bằng | Khớp gold? |
|---|---|---|
| `AGREEMENT` | "cùng điểm cụ thể… cùng chiều cùng mức độ" — 70/70 | ✅ |
| `PARTIAL_AGREEMENT` | "cùng chiều…, khác phạm vi" — 210/210 | ✅ |
| `PARTIAL_CONTRADICTION` | "cùng điểm…, A khen B chê có điều kiện" — 160/160 | ✅ |
| `CONTRADICTION` | "cùng điểm cụ thể… không thể cùng đúng" — 22/22 | ✅ |
| **`COMPLEMENTARY`** | **"khác điểm cụ thể: X vs Y" — 494/496** | 🔴 phần lớn phải là `UNRELATED` |

Bốn lớp không phải đụng tới — đó là phần đỡ tốn công nhất của phát hiện này.

Gold dành `COMPLEMENTARY` cho trường hợp hẹp hơn nhiều: **cùng một thiếu sót, hai góc bổ sung
nhau**. Phép thử một câu:

> Có tồn tại MỘT vấn đề chung mà cả hai claim đều nhắm tới không?
> có → `COMPLEMENTARY` · không → `UNRELATED`

### 10.3 · 110 cặp `synthetic_cross_paper` bị bỏ hẳn

Sai cả hai mặt: không học được (mục 9.3), **và** sai định nghĩa — gold không hề coi "khác
paper" là UNRELATED, cả 18 cặp UNRELATED của gold đều **cùng cohort, cùng criterion**, chỉ
khác reviewer. Điều này cũng lật lại quyết định bỏ lớp UNRELATED ở mục 9.3: lớp đó **được giữ**,
chỉ là dữ liệu của nó phải đến từ việc gán lại chứ không từ luật ghép chéo paper.

### 10.4 · Đã làm xong

| | |
|---|---|
| ✅ | [`RUBRIC_GOLD.md`](RUBRIC_GOLD.md) — contract rút từ 129 `annotation_note`, cây quyết định 2 chiều |
| ✅ | `src/data/relabel_complementary.py` — xuất batch + `--apply` ghi ngược |
| ✅ | **Gán lại 496 cặp**: 343 → `UNRELATED`, 153 giữ `COMPLEMENTARY`; bỏ 110 synthetic. `1099 → 989` |
| ✅ | `split.py` chạy lại; `SIZE_W` 3.0 → 6.0 vì phân bố nhãn đổi (xem comment trong file) |
| ✅ | Notebook: `xlm-roberta-base`, 6 lớp, chốt chặn ở mục 3, mục 10 chấm trên gold |

Quyết định từng cặp kèm lý do: `pipeline_data/interim/relabel/decisions.jsonl`.
Bản silver trước khi gán lại: `processed/trackB_silver_pre_gold_relabel.jsonl`.

### 10.5 · Split sau khi gán lại

```
train 788 / val 100 / test 101      lệch nhãn lớn nhất 1.6 điểm %  (trước 2.2)
rò rỉ claim text: 0 / 0 / 0         (trước train∩val = 1; cặp chéo paper gây ra đã bị bỏ)
5-fold  kích thước 186-193          CONTRADICTION 4/5/4/5/4
```

`SIZE_W = 3.0` với phân bố mới cho CONTRADICTION 5/5/**2**/5/5 và fold lệch 172-201;
`6.0` cho 4/5/4/5/4 và 186-193 — tốt hơn ở **cả hai** mặt nên không phải đánh đổi.

### 10.6 · ⚠ Còn lệch tiên nghiệm giữa train và gold

| | silver | gold |
|---|---:|---:|
| UNRELATED | **35.0%** | **14.0%** |
| COMPLEMENTARY | 15.5% | 22.5% |
| AGREEMENT | 7.7% | 15.5% |
| CONTRADICTION | 2.9% | 10.9% |

Model sẽ thiên về đoán `UNRELATED` trên gold. Trọng số lớp bù một phần (`UNRELATED w=0.48`,
`CONTRADICTION w=5.25`). Nếu ma trận nhầm lẫn ở muc 10 cho thấy cột UNRELATED phình ra thì
đây là nguyên nhân, **không phải** gán lại sai.

Gốc rễ: B3 ghép cặp theo "cùng paper + **cùng aspect**", mà aspect của ICLR rất thô
(`soundness` gộp 5014 claim) so với `criterion_id` của gold (17 tiêu chí). Aspect thô ⇒ nhiều
cặp cùng nhãn nhưng khác issue ⇒ nhiều UNRELATED. Đây là đặc tính của **cách sinh cặp**,
không sửa được bằng gán nhãn.

### 10.6 · Hai việc còn treo

- **`processed/fewshot.jsonl` (39 ví dụ) vẫn theo rubric cũ.** Nó là few-shot đã dùng để gán
  nhãn 1060 cặp còn lại. Sau khi đổi contract thì nên dựng lại few-shot **từ gold**, không phải
  từ 39 cặp cũ. 8 cặp trong đó đang mang nhãn `COMPLEMENTARY` nên cũng nằm trong danh sách gán lại.
- **Rà phụ ranh giới `COMPLEMENTARY` ↔ `PARTIAL_AGREEMENT`** trên 210 cặp PARTIAL_AGREEMENT.
  Hai lớp này ở gold đều là "cùng hướng, khác góc", khác nhau ở chỗ neo vào một thiếu sót cụ thể
  hay một phán xét tổng thể — xem `RUBRIC_GOLD.md` muc 2. Ưu tiên thấp hơn 496 cặp kia.
- **2/18 cặp UNRELATED của gold có note không khớp phép thử** — xem `RUBRIC_GOLD.md` muc 5.
  Chiếm 1.6%, không chặn việc gì, nhưng nên xác nhận trước khi chốt con số cuối.

---

## 11 · Lần train hợp lệ đầu tiên — có số trên gold *(2026-08-29b)*

Lần chạy ở muc 9 dùng **nhầm notebook**: bản 36 cell trước khi sửa, không phải bản 38 cell đã
commit. Nó chạy `roberta-base`, tự chia lại split, gộp val vào train, và **không có cell chấm
gold**. Lần này là bản đúng — lần đầu tiên có con số trên nền người.

### 11.1 · Kết quả

| | muc 9 (`roberta-base`, notebook cũ) | **muc 11 (`xlm-roberta-base`, notebook đúng)** |
|---|---:|---:|
| CV macro-F1 (silver, 950 cặp) | 0.3014 ± 0.0564 | **0.2836 ± 0.0393** |
| test EN (101 cặp) | 0.4896 | 0.2932 |
| **GOLD (129 cặp, VI)** | *không đo được* | **0.2548** |
| majority / stance-rule | 0.0431 / 0.1597 | như bên trái |

Số xuống. **Nguyên nhân không phải là những thứ vừa sửa** — xem 11.2.

### 11.2 · 🔴 Model chưa hội tụ — nguyên nhân lớn nhất

Cùng 6 epoch, cùng `lr`, chỉ khác backbone:

```
epoch      1       2       3       4       5       6
roberta   1.7978  1.4977  1.0608  0.7328  0.4834  0.3395
xlm-r     1.8070  1.7442  1.4531  1.1270  0.8744  0.7267
delta    +0.009  +0.247  +0.392  +0.394  +0.391  +0.387
```

Khoảng cách mở ra ở epoch 3 rồi **đứng yên** — `xlm-r` không chậm hơn, nó bị cắt trước khi kịp
vào giai đoạn hội tụ. Trong CV (5 epoch, không val) còn nặng hơn:

```
fold 0  loss=1.2325  (39% quãng đường tới hội tụ)  macro-F1=0.2559
fold 1  loss=0.9822  (56%)                          macro-F1=0.3002
fold 2  loss=0.9045  (61%)                          macro-F1=0.2547
fold 3  loss=1.3703  (29%)                          macro-F1=0.3539   <- ít train nhất, điểm cao nhất
fold 4  loss=1.2216  (39%)                          macro-F1=0.2535
                                        (đoán bừa = ln 6 = 1.7918)
```

Tương quan loss <-> macro-F1 là **r = +0.45**: fold train tệ nhất cho điểm cao nhất. Đó là chữ
ký của vùng nhiễu, không phải của học. `epochs=6` / `lr=2e-5` được chỉnh cho `roberta-base`.
Early stopping còn làm nặng thêm: val đang leo đều (0.2913 -> 0.3017 -> 0.3109) thì hụt một
nhịp ở epoch 6 (0.2822) là chốt luôn epoch 5.

> **0.3014 -> 0.2836 không phải "sửa xong thì tệ đi".** Đó là một `roberta` đã hội tụ so với
> một `xlm-r` chưa hội tụ. Hai thứ khác nhau, chưa so được với nhau.

### 11.3 · ✅ Đổi backbone đã làm đúng việc — kết quả tốt nhất của lần chạy

Gold 0.2548 vs test tiếng Anh 0.2932: **chênh chỉ 0.038**, trong khi train là 100% tiếng Anh
(review ICLR về ML) còn gold là 100% tiếng Việt (phản biện đề tài đại học VN). Vừa
cross-lingual zero-shot vừa đổi miền mà chỉ mất 0.04 là chuyển giao rất tốt — và đạt được
**ngay cả khi model chưa hội tụ**. `roberta-base` không thể cho con số này ở bất kỳ giá nào.

**Khoảng cách 1 của muc 10.1 coi như đã đóng.** Khoảng cách 2 (miền) thì chưa: acc theo cohort
là 0.216 / 0.385 / 0.308 — chênh gần 2 lần.

### 11.4 · 🔴 Lệch tiên nghiệm xảy ra đúng như muc 10.6 dự báo

| lớp | prior silver | thật (gold) | model đoán | tỉ lệ |
|---|---:|---:|---:|---:|
| UNRELATED | 34.6% | 14.0% | **35.7%** | **2.56x** |
| COMPLEMENTARY | 15.5% | 22.5% | 29.5% | 1.31x |
| PARTIAL_CONTRADICTION | 17.0% | 18.6% | 5.4% | **0.29x** |
| CONTRADICTION | 3.2% | 10.9% | 0.0% | **0.00x** |

Model đoán UNRELATED 35.7% trên gold — **sao chép gần như nguyên prior của silver (34.6%)**,
trong khi sự thật chỉ 14%. Muc 10.6 viết trước khi chạy: *"nếu ma trận nhầm lẫn ở mục 10 cho
thấy cột UNRELATED phình ra thì đây là nguyên nhân, **không phải** gán lại sai."* Nó phình ra,
đúng 2.56 lần.

-> Nên dòng tự chẩn đoán của cell mục 10 — *"nếu COMPLEMENTARY <-> UNRELATED vẫn dẫn đầu thì
việc gán lại chưa tới nơi"* — **đang đọc sai tình huống của chính nó**. Ô 14x COMPLEMENTARY ->
UNRELATED là hệ quả cơ học của cột phình, không phải bằng chứng relabel hỏng. Điều kiện đó chỉ
có nghĩa **sau khi** đã bù prior. **Đã sửa** — xem 11.9.

### 11.5 · 🔴 `axis_dist` chấm ngược contract — dùng phân rã Q1/Q2

`axis_dist` giả định 6 nhãn nằm trên **một trục**. Cây quyết định của `RUBRIC_GOLD.md` muc 1 là
**hai chiều**, và trong đó `PARTIAL_AGREEMENT` nằm ở nhánh *khác-issue* cùng với `UNRELATED`,
không phải cạnh `AGREEMENT`. Hệ quả rơi đúng vào hai ô lỗi lớn nhất:

| ô lỗi | `axis_dist` | theo cây gold |
|---|---|---|
| COMPLEMENTARY <-> UNRELATED | 1 — *nhẹ* | **sai Q1** (vượt nhánh, loại nặng nhất) |
| PARTIAL_AGREEMENT <-> UNRELATED | 3 — *nặng* | sai Q2 (cùng nhánh) |

Nên `đúng-hoặc-lệch-1-bước` và `sai nặng` **không dùng để kết luận được**. Thay bằng phân rã
Q1/Q2 (đã thêm vào notebook):

| | đúng | sai Q1 (nhầm nhánh) | sai Q2 |
|---|---:|---:|---:|
| test EN — muc 9 (đã hội tụ) | 0.495 | 0.293 | 0.212 |
| test EN — muc 11 | 0.356 | 0.465 | 0.178 |
| **GOLD VI** | 0.295 | **0.496** | 0.209 |

**Sai Q2 gần như bất động (0.18–0.21) qua cả ba cột.** Toàn bộ phần suy giảm dồn vào Q1. Trên
gold, một nửa số cặp nhầm ngay ở câu *"hai claim có cùng một issue không"*.

Kiểm chứng phụ, gộp nhãn trên ma trận nhầm lẫn của gold: gộp dọc trục hầu như không được gì
(6->5 gộp CONTRADICTION vào PARTIAL_CONTRADICTION: 0.2548 -> 0.3027; 6->4 gộp cả hai đầu trục:
0.3331), nhưng riêng Q1 nhị phân đã là 0.4929. **Vấn đề không phải "6 nhãn khó", mà là một
quyết định nhị phân model chưa làm được** — và nó là thứ mất đi đầu tiên khi model thiếu train.

### 11.6 · CONTRADICTION: đã đủ bằng chứng để kết luận

- CV: F1 0.030, precision 0.022, recall 0.045 -> **đoán 45 lần, đúng 1**.
- Gold: **không đoán lần nào** (cột CONTRA toàn 0), dù gold có 14 mẫu = 10.9%.
- CV chỉ chấm được **22** mẫu chứ không phải 29 — 39 cặp few-shot bị ghim vào train ở mọi vòng
  nên 989 -> 950.

Trọng số lớp 5.25–6.3 không cứu được; ở mức này lớp đó **bơm nhiễu vào macro-F1** chứ không
đóng góp. Muc 9.7 xếp nó cuối cùng là đúng: kéo 0.000 -> 0.35 chỉ được +0.058 macro, tốn
300–400 nhãn tay.

### 11.7 · Hiệu chuẩn và learning curve

**Hiệu chuẩn khá hơn nhưng chưa tính là thắng.** ECE 0.3469 -> 0.3092; confidence khi đúng
0.855 -> 0.709. Ngưỡng 0.8 giữ 26.7% với acc 0.519 so với nền 0.356 — **+0.16**, trước chỉ
+0.05. Ngưỡng ABSTAIN bắt đầu khả thi. Nhưng model chưa hội tụ thì confidence thấp là đương
nhiên — phải đo lại sau khi hội tụ.

*(Phát hiện kèm: `predict_logits` **cộng** logit hai chiều rồi mục 7.7 `softmax` lên tổng đó.
Cộng hai logit làm thang logit gấp đôi -> softmax nhọn giả tạo -> ECE đo ra tệ hơn thực tế. Đã
đổi sang **trung bình**; `argmax` không đổi nên mọi macro-F1 giữ nguyên.)*

**Learning curve vẫn chưa đọc được.** `0.115 -> 0.225 -> 0.206 -> 0.219`; lần trước
`0.226 -> 0.169 -> 0.220 -> 0.268`. Hai lần chạy nghịch nhau ở hai chỗ khác nhau. Thêm một lý
do ngoài "1 seed / chấm trên test 101 cặp" đã nêu ở muc 9.5: mức 25% kết thúc ở **loss 1.6684**,
sát mức đoán bừa 1.79 — đường cong đang đo optimizer thất bại chứ không đo giá trị của dữ liệu.

### 11.8 · Ba lỗi nhỏ

- **Cell mục 11 crash**: `cfg.drop_unrelated` đã đổi tên thành `drop_synthetic_unrelated` ->
  `AttributeError`. Checkpoint và `labels.json` **đã lưu xong** trước khi crash, chỉ dòng cảnh
  báo cuối chết. Nội dung cảnh báo đó cũng hết đúng: quyết định bỏ lớp UNRELATED ở muc 9.3 đã
  bị lật ở muc 10.3.
- **Cell mục 3** in `đã bỏ synthetic_cross_paper: 989 -> 989 cặp` — 0 cặp, vì chúng đã bị loại
  từ lúc gán lại. Nghe như bộ lọc còn đang hoạt động.
- **`torch.manual_seed` gọi một lần lúc định nghĩa cell**, không gọi trong `train_model`. Mỗi
  fold và mỗi mức learning-curve khởi tạo từ RNG state khác nhau -> ±0.0393 của CV trộn phương
  sai dữ liệu với phương sai seed.

### 11.9 · Đã sửa trong notebook

| | sửa gì | vì |
|---|---|---|
| ✅ | `epochs` 6 -> **20**, `patience` 2 -> **4** | 11.2 |
| ✅ | `train_model` in `CHƯA HỘI TỤ` nếu loss cuối > 0.6 | 11.2 — chặn việc đọc nhầm một model chưa train xong thành model kém |
| ✅ | `sched_epochs`: CV dùng chung quỹ đạo lr với mục 6 | CV train `BEST_EPOCH` epoch nhưng epoch đó do val chọn dưới lịch dài `cfg.epochs`; lịch ngắn hơn thì lr decay về 0 sớm, hai số hết so được |
| ✅ | `torch/np/random.seed` chuyển **vào trong** `train_model` | 11.8 — tách phương sai dữ liệu khỏi phương sai khởi tạo |
| ✅ | `prior_shift_logits()`; mục 10 báo 3 số: thô / prior đều / prior gold | 11.4 |
| ✅ | Điều kiện cảnh báo COMP<->UNREL chỉ xét **sau khi bù prior** | 11.4 |
| ✅ | `q1_q2_split()` thêm vào mục 7.3, 8 và 10 | 11.5 |
| ✅ | `predict_logits` TTA: tổng -> **trung bình** | 11.7 |
| ✅ | `cfg.drop_unrelated` -> cảnh báo lệch tiên nghiệm; `labels.json` ghi `train_prior` | 11.8 |
| ✅ | Thông báo `synthetic_cross_paper`; bảng prior train-vs-gold in ngay ở mục 3 | 11.8 |

Bản notebook trước khi vá: `notebooks/train_relation_classifier.ipynb.bak`.
Output của các cell đã sửa được xoá — số của lần chạy đó nằm ở muc 11.1–11.7.

### 11.10 · Thứ tự việc còn lại

| | |
|---|---|
| ⬜ | **Chạy lại notebook, đọc dòng `loss=` TRƯỚC.** Loss cuối phải xuống 0.3–0.5. Nếu 20 epoch vẫn chưa tới thì `lr=3e-5` rồi chạy lại. Chưa hội tụ thì mọi số bên dưới vô nghĩa |
| ⬜ | Đọc dòng `prior đều` ở mục 10 — nếu cao hơn số thô đáng kể thì bù prior là món rẻ nhất trong cả danh sách |
| ⬜ | Learning curve >=3 seed, chấm bằng CV (muc 9.5) — chỉ có nghĩa sau khi hội tụ |
| ⬜ | 50 cặp gán độc lập để có trần kappa — vẫn chưa có, vẫn là thứ chặn việc kết luận 0.25 là gần trần hay còn xa |
| ⬜ | Baseline NLI zero-shot (muc 5) — phép thử gắt nhất, vẫn chưa chạy |
| ⬜ | Chỉ sau khi hội tụ mới quyết: **hai đầu ra Q1 nhị phân + Q2 5-way** (11.5) hay tăng mẫu CONTRADICTION (muc 8). Bằng chứng hiện tại nghiêng hẳn về Q1 |

---

## 12 · Hội tụ được, nhưng hỏng ở đầu kia *(2026-08-29c)*

Trần epoch 6 -> 20 có tác dụng: loss xuống 0.0083, val macro-F1 lên 0.3109 -> 0.3741. Nhưng
**test và gold cùng giảm.** Muc 11.2 chẩn đoán thiếu train là đúng; lần này lỗi nằm ở phía
ngược lại.

### 12.1 · Kết quả

| | muc 11 (2026-08-29b, epoch 5) | **muc 12 (2026-08-29c, epoch 10)** |
|---|---:|---:|
| val macro-F1 *(dùng để chọn epoch)* | 0.3109 | **0.3741** ↑ |
| test EN | **0.2932** | 0.2265 ↓ |
| CV silver | 0.2836 ± 0.0393 | 0.2894 ± **0.0141** |
| **GOLD** | **0.2548** | **0.1971** ↓ |
| ECE | 0.3092 | **0.5227** ↓↓ |
| flip-rate gold / test | 0.163 / 0.158 | **0.101** / 0.139 |

### 12.2 · 🔴 Val chọn sai epoch — val quá nhỏ để phân xử

```
epoch   6       7       8       9       10      11      12      13      14
loss    0.6218  0.3784  0.1983  0.1011  0.0557  0.0287  0.0196  0.0092  0.0083
val     0.3273  0.3373  0.3058  0.3226  0.3741* 0.3553  0.3391  0.3183  0.2891
```

Val chọn epoch 10 — nơi train loss đã là **0.0557**, tức thuộc lòng. Val tăng còn test và
gold giảm: đó là overfit **qua khâu chọn epoch**, không phải qua khâu train.

Nguyên nhân: val chỉ 100 cặp và có `CONT:2`. Nếu model bắt trúng 1 trong 2 mẫu CONTRADICTION
với precision cao thì riêng lớp đó nhảy từ F1 0 lên ~0.67, **đóng góp ±0.11 vào macro-F1 val
chỉ từ một mẫu** — lớn hơn toàn bộ khoảng cách giữa epoch 7 và epoch 10 (0.037). Val ở kích
thước này không phân xử được câu hỏi đang hỏi.

### 12.3 · 🔴 Bù prior không ăn — và biết vì sao

```
thô, không bù                          0.1971
prior ĐỀU (label-free)                 0.1969      chỉ 6/129 dự đoán đổi
prior GOLD (oracle)                    0.2024
```

Muc 11.4 xếp đây là món rẻ nhất còn lại. Nó hỏng, vì **logit đã bão hoà**: ECE 0.5227,
confidence 0.932 khi đúng / 0.864 khi sai. Confidence 0.93 trên 6 lớp tương ứng khoảng cách
logit ~4.2 so với lớp nhì, trong khi biên độ hiệu chỉnh prior chỉ ~2.4
(`-log 0.032 = 3.44` cho CONTRADICTION, `-log 0.346 = 1.06` cho UNRELATED). Không đủ lật
argmax.

> **Bù prior cần logit còn hiệu chuẩn được.** Nó không phải món độc lập — nó nằm *sau* việc
> gỡ train quá tay trong chuỗi nhân quả. Đo lại sau khi ECE về vùng 0.3.

### 12.4 · PARTIAL_AGREEMENT là *bể chứa*, không phải *nguồn* — đừng bỏ

Ma trận nhầm lẫn gold làm lớp này trông như thủ phạm. Nó không phải.

| lớp | gold F1 | hạng | CV F1 | hạng | n (CV) |
|---|---:|---:|---:|---:|---:|
| UNRELATED | 0.435 | 1 | 0.448 | 1 | 339 |
| AGREEMENT | 0.296 | 2 | 0.341 | 3 | 70 |
| **PARTIAL_AGREEMENT** | **0.289** | **3** | **0.407** | **2** | 210 |
| COMPLEMENTARY | 0.091 | 4 | 0.256 | 5 | 149 |
| PARTIAL_CONTRADICTION | 0.071 | 5 | 0.323 | 4 | 160 |
| CONTRADICTION | 0.000 | 6 | 0.000 | 6 | 22 |

Trên CV nó là lớp **tốt thứ nhì**; bỏ khỏi trung bình macro làm điểm **giảm** 0.2958 -> 0.2736.

Nó trông tệ vì bị đoán quá tay (52 lần trên gold cho 24 mẫu thật), và **bể chứa đó là ngẫu
nhiên** — nó đổi chỗ giữa hai lần chạy trên cùng dữ liệu, cùng rubric, cùng split, chỉ khác
epoch:

```
số lần model ĐOÁN         2026-08-29b (ep5)   2026-08-29c (ep10)   thật
PARTIAL_AGREEMENT                22                  52             24
COMPLEMENTARY                    38                  15             29
UNRELATED                        46                  51             18
```

Bỏ một bể chứa chỉ chuyển dòng rò sang bể khác. Và gộp nó cũng không phải món lời nhất —
đo trên gold:

```
gộp P_AGREE vào COMPLEMENTARY      +0.050
gộp P_AGREE vào AGREEMENT          +0.027
gộp P_AGREE vào UNRELATED          -0.009    (tệ đi)
--- đối chứng ---
gộp COMPLEMENTARY vào UNRELATED    +0.062    <- lời hơn mọi phương án P_AGREE
gộp CONTRADICTION vào P_CONTRA     +0.044
chỉ giữ Q1 nhị phân                +0.212    <- lớn hơn tất cả cộng lại
```

**Kết luận: không đụng vào bộ nhãn.**

### 12.5 · 🔴 Nhánh Q1 bị lật ngược giữa silver và gold

```
                            cùng-issue   khác-issue
sự thật (gold)                 67.4%       32.6%
prior silver (train)           43.4%       56.6%    <- đã ngược sẵn
model 2026-08-29b (ep5)        47.3%       52.7%
model 2026-08-29c (ep10)       20.2%       79.8%    <- đi xa hơn cả prior train
```

Gold có 2/3 là *cùng issue*; silver dạy 43%; model lần này đoán 20%. Sai Q1 trên gold là
**0.581**. Đây là chỗ mất điểm lớn nhất, và nó độc lập với việc bộ nhãn có 5 hay 6 lớp.

Phân rã Q1/Q2 lần này (thước đo mới thêm ở muc 11.9):

| | đúng | sai Q1 | sai Q2 |
|---|---:|---:|---:|
| test EN | 0.366 | 0.416 | 0.218 |
| CV silver | 0.371 | 0.438 | 0.192 |
| **GOLD** | 0.256 | **0.581** | 0.163 |

Sai Q2 vẫn bất động quanh 0.16–0.22 như ở muc 11.5. Mọi thứ xấu đi đều dồn vào Q1.

### 12.6 · ✅ Ba thứ lần này làm được

- **Phương sai CV sụp: ±0.0393 -> ±0.0141.** Việc chuyển `manual_seed` vào trong
  `train_model` (muc 11.9) có tác dụng đúng như dự tính.
- **flip-rate trên gold 0.163 -> 0.101** — chạm mốc <10% đặt từ muc 4.5, mốc mà pipeline
  ensemble cũ trượt xa (Qwen 50%, Gemma 44%).
- **Learning curve lần đầu đọc được**, vì cả bốn mức đều hội tụ:
  `0.183 -> 0.229 -> 0.222 -> 0.224` tại n = 197 / 396 / 606 / 788.
  **Phẳng từ n≈400.** Câu hỏi treo từ muc 9.5 có câu trả lời: *thêm silver kiểu này không
  còn lời.* (Vẫn 1 seed, chấm trên test 101 cặp — nhưng ba mức cuối nằm trong 0.006 của nhau
  thì kết luận "phẳng" không nhạy với nhiễu đó.)

### 12.7 · Đã sửa trong notebook

| | sửa gì | vì |
|---|---|---|
| ✅ | `min_val_support=5` — lớp dưới 5 mẫu val bị loại khỏi **tiêu chí chọn epoch** (`val_sel`); vẫn train, vẫn chấm ở mọi mục khác | 12.2 — CONTRADICTION (n=2) đang một mình lái ±0.11 macro-F1 val |
| ✅ | `min_delta=0.01` — hoà thì giữ epoch **sớm hơn** | 12.2 — thiên vị về phía ít thuộc lòng |
| ✅ | `loss_floor=0.30` — chạm sàn là dừng, bất kể val nói gì | 12.2 |
| ✅ | In `val6` (6 lớp, để so lần cũ) **và** `val_sel` (tiêu chí thật); in cả train loss tại epoch được chọn | 12.2 |
| ✅ | **CV cắt theo `loss_floor`, không theo `BEST_EPOCH`** | các fold hội tụ rất khác nhau: cùng 10 epoch, fold 2 kết ở loss 0.0448 còn fold 4 ở 0.3365. Cắt cứng ở epoch 6 thì **4/5 fold dừng ở loss 0.81–1.45**, chưa hội tụ |

Mô phỏng luật mới trên chính chuỗi loss/val của lần chạy này: chọn **epoch 6** (loss 0.6218)
thay vì epoch 10 (loss 0.0557), và dừng sau epoch 8. CV thì mỗi fold dừng ở epoch 7–11 với
loss 0.23–0.29 — cùng mức fit, cả 5 fold đều qua ngưỡng hội tụ.

### 12.8 · Việc còn lại

| | |
|---|---|
| ⬜ | **Chạy lại.** Kỳ vọng: test/gold về ≥ mức 2026-08-29b (0.2932 / 0.2548), ECE về vùng 0.3. Nếu gold vẫn dưới 0.25 thì luật chọn epoch không phải nguyên nhân còn lại |
| ⬜ | **Rồi mới đọc lại bù prior** ở mục 10 — nó chỉ có đòn bẩy khi logit hết bão hoà (12.3) |
| ⬜ | Nếu hai bước trên xong mà gold vẫn ~0.25: chuyển sang **hai đầu ra Q1 nhị phân + Q2 5-way**. Bằng chứng đã đủ mạnh (12.4, 12.5) và mọi phương án gộp nhãn đều thua xa nó |
| ⬜ | Lệch prior nhánh silver-vs-gold (43% / 67%) là đặc tính của **cách sinh cặp** ở B3, không sửa được bằng gán nhãn — xem muc 10.6 |
| ⬜ | 50 cặp gán độc lập để có trần κ — vẫn là thứ chặn việc kết luận 0.25 là gần trần hay còn xa |
| ⬜ | Baseline NLI zero-shot (muc 5) — vẫn chưa chạy |
| ⬜ | Learning curve: đã đọc được là "phẳng từ n≈400", nên **muc 8 (tăng mẫu CONTRADICTION) tụt ưu tiên thêm một bậc** — thêm dữ liệu cùng kiểu không phải hướng đi |

---

## 13 · Ba mức nhãn, hold-out gold, và baseline có thật *(2026-08-29d, notebook đã vá — CHƯA chạy)*

Muc 12 để lại một bế tắc có thật: mọi con số nằm quanh 0.20–0.29 và không có mốc nào để
nói nó tốt hay tệ. Nhưng ba trong bốn nguyên nhân của bế tắc đó là **cách đo**, không phải
model.

### 13.1 · Đo bằng thước khắc nghiệt nhất có thể

`macro-F1` trên 6 lớp **chia đều** cho một lớp có 22 mẫu CV và F1 = 0.000. Muc 11.6 đã ghi
lớp đó *"bơm nhiễu vào macro-F1 chứ không đóng góp"* — và mọi con số từ muc 9 tới muc 12
đều trả cái giá đó. Không có gì sai với việc báo cáo full6; sai là **chỉ** báo cáo full6.

Notebook giờ có `LABEL_VIEW` với ba mức, và mọi chỗ chấm in **cả ba trong một lần chạy**
(`eval_views`) — không phải train ba lần, mà đọc lại cùng một dự đoán ở độ mịn thô hơn,
đúng như hệ thống thật sẽ làm.

| view | lớp | trả lời câu hỏi |
|---|---|---|
| `full6` | 6 | contract nghiên cứu. Con số trung thực nhất **và thấp nhất** |
| `merge5` | 5 | `PARTIAL_CONTRADICTION` + `CONTRADICTION` → một lớp. Lớp đó từ 22 lên ~182 mẫu CV |
| `deploy3` | 3 | `CONSENSUS` / `CONFLICT` / `UNRELATED` — độ mịn downstream thật sự tiêu thụ |

`merge5` là phép gộp **duy nhất** có cơ sở cấu trúc: hai lớp đó cùng nhánh *cùng-issue*,
cùng chiều, chỉ khác **mức độ** — một trục độ thật. Đo được +0.048 (11.5) và +0.044 (12.4)
trên hai checkpoint khác nhau, phép gộp duy nhất lặp lại được.

> 🔴 **Không gộp `PARTIAL_AGREEMENT` vào `AGREEMENT`** dù nó ăn +0.027. P_AGREE nằm ở nhánh
> *khác-issue* — số học xác nhận: 18.6 + 14.0 = **32.6%**, khớp đúng tỉ lệ khác-issue đo ở
> 12.5. Gộp nó là vượt Q1, đúng ranh giới đang gánh 0.581 lỗi.

> ⚠ **Δ macro-F1 giữa các mức KHÔNG dùng để chọn bộ nhãn.** Gộp hai lớp mà model đang lẫn
> thì điểm luôn tăng bất kể phép gộp có nghĩa hay không. Bằng chứng: `COMPLEMENTARY →
> UNRELATED` ăn **+0.062** (12.4), cao nhất trong mọi phương án — mà đó là phép nhập hai
> nhánh Q1 tại đúng chỗ chúng chia đôi. Bảng Δ đo **độ lẫn của model**, không đo bài toán.
> Đây là lý do đúng cho kết luận *"không đụng vào bộ nhãn"* ở 12.4, thay cho lý do
> "bể chứa ngẫu nhiên" đã ghi ở đó.

### 13.2 · 🔴 Gold vừa là tập chọn vừa là tập báo cáo

Tới hết muc 12 đã có **~8 phương án bộ nhãn được so trên chính gold** (11.5, 12.4), trong
khi gold chỉ 129 cặp và là con số báo cáo. Mọi kết quả kiểu "+0.078" vì thế **không kiểm
chứng được**.

Đã sửa: `cfg.gold_holdout_cohort` giữ riêng `case-03-admissions-rag-chatbot` (39 cặp),
luật chọn cố định — cohort cuối theo thứ tự alphabet, chốt **trước** khi nhìn kết quả.
Mọi chẩn đoán ở mục 10 giờ đọc trên DEV (90 cặp); HOLD-OUT chỉ in ở bảng tổng hợp.

### 13.3 · Baseline: hai cái sai, một cái thiếu

- **`majority` ghim cứng `COMPLEMENTARY`** — tàn dư từ trước đợt gán lại ở muc 10, khi lớp
  đó còn chiếm 45%. Sau khi gán lại thì lớp đông nhất là `UNRELATED` (35.0%). Ghim sai lớp
  làm baseline **yếu đi giả tạo**, tức thổi phồng khoảng cách model-vs-baseline. Đã sửa:
  lấy đúng lớp đông nhất của train.
- **macro-F1 tính trên tập nhãn khác nhau giữa các dòng** — `majority` chỉ đoán 1 lớp nên
  sklearn chia macro cho ít lớp hơn dòng model. Đã ghim `labels=` toàn bộ lớp của view.
- **NLI zero-shot chưa từng chạy** dù được ghi là *"phép thử gắt nhất"* từ muc 5 qua 5 mục.
  Đã thêm cell 4.b: `mDeBERTa-v3-base-xnli`, trung bình xác suất cả hai thứ tự, ánh xạ
  `entailment/contradiction/neutral → CONSENSUS/CONFLICT/UNRELATED`. Không ngưỡng, không
  tham số — không có chỗ nào tinh chỉnh cho vừa kết quả.

  Ánh xạ đó chỉ tự nhiên ở `deploy3`, và **đó là một kết quả chứ không phải giới hạn của
  phép so**: model có sẵn không diễn đạt nổi contract 6 lớp (không có khái niệm "cùng
  chiều nhưng khác phạm vi"). Ô "—" ở full6/merge5 nên được báo cáo đúng như vậy.

### 13.4 · Tỉ lệ nhãn: chỉnh bằng trọng số, đừng resample

Câu hỏi "có cần sửa tỉ lệ nhãn của tập train không". Có lệch thật, và nó là con số **duy
nhất sống sót qua mọi phương án gộp lớp**:

| view | chênh lớn nhất |
|---|---|
| full6 | `UNRELATED` silver 35.0% vs gold 14.0% → **+21.0 điểm** |
| merge5 | `UNRELATED` → **+21.0** |
| deploy3 | `UNRELATED` → **+21.0** |

Ba cách chỉnh, **không dùng đồng thời quá một**:

| | cách | đánh giá |
|---|---|---|
| (a) | `cfg.weight_target="gold_dev"` — trọng số lớp nhắm vào prior thật của miền đích thay vì phân bố đều | ✅ rẻ nhất, không mất dữ liệu, một dòng. Khai vào manifest: dùng phân bố nhãn của 2/3 cohort gold, không dùng nhãn cohort hold-out |
| (b) | `prior_shift_logits` lúc suy luận | ✅ đảo ngược được, nhưng 12.3 đã đo là **hỏng khi logit bão hoà** — chỉ đo lại sau khi ECE về vùng 0.3 |
| (c) | resample tập train | 🔴 **không khuyến nghị**: để ép `UNRELATED` 35.0% → 14.0% phải vứt **189/273** cặp, train 788 → 599. Vứt nhãn thật để làm đẹp một phân bố mà (a) đạt cùng hiệu quả với chi phí bằng 0 |

Lưu ý (a) là **tổng quát hoá của cái đang chạy**, không phải thứ mới: `w ∝ 1/p_train`
hiện tại chính là "(a) với đích = phân bố ĐỀU". Mà gold không hề đều — nên mặc định hiện
tại đang bù **quá tay theo hướng ngược lại**.

Và gốc rễ vẫn không nằm ở đây: +21.0 điểm sinh từ đợt gán lại ở muc 10 cộng độ mịn ghép
cặp ở 10.6, tức **cách sinh dữ liệu**. Trọng số chỉ che, không chữa.

### 13.5 · Cái đã có mà chưa bao giờ đưa lên bảng

Mục 11 mới (`REPORT` + bảng tổng hợp) gom sẵn để dán vào báo cáo. Hai thứ đã thắng từ lâu
mà chưa từng xuất hiện cạnh nhau:

- **flip-rate 0.101 trên gold** (12.6) — đạt mốc <10% đặt ở 4.5, mốc mà pipeline gốc
  trượt xa (Qwen 50%, Gemma 44%, SeaLLM 18%). Đây là lỗi đã **giết** Track B (muc 0:
  94% số cặp đi debate là do một model tự mâu thuẫn khi đảo thứ tự), và nó đã được sửa.
- **Ensemble gốc**: 0% unanimous, 53% phải debate, khớp nhãn tay **48.2%**. Đó mới là
  "base model gốc" của dự án, và nó hỏng.

### 13.6 · Thứ tự chạy

| | | kỳ vọng |
|---|---|---|
| 1 | Chạy **mục 3.b** trước hết — nó in tỉ lệ cặp bị cắt ở `max_len=160`. Gold dài gấp 1.7× silver (31.4 vs 18.0 từ/claim) và tiếng Việt tách nhiều subword hơn | nếu gold bị cắt hơn train >5 điểm thì **tăng `max_len` rồi mới chạy tiếp** — mọi số gold trước đó là cận dưới |
| 2 | Chạy hết `LABEL_VIEW="full6"` | test/gold về ≥ mức 2026-08-29b, ECE về vùng 0.3 (12.8) |
| 3 | Đọc **mục 4.b** (NLI) trước khi đọc mục 11 | con số trả lời "hơn model có sẵn không" |
| 4 | Đọc bảng mục 11, dòng `deploy3` | đây là số triển khai |
| 5 | Chỉ khi (2) xong mới thử `weight_target="gold_dev"`, rồi mới đọc lại bù prior | 13.4, và 12.3 |

⚠ **Vẫn chưa có trần κ.** Không có nó thì không kết luận được con số nào là gần trần hay
còn xa. Rẻ nhất bây giờ: người thứ hai gán **40 cặp gold**, tính κ với NTH — gold mới là
số báo cáo nên trần phải đo trên gold, không phải trên 50 cặp silver như muc 5 đề xuất.

### 13.7 · Đã sửa trong notebook

| | sửa gì | vì |
|---|---|---|
| ✅ | `LABEL_VIEW` + `eval_views` — in cả 3 mức nhãn ở mục 7, 8, 10 | 13.1 |
| ✅ | `cfg.gold_holdout_cohort`; mục 10 tách DEV/HOLD, chẩn đoán chi tiết chỉ đọc DEV | 13.2 |
| ✅ | Mục **3.b** mới: nạp gold sớm, chia cohort, đo tỉ lệ bị cắt ở `max_len` | 13.2, 13.6 |
| ✅ | Mục **4.b** mới: baseline NLI zero-shot | 13.3 |
| ✅ | `majority` lấy lớp đông nhất thật; baseline chấm trên cả test/gold-dev/gold-hold | 13.3 |
| ✅ | `f1_score(labels=...)` ghim tập nhãn ở mọi dòng | 13.3 |
| ✅ | `cfg.weight_target` — trọng số lớp nhắm `uniform` hoặc `gold_dev` | 13.4 |
| ✅ | Mục **11** mới: bảng tổng hợp model-vs-baseline + flip-rate + phần "khai kèm" | 13.5 |
| ✅ | `labels.json` ghi thêm `label_view`, `view_map`, `weight_target`, `gold_holdout_cohort` | manifest |

Bản trước khi vá: `notebooks/train_relation_classifier.ipynb.bak2`.
Mọi output của lần chạy muc 12 đã bị xoá — số của lần đó nằm ở 12.1–12.6.

## 14 · Bỏ loss_floor gold-tuned, chọn epoch chỉ bằng train/val *(2026-08-29e)*

### 14.1 · Vấn đề

`loss_floor=0.30` (muc 12.2) là một hằng số chọn **bằng cách nhìn gold**: có đúng 2 điểm dữ
liệu — loss dừng ở 0.87 → gold 0.2548, loss dừng ở 0.056 → gold 0.1971 — và 0.30 được đặt
nằm giữa hai điểm đó. Cơ chế DEV/HOLD-OUT ở muc 13.2 (`cfg.gold_holdout_cohort`) cũng không
sửa được gốc rễ: nó vẫn đọc 90/129 cặp gold để ra quyết định trước khi khoá candidate.
Không dùng bất kỳ phần nào của gold để chọn epoch/checkpoint — bất kể ai gán nhãn gốc cho
gold — là điều kiện bắt buộc trước khi khoá candidate cho benchmark ngoài.

### 14.2 · Sửa

Bỏ hẳn `cfg.loss_floor`. Thay bằng **val loss**, tính mỗi epoch cùng lúc với `val_sel`
(dùng đúng `loss_fn` có trọng số lớp, không phải CE trần) — số liên tục trên cả 6 lớp, không
bị một mẫu CONTRADICTION lật làm nhảy ±0.11 như macro-F1. Dừng sớm ở tín hiệu nào tới trước:

- `val_sel` không vượt biên `min_delta` sau `cfg.patience` epoch liền (như cũ), HOẶC
- `val loss` không giảm sau `cfg.patience` epoch liền (mới — thay cho `loss_floor`).

5-fold CV (muc 8) không còn cắt theo sàn loss; mỗi fold train **đúng `BEST_EPOCH`** mà muc 6
chọn (trần epoch duy nhất). Fold hội tụ nhanh/chậm khác nhau sẽ dừng ở mức fit khác nhau
thật — phương sai đo được giữa các fold vì thế phản ánh đúng dữ liệu nhỏ, không bị che.

### 14.3 · Thứ tự chạy để giữ sạch

Gold (`gold_test.jsonl`) chỉ được nhìn **đúng một lần**, sau khi checkpoint đã lưu:

1. Chạy mục 1–2 (setup, data, split, class weights) — không đụng gold.
2. Chạy mục 6 (train + chọn `BEST_EPOCH` bằng val loss/val_sel) rồi mục 8 (5-fold CV trên
   silver) — không đụng gold. Đọc kết quả, điều chỉnh `lr`/`epochs` nếu cần, lặp lại **chỉ
   trên val/CV**, không mở mục 10 trong lúc này.
3. Lưu checkpoint (muc 11) — khoá candidate tại đây, trước khi nhìn gold.
4. Chạy mục 10 (đánh giá gold) **đúng một lần**. Số ra là số cuối, không quay lại bước 2 dù
   kết quả thế nào.
5. Bỏ qua mục 3.b, 4.b, mục 13 (DEV/HOLD, `weight_target=gold_dev`), và phần so sánh
   mDeBERTa nếu chúng đọc gold trước bước 3 — chỉ dùng lại sau khi đã có candidate khoá,
   cho một vòng đánh giá bổ sung riêng, không phải để chọn lại candidate.

`weight_target` giữ nguyên mặc định `"uniform"` cho lần chạy này — không đặt `"gold_dev"`.

### 14.4 · Bug có sẵn, không liên quan tới mục này

Cell so sánh mDeBERTa (sau muc 11) có 2 chỗ `print(f"` bị xuống dòng thật ngay sau dấu
ngoặc kép mở — `SyntaxError: unterminated f-string literal` nếu chạy bằng CPython chuẩn.
Lỗi có sẵn từ bản vá 13.7 (chưa từng chạy), không phải do sửa ở đây. Không chặn bước 1–4 ở
14.3 vì cell đó nằm sau điểm khoá candidate; sửa khi nào dùng tới đoạn so sánh backbone.

---

## 15 · Chọn backbone cho workflow — mDeBERTa-v3-base-xnli *(2026-09-01)*

Sau khi so sánh trực tiếp trong notebook (mục 4.b zero-shot và mục 9.b fine-tuned), mDeBERTa
vượt xlm-r-base ở cả test-EN lẫn gold. Quyết định: **mDeBERTa-v3-base-xnli là backbone đưa
vào workflow**, không phải xlm-r-base.

Vì lý do đó, **5-fold CV phải đo lại với backbone này** trước khi khoá checkpoint. Cell mới
đã thêm vào notebook ngay sau mục 9.b (cell 44 trong file hiện tại), dùng `cfg_deb` và cùng
logic hoàn toàn với mục 8 (cùng folds, cùng `loss_floor`, cùng lr schedule, cùng
`sched_epochs`) để số so được trực tiếp.

### 15.1 · Thứ tự chạy

Quy trình 14.3 vẫn giữ nguyên, chỉ thay backbone:

1. Mục 1–3 (setup, data, split) — không thay đổi.
2. Mục 6 + **9.b** (train mDeBERTa trên train_rows, chọn `BEST_EPOCH_DEB` bằng val) —
   **không đụng gold**.
3. Mục 8 (CV xlm-r, chỉ tham chiếu) + **cell 44 (CV mDeBERTa)** — đọc kết quả silver, điều
   chỉnh nếu cần, **không mở mục 10**.
4. Mục 12 (lưu checkpoint): lưu **`model_deb`** thay vì `model`.
5. Mục 10 (gold): đúng một lần, sau khi checkpoint đã khoá.

### 15.2 · Giữ nguyên hyperparameter

`cfg_deb = replace(cfg, model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")`
— không thay đổi lr/epochs/patience/loss_floor/symmetric. So sánh kiểm soát một biến: nếu
5-fold CV với mDeBERTa thắng xlm-r, lợi thế đến từ backbone, không phải từ tuning khác.

---

## 16 · Gộp gold DEV + HOLD-OUT thành một tập duy nhất *(2026-09-01)*

Gold từ đây dùng **toàn bộ 129 cặp** làm tập báo cáo, không chia cohort nữa.

### 16.1 · Lý do

Hold-out cohort (`case-03-admissions-rag-chatbot`, 39 cặp) được tạo để tránh lặp chọn
tham số trên gold. Với quyết định của §15 — **không tuning bất kỳ tham số nào dựa vào gold**
— hold-out không còn vai trò. Gộp lại để có 129 cặp thay vì 90 khi đánh giá, giảm phương
sai ước lượng.

### 16.2 · Tham số từng đọc trên gold (cần xử lý thủ công)

Một tham số trong notebook **vẫn** dùng kết quả gold để quyết định:

- **Chọn checkpoint (mục 12 — lưu model)**: code hiện tại so sánh `REPORT["model:gold"]`
  và `REPORT["model_deb:gold"]` để chọn xlm-r hay mDeBERTa. Đây là quyết định sau khi
  nhìn vào gold. Nếu muốn giữ sạch hoàn toàn, hãy thay bằng so sánh `CV-silver` hoặc
  chọn thủ công trước khi chạy mục 12.

### 16.3 · Các thay đổi trong notebook

| Cell | Thay đổi |
|---|---|
| 9 (markdown §3.b) | Xóa ngôn ngữ DEV/HOLD |
| 10 (load gold) | Xóa `gold_dev` / `gold_hold` / `GOLD_DEV_PRIOR` |
| 43 (§9.b mDeBERTa eval) | `store="model_deb:gold"` thay vì dev/hold riêng |
| 44 (CV mDeBERTa) | Sửa syntax bugs (literal newline trong f-string) |
| 46 (§10 gold eval) | Toàn bộ 129 cặp; xóa `gold_rows = gold_dev`; xóa `gold_rows_full` |
| 48 (§11 bảng) | `_TABS` chỉ còn `("gold","gold")`; xóa dòng khai kèm hold-out |
| 50 (§12 lưu) | `REPORT[...:gold]` thay vì `...:gold-dev` |
