# Giai đoạn Train — Relation Classifier

> Fine-tune bộ phân loại quan hệ 6 lớp giữa hai atomic claim của hai reviewer.
> Đầu vào: [`phase2_trackb/processed/trackB_silver.jsonl`](../phase2_trackb/processed/trackB_silver.jsonl) (1099 cặp).
> Kế hoạch tổng: [`PLAN.md`](PLAN.md) · Nguồn few-shot: [`FEWSHOT.md`](FEWSHOT.md)

> 🔴 **Đọc trước khi bấm train** *(cập nhật 2026-08-28)* — có hai thứ đang hỏng, chưa sửa:
> 1. **Split chưa stratified theo nhãn.** Val/test lệch tới 2.5×, 5-fold lệch 13× ở
>    CONTRADICTION. Chạy train ngay bây giờ sẽ ra macro-F1 không so sánh được giữa các fold.
>    → [mục 2](#2--chia-dữ-liệu)
> 2. **CONTRADICTION chỉ 29 mẫu, test có 2.** Đã đo hết headroom và bốn hướng xử lý.
>    → [mục 8](#8--tăng-mẫu-contradiction-phân-tích-2026-08-28-chưa-thực-thi)
>
> Việc cụ thể để làm khi quay lại: [mục 8.5](#85--việc-cụ-thể-khi-quay-lại).

---

## 0 · Vì sao giai đoạn này tồn tại ở dạng hiện tại

Pipeline gốc (`notebooks/track_b_pipeline.ipynb`, các bước B4→B6) dùng **ensemble 3 model +
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

Ghi ra `phase2_trackb/processed/splits/`:

```
trackB_train.jsonl   879 cặp / 180 paper
trackB_val.jsonl     110 cặp /  21 paper
trackB_test.jsonl    110 cặp /  21 paper
folds.json           gán fold 0-4 cho từng pair_id (dùng cho CV)
class_weights.json   trọng số inverse-frequency tính trên train
```

Script tự `assert` không paper nào nằm ở hai phần, và **báo cáo rò rỉ ở mức text của claim**
(chặt hơn mức paper). Hiện tại: `train∩val = 3`, `train∩test = 1` claim — đến từ cặp
UNRELATED chéo paper, vì `group_of()` chỉ lấy paper bên trái.

<details>
<summary><b>Vì sao không dùng union-find gộp cả hai paper của cặp chéo</b></summary>

110 cạnh chéo trên 263 paper sẽ nối phần lớn paper thành **một thành phần liên thông
khổng lồ**, khiến việc nhóm mất hoàn toàn tác dụng (mọi thứ rơi vào cùng một fold).
Lấy paper trái là đánh đổi có ý thức; phần rủi ro còn lại được đo trực tiếp và báo cáo
thay vì giấu đi.
</details>

### ⚠ CHƯA stratified theo nhãn — cần sửa trước khi train *(ghi nhận 2026-08-28)*

`assign_groups()` ([`split.py:66-67`](../src/train/split.py#L66-L67)) chỉ cân bằng **tổng số cặp**
so với hạn ngạch, không hề đọc `relation`. Docstring ở [`split.py:49`](../src/train/split.py#L49)
viết *"ưu tiên giữ cân bằng nhãn"* — **sai, code không làm điều đó.**

Hệ quả đo được trên các file split hiện tại:

| Nhãn | toàn bộ | train | val | **test** |
|---|---:|---:|---:|---:|
| AGREEMENT | 6.9% | 65 (7.4%) | 8 (7.3%) | **3 (2.7%)** ← lệch 2.5× |
| PARTIAL_AGREEMENT | 19.7% | 184 (20.9%) | 21 (19.1%) | **12 (10.9%)** ← lệch 1.8× |
| COMPLEMENTARY | 45.1% | 388 (44.1%) | 50 (45.5%) | 58 (52.7%) |
| PARTIAL_CONTRADICTION | 15.3% | 127 (14.4%) | 16 (14.5%) | 25 (22.7%) |
| CONTRADICTION | 2.6% | 23 (2.6%) | 4 (3.6%) | **2 (1.8%)** |
| UNRELATED | 10.3% | 92 (10.5%) | 11 (10.0%) | 10 (9.1%) |

Train khá sát phân phối gốc (lệch < 1 điểm %) nên `class_weights.json` vẫn dùng được.
**Val và test thì không.**

5-fold còn tệ hơn — CONTRADICTION dao động **13 mẫu (f0, 5.9%) xuống 1 mẫu (f2, 0.5%)**,
chênh 13 lần. Macro-F1 giữa các fold sẽ nhiễu tới mức không so sánh được với nhau.

**Nguyên nhân:** nhãn tương quan mạnh trong cùng paper (một paper thường sinh cụm cặp cùng
loại), nên greedy theo size để nhãn hiếm dồn cục.

**Cách sửa:** đổi tiêu chí chọn part từ "thiếu tổng" sang "thiếu theo từng nhãn" — greedy
stratified-group, cùng ý tưởng `StratifiedGroupKFold` của sklearn: với mỗi nhóm paper, thử bỏ
vào từng part, chọn part làm **tổng độ lệch nhãn** nhỏ nhất so với hạn ngạch từng lớp. Giữ
nguyên `group_of()` nên không phát sinh rò rỉ paper. **Chưa làm.**

### ⚠ Test có đúng 2 mẫu CONTRADICTION

F1 của lớp đó trên test chỉ có thể là 0, 0.5 hoặc 1.0 — **con số ngẫu nhiên, không đọc được**.
Vì vậy **luôn đọc kết quả từ 5-fold CV**, test chỉ dùng cho lần chốt checkpoint cuối.

Stratified hoá cũng chỉ nâng được lên ~3 mẫu. Đây là giới hạn vật lý của 29 mẫu / 1099 cặp —
muốn test đọc được thì phải **tăng số mẫu CONTRADICTION**, xem mục 7.

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
https://colab.research.google.com/github/navihat/build-phase2-finetune/blob/main/notebooks/train_relation_classifier.ipynb
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
| ✅ | 1099 cặp, cả 6 lớp đều có mẫu |
| ✅ | Chia theo paper + 5-fold, có kiểm tra rò rỉ **paper** |
| ✅ | Script train + notebook Colab, 7 phép đánh giá |
| 🔴 | **Split chưa stratified theo nhãn** — val/test lệch tới 2.5×, fold lệch 13× ở CONTRADICTION (mục 2) |
| 🔴 | **CONTRADICTION 29 mẫu là nút thắt** — test chỉ 2 mẫu, F1 lớp đó không đọc được (mục 8) |
| ⚠ | **Script train mới chỉ kiểm cú pháp, chưa chạy end-to-end** (máy chưa có torch) |
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
| ⬜ | Sửa `assign_groups()` → stratified-group, chạy lại `split.py` (mục 2) |
| ⬜ | Viết `src/data/mine_contradiction.py` — xếp hạng ứng viên bằng NLI, xuất JSONL để gán nhãn |
| ⬜ | Gán nhãn 300–400 ứng viên top, ghi `source: mined_nli_contradiction` |
| ⬜ | Thêm swap-aug chọn lọc theo lớp vào `train.py` |
| ⬜ | Nếu sau đó CONTRADICTION vẫn < 60 mẫu → chốt phương án ④, sửa `LABELS` ở cả split và train |
