# Phase 2 — Checklist thực thi Track B

> **Dùng file này để tự kiểm soát tiến độ khi chạy Colab.** Nó độc lập: mọi ngưỡng, mọi điều kiện dừng, mọi việc phải làm đều nằm ở đây, không cần hỏi lại ai.
>
> Notebook: [`track_b_pipeline.ipynb`](../notebooks/track_b_pipeline.ipynb) · Kế hoạch đầy đủ: [`docs/PLAN.md`](PLAN.md)
>
> **Cách dùng:** tick `[x]` khi xong. Gặp dòng ⛔ thì **dừng lại thật**, đừng chạy tiếp cho xong.

---

## 0 · Trước khi mở Colab

- [ ] Upload dữ liệu raw (`.json` chứa key `Review_*_full`) vào `MyDrive/phase2_trackb/raw/`
- [ ] Xác nhận Colab đang có **A100 40GB** (`Runtime > Change runtime type`)
- [ ] Ghi số units còn lại lúc bắt đầu: `________ / 99.57`

### Ba quyết định phải chốt trước — đã có mặc định, nhưng phải biết mình đang chọn gì

| # | Quyết định | Mặc định trong notebook | Đổi bằng cách |
|---|---|---|---|
| D1 | **Nguồn few-shot** | Không có file → chạy **zero-shot** | Đặt `fewshot.jsonl` vào `processed/` |
| D2 | **Soft-label sau debate** | `soft_label` = phân bố trên 6 quan sát; `relation` = phán quyết Judge | Sửa cell B6 |
| D3 | **Judge model** | `Qwen3-14B-AWQ` (chưa nâng cấp) | Sửa `MODELS["judge"]` sau khi qua pilot Q5 |

**D1 đáng cân nhắc nhất.** Không có few-shot, 3 model tự đặt ngưỡng ranh giới theo prior riêng → "đồng thuận 3/3" ở B5 mất ý nghĩa hiệu chuẩn, tỉ lệ debate nhiều khả năng tăng (đội chi phí ở đúng bước đắt nhất). Chi phí để có: **~1h người, 0 GPU**.

- [ ] **D1 đã chốt:** ☐ tự gán 40 cặp ☐ chấp nhận zero-shot (đã hiểu rủi ro)
- [ ] **D2 đã đọc và duyệt** — hệ quả: cặp `route=debate` có thể có `relation ≠ argmax(soft_label)`. Đây là chủ đích: nhãn cứng = kết quả phân xử, nhãn mềm = độ bất định thô của ensemble.
- [ ] **D3 để nguyên** cho tới khi có kết quả pilot Q5

<details>
<summary><b>Nếu chọn tự gán 40 cặp — format và cách phân bổ</b></summary>

Đặt tại `processed/fewshot.jsonl`, mỗi dòng:
```json
{"left": "...", "right": "...", "label": "PARTIAL_CONTRADICTION", "why": "một câu ngắn"}
```

Phân bổ ~6–7 cặp/nhãn, **ưu tiên phủ 5 ranh giới liền kề** (đây mới là chỗ model hay sai, không phải các ca dễ):

```
AGREEMENT — PARTIAL_AGREEMENT — COMPLEMENTARY — PARTIAL_CONTRADICTION — CONTRADICTION
                                     ⊥
                                 UNRELATED
```

Nguyên liệu: lấy câu thật từ corpus (chạy B1+B2+B3 ở `PILOT_MODE` trước, rồi bốc cặp từ `interim/pairs.jsonl`).

⚠️ Cặp đã dùng làm few-shot **tự động bị loại khỏi tập train** ở B3 (R-1) — notebook làm việc này, không cần tự lọc.
</details>

---

## 1 · Setup (cell 0.1 → 0.5)

- [ ] `0.1` cài đặt xong (Colab có thể yêu cầu restart → restart rồi chạy tiếp từ `0.2`)
- [ ] `0.2` Drive mount, thấy `PILOT_MODE = True`
- [ ] `0.3` in ra `OK. Gọi budget_report()...`
- [ ] `0.4` model loader nạp được
- [ ] `0.5` rubric nạp — **ghi lại**: few-shot ☐ có (`___` ví dụ) ☐ không (zero-shot)

---

## 2 · B1 · Tách câu

- [ ] Chạy cell kiểm tra raw — thấy đúng file mình upload
- [ ] Chạy B1

**Ghi số liệu:**

| Chỉ số | Giá trị |
|---|---|
| Số paper | |
| Review duy nhất (sau dedup) | |
| Tổng số câu | |
| **Trung bình câu/review** | ← **pilot Q2** |
| Review bị cắt hậu-rebuttal | |

⚠️ **Số review duy nhất phải nhỏ hơn nhiều so với số entry trong file raw.** Trong file IMPACT, một paper xuất hiện ở nhiều pair-entry, và tên trường `Review_1_full`/`Review_2_full` là **vị trí, không phải danh tính** — notebook giải mã số review thật từ pair key rồi mới dedup. Nếu con số này bằng đúng số entry × 2 thì việc giải mã đã hỏng, dừng lại kiểm tra.

⚠️ Số review bị cắt hậu-rebuttal = 0 trên corpus lớn là **đáng ngờ** — kiểm tra `REBUTTAL_MARKERS` có khớp văn phong của corpus không. Bỏ lọt bước này sẽ ghép claim **đã bị chính người viết rút lại** với claim của reviewer kia → nhãn sai không cách nào phát hiện được ở hạ nguồn.

---

## 3 · B2 · Trích claim + guard-rail

- [ ] Chạy B2 (Qwen3-14B, JSON-schema-constrained)
- [ ] Chạy B2-guard (lexical → NLI entailment)

**Ghi số liệu:**

| Chỉ số | Giá trị | Ngưỡng |
|---|---|---|
| Claim thô | | |
| **Tỉ lệ qua guard-rail** | `___%` | ← **pilot Q3** |
| Loại ở tầng lexical | | |
| Loại vì `nli_neutral` | | |
| Loại vì `nli_contradiction` | | ⚠️ xem dưới |

**Cách đọc:**

| Quan sát | Nghĩa là gì | Làm gì |
|---|---|---|
| Qua guard-rail **< 50%** | Prompt B2 đang cho model diễn giải quá tay (thường vậy), không phải NLI sai | Đọc `reports/b2_rejected.jsonl`, siết prompt B2 |
| `nli_contradiction` **> 5%** | Model đang **đảo cực tính** khi trích claim | Nghiêm trọng — siết dòng "TUYỆT ĐỐI KHÔNG đảo cực tính" trong `B2_SYSTEM` |
| `NEGATIVE` chiếm tỉ trọng lớn trong stance | Prompt sai thứ tự ưu tiên stance | `RECOMMENDATION` phải thắng `CONCERN`; `NEGATIVE` gần như không xuất hiện ở đường rule production (plan §11 rủi ro #6) |

- [ ] Đã mở `reports/b2_rejected.jsonl` đọc thử **ít nhất 10 ca bị loại** — xác nhận chúng đáng bị loại thật, không phải guard-rail quá gắt

---

## 4 · B3 · Ghép cặp

- [ ] Chạy B3

| Chỉ số | Giá trị |
|---|---|
| Cặp ứng viên sinh ra | |
| Cặp sau lấy mẫu | |
| Phân bố theo aspect | |

⚠️ Một aspect chiếm **> 40%** tổng số cặp → cân bằng theo aspect đã hỏng, kiểm tra lại logic lấy mẫu.

---

## 5 · ⛔ PILOT GATE

**Chạy hết B4 → B5 → Debate → B6 ở `PILOT_MODE = True` trước.** Sau đó điền bảng dưới đây. **Chưa đủ 5 câu trả lời thì không được đặt `PILOT_MODE = False`.**

Chạy full mà chưa qua gate = đặt cược toàn bộ ngân sách vào ước lượng chưa ai kiểm, trong khi không có API dự phòng để chạy lại.

| # | Câu hỏi | Trả lời | Quyết định kéo theo |
|---|---|---|---|
| Q1 | Throughput thực tế (units cho 100 cặp) | | Nhân lên `×20` ra ước tính full 2.000 cặp. Vượt ngân sách → giảm `TARGET_PAIRS` |
| Q2 | Trung bình câu/review | | B2 chạy trên **toàn bộ câu thô** — khối lượng này **không** co giãn khi giảm số cặp đích |
| Q3 | Tỉ lệ claim qua guard-rail | `___%` | < 50% → sửa prompt B2 trước khi chạy full |
| Q4 | n=3 khác n=5 bao nhiêu % | `___%` | **< 5%** → hạ `N_SELF_CONSISTENCY = 3`, tiết kiệm 40% lượt sinh B4. **≥ 5%** → giữ n=5 |
| Q5 | Judge 32B vs 14B | | Chỉ nâng nếu VRAM vừa **và** dự phòng sau khi trừ vẫn **≥ 37 units** |

- [ ] Đủ 5 câu trả lời
- [ ] `budget_report()` — units đã tiêu cho pilot: `______`
- [ ] Ước tính units cho full run: `______` (Q1 × 20 + train 17.7)
- [ ] **Ước tính đó ≤ units còn lại − 37** ← nếu không, giảm `TARGET_PAIRS` rồi tính lại

⛔ **Điều kiện dừng — không thương lượng:**
- Units còn lại **< 37** → dừng, không chạy full
- Q3 < 30% → B2 hỏng, sửa prompt trước, đừng chạy tiếp
- Full run ước tính vượt units còn lại → **giảm `TARGET_PAIRS`**, không phải "chạy thử xem sao"

---

## 6 · Full run

- [ ] Đặt `PILOT_MODE = False`
- [ ] Áp dụng kết quả Q4 (`N_SELF_CONSISTENCY`) và Q5 (`MODELS["judge"]`) nếu có đổi
- [ ] **Xoá các file interim của pilot** để không bị `[SKIP]` nhầm sang dữ liệu pilot:
  ```python
  for f in ["claims_raw","claims","pairs","routed","labels_qwen","labels_gemma",
            "labels_seallm","debate_results"]:
      (DIRS["interim"]/f"{f}.jsonl").unlink(missing_ok=True)
  ```
  ⚠️ **Giữ lại `sentences.jsonl`** — B1 không phụ thuộc `PILOT_MODE`, chạy lại chỉ tốn thời gian vô ích.

- [ ] B2 + guard-rail
- [ ] B3
- [ ] B4 — Qwen  ·  ⚠️ chạy lâu, đừng đóng tab
- [ ] B4 — Gemma
- [ ] B4 — SeaLLM
- [ ] B5 routing
- [ ] Debate
- [ ] B6 → `trackB_silver.jsonl`

**Sau B5 — ghi lại:**

| Route | Số cặp | % |
|---|---|---|
| unanimous | | |
| majority | | |
| debate | | |
| dropped_far | | |

⚠️ **Debate > 35%** → rubric chưa đủ sắc (hoặc đang zero-shot). Sửa rubric rẻ hơn nhiều so với debate 400+ cặp — *"Một giờ đầu tư vào rubric tiết kiệm nhiều giờ compute"* (plan §6).

⚠️ **`dropped_far` > 25%** → nhiều cặp ứng viên rác, xem lại B3.

**Sau B6 — ghi lại:**

| Chỉ số | Giá trị |
|---|---|
| Tổng cặp trong silver | |
| Phân bố `relation` | |
| Nhãn **không xuất hiện lần nào** | ← ⚠️ |
| ABSTAIN bị loại | |

⚠️ **Nhãn nào không có mẫu nào thì model không học được lớp đó.** Nếu `AGREEMENT`/`UNRELATED` vắng mặt → rubric hoặc B3 đang không sinh ra loại cặp này. Phải xử lý trước khi train, không train rồi mới phát hiện.

---

## 7 · Spot-check (thay cho bridge đã bỏ)

Đây là **kiểm chất lượng duy nhất còn lại**. Bridge/κ đã bị bỏ khỏi phạm vi → không có số đo độc lập nào khác.

- [ ] Chạy cell spot-check → tải `reports/spotcheck_sample.csv`
- [ ] Đọc tay **cả 50 dòng**, điền cột `BẠN_ĐỒNG_Ý?` (y/n); nếu `n` thì ghi nhãn đúng
- [ ] Ghi kết quả vào `reports/spotcheck.md`

**Tìm MẪU LỖI HỆ THỐNG, không phải tỉ lệ %:**

| Mẫu quan sát được | Nguyên nhân | Hành động |
|---|---|---|
| Một ranh giới lệch **đều một hướng** (VD `PARTIAL_CONTRADICTION` → `CONTRADICTION`) | Rubric mờ ở đúng ranh giới đó | Siết `RANH GIỚI KHÓ` trong `RUBRIC`, **chạy lại B4** |
| Claim trích hỏng lọt guard-rail | Prompt B2 / ngưỡng NLI | Sửa B2, chạy lại từ B2 |
| Một aspect toàn nhãn rác | B3 ghép cặp sai cho aspect đó | Xem lại B3 |
| `route=debate` sai nhiều hơn hẳn `unanimous` | Rubric hẹp đưa Judge chưa đủ sắc | Sửa `narrow_rubric()` |

- [ ] Tỉ lệ đồng ý: `___%`
- [ ] **Không** thấy mẫu lỗi hệ thống nào → đi tiếp
- [ ] Có mẫu lỗi hệ thống → sửa rubric, **chạy lại B4** (dự phòng ~61 units đủ cho **một** lần chạy lại), rồi spot-check lại

⚠️ **R-2:** sửa rubric theo **mẫu lỗi hệ thống** là đúng mục đích. Sửa rubric để ép đúng từng cặp cụ thể vừa đọc thì **không** — đó là tune trên chính tập dùng để kiểm, và giờ không còn κ độc lập nào phát hiện được việc đó.

⚠️ **Spot-check không phải κ và không được trình bày như thể là.** Cỡ mẫu ~50 không đủ tính κ; không có gold độc lập; người chấm cũng là người viết rubric nên thiên lệch xác nhận là có thật và không khử được. Nó bắt lỗi **thô và hệ thống**, không chứng minh silver set đúng.

---

## 8 · Dịch + test giữ nhãn

- [ ] Chạy cell dịch (60% EN / 40% VI)
- [ ] Chạy test giữ nhãn

| Chỉ số | Giá trị | Ngưỡng |
|---|---|---|
| Lật nhãn tổng | `___%` | |
| **Lật nhãn nhóm `PARTIAL_*`** | `___%` | **> 15% → leo thang** |

Trục `PARTIAL_*` hoàn toàn là **trục hedging**. MT chuyên dụng có xu hướng chuẩn hoá ngôn ngữ rào đón cho gọn, làm `PARTIAL_CONTRADICTION` trượt thành `CONTRADICTION`.

- [ ] Dưới 15% → giữ VinAI-Translate
- [ ] Trên 15% → chuyển riêng nhóm câu có hedging sang dịch bằng LLM local (SeaLLM/Qwen3, prompt giữ tình thái). Chỉ là tập con nên chi phí thấp.

---

## 9 · Manifest — 🔴 Hiếu đang bị chặn

- [ ] Chạy cell manifest
- [ ] **Gửi `train_validation_manifest.json` cho Hiếu**

Field `train_validation_hash_source` của Hiếu **chỉ có thể đến từ file này**. Gửi ngay khi có file train đầu tiên — **kể cả bản pilot** — đừng đợi chạy xong hết. Hiếu không tạo được gold set khi chưa có nó.

- [ ] Xác nhận trong manifest có đủ mục `known_limitations`, đặc biệt:
  - "Không có bridge/κ: dataset chưa từng được đối chiếu với nhãn người trên tập kiểm độc lập"
  - "Toàn bộ nhãn do LLM sinh"
  - (nếu zero-shot) "Không có ví dụ few-shot nào neo hiệu chuẩn 3 model"

---

## 10 · Bàn giao

- [ ] Tải về từ Drive: `train_en.jsonl`, `train_vi.jsonl`, `trackB_silver.jsonl`, `train_validation_manifest.json`
- [ ] Tải `reports/` (b2_guard, spotcheck, translation_fidelity) và `configs/` (seeds, budget_log)
- [ ] Commit vào repo — **xem ghi chú scope dưới**
- [ ] Cập nhật `WORKLOG.md`
- [ ] Ghi units cuối cùng còn lại: `______`

### ⚠️ Scope guard

`phase-2/` nằm **ngoài** allowlist `experiments/relation_classifier/**` của MODEL-001. CI sẽ chặn nếu PR target `feat/gtq-upgrade`.

- [ ] PR **không** target `feat/gtq-upgrade` → commit thẳng vào `phase-2/`
- [ ] PR **có** target `feat/gtq-upgrade` → mirror sang `experiments/relation_classifier/`, giữ PR chỉ chứa vùng đó cộng một dòng `WORKLOG.md`

---

## Bảng theo dõi ngân sách

Cập nhật sau mỗi phiên. `budget_report()` in ra tự động.

| Ngày | Việc | Units tiêu | Units còn | Ghi chú |
|---|---|---|---|---|
| | Pilot | | | |
| | B2 full | | | |
| | B4 Qwen | | | |
| | B4 Gemma | | | |
| | B4 SeaLLM | | | |
| | Debate | | | |
| | Dịch | | | |
| | Train | | | |

🔴 **Sàn an toàn: 37 units.** Chạm sàn → dừng, đánh giá lại, không chạy tiếp theo quán tính.

---

## Khi có sự cố

| Triệu chứng | Xử lý |
|---|---|
| Colab ngắt giữa chừng | Mọi stage đã checkpoint xuống Drive. Chạy lại từ cell 0.2, các stage xong sẽ `[SKIP]`, không tính tiền lại. |
| OOM khi nạp model | `free_model()`. Còn OOM → `Runtime > Restart`, chạy lại từ 0.2. Checkpoint an toàn trên Drive. |
| VRAM không giải phóng sau `free_model()` | API teardown của vLLM đổi theo version. Cách chắc chắn: `Runtime > Restart` rồi chạy tiếp stage kế. |
| Muốn chạy lại một stage | `force=True` trong `stage(...)`, hoặc xoá file output của nó. |
| Kết quả một stage trông sai | **Đừng chạy tiếp.** Chạy tiếp trên đầu vào hỏng chỉ nhân rộng lỗi và đốt units. |
