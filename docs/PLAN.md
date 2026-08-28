# Phase 2 — Xây dựng dữ liệu fine-tune cho Relation Classifier

> **Owner:** Trương Văn Thái
> **Task nguồn:** [MODEL-001](../tasks/MODEL-001-finetuned-relation-classifier.md)
> **Contract ràng buộc:** [`specs/relation-classifier-contract.md`](../specs/relation-classifier-contract.md) — 6 nhãn, batch semantics, fail-closed
> **Trạng thái:** Đang thực hiện

---

## 1. Mục tiêu

Sinh dữ liệu huấn luyện cho relation classifier 6 nhãn, từ corpus peer review tiếng Anh.

**Chỉ một luồng: Track B.**

| Track | Nguồn | Sản lượng | Vai trò |
|---|---|---|---|
| **B** | toàn văn `Review_*_full` | ~1.000–2.000 cặp silver | toàn bộ dữ liệu huấn luyện, phủ cả 6 nhãn |

### Track A và Bridge đã bị loại khỏi phạm vi — hệ quả phải khai

Quyết định (do ràng buộc thời gian): bỏ Track A và bỏ Bước 3 bridge scoring. Ba hệ quả **bắt buộc khai trong model card**, không được im lặng bỏ qua:

1. **Không còn cổng chất lượng nào trước khi train.** Không có κ, không có confusion matrix, không có bất kỳ số đo nào cho biết silver set đáng tin tới đâu trước khi đổ 1.5h/17.7 units vào train. Cổng quyết định κ ≥ 0.70 / 0.50–0.70 / < 0.50 trước đây không còn tồn tại — pipeline chạy thẳng từ silver sang train.
2. **Không còn nhãn người ở bất kỳ đâu trong pipeline.** Toàn bộ nhãn đều do LLM sinh; chất lượng phụ thuộc hoàn toàn vào rubric + thiết kế ensemble/debate, không có neo ngoài nào kiểm chứng.
3. **Nguồn few-shot mất chỗ dựa** — xem quyết định bắt buộc ngay dưới đây.

### Quyết định còn treo: nguồn 40 cặp few-shot ⚠️

`trackA_fewshot.jsonl` trước đây được dựng ở Bước 0 từ nhãn người. Nó được dùng ở **hai chỗ** trong Track B: prompt gán nhãn ensemble (B4) và prompt Judge lúc debate (§6). Không có nó, 3 model tự đặt ngưỡng ranh giới theo prior riêng của từng model — khi đó "đồng thuận 3/3" ở B5 chỉ phản ánh 3 model tình cờ trùng prior, không phải cùng áp một chuẩn; đồng thời tỉ lệ bất đồng nhiều khả năng tăng, đẩy thêm cặp vào debate (bước đắt nhất).

| Phương án | Chi phí | Đánh đổi |
|---|---|---|
| **(a) Tự gán nhãn ~40 cặp** *(khuyến nghị)* | ~1h người, **0 GPU** | Giữ được neo hiệu chuẩn duy nhất còn lại. Nguyên liệu lấy từ chính câu văn trong corpus; chỉ cần phủ đủ 5 ranh giới liền kề trên trục quan hệ, ~6–7 cặp/nhãn. |
| (b) Chạy zero-shot, không few-shot | 0 | Rẻ nhất nhưng bỏ luôn neo cuối cùng — cộng dồn với việc đã mất bridge, pipeline không còn bất kỳ điểm tựa người nào. Rủi ro cao nhất, phải khai rõ trong model card. |

Chưa chốt phương án nào thì **chưa viết được prompt B4 và prompt Judge** — đây là blocker thực thi, không phải chi tiết trang trí.

---

## 2. Tài nguyên và ngân sách

**99.57 compute units trên Google Colab Pro. Không gọi API ngoài.** Mọi model chạy local.

Quy đổi: A100 40GB ≈ 11.8 u/h → **~8.4 giờ tổng**.

| Hạng mục | Giờ | Units |
|---|---|---|
| Pilot đo throughput (100 cặp) | 0.2 | 2.4 |
| Trích claim + stance + aspect (B2 — phụ thuộc khối lượng câu thô, **không** co giãn theo mục tiêu cặp cuối, xem §5 B2) | 0.5 | 5.9 |
| Ensemble gán nhãn ~2.000 cặp | 0.7 | 8.5 |
| Debate ~400 cặp (20% × 2.000 — dưới cap 1.200, cap không còn là ràng buộc thực tế ở quy mô này) | 0.3 | 3.3 |
| Dịch (VinAI-Translate) | 0.04 | 0.5 |
| Train + sweep + 3 seed | 1.5 | 17.7 |
| **Tổng** | **3.2** | **38.3** |
| **Dự phòng** | | **~61** |

> Track A (1.2 + 2.4) và bridge scoring (2.4) đã bị loại → giải phóng thêm 6 units. Dự phòng dư dả **không** phải lý do để nới tay: giờ không còn cổng chất lượng nào, nên phần dư này là bảo hiểm cho việc phải chạy lại B4 nếu phát hiện silver set hỏng bằng spot-check thủ công.

> **Bắt buộc chạy pilot 100 cặp trước khi cam kết ngân sách.** Mọi con số trên là ước lượng. Không có API dự phòng, đoán sai 2× là mất khả năng rerun.
> Phần ~18 units mới giải phóng (do giảm mục tiêu Track B) là **surplus thật**, không phải sàn an toàn — sàn an toàn gốc ~37 units vẫn phải giữ nguyên bất kể quyết định nâng cấp model nào bên dưới.

**Pilot phải trả lời đủ 5 câu hỏi này** (không chỉ đo throughput):

| # | Câu hỏi | Quyết định phụ thuộc |
|---|---|---|
| 1 | Throughput thực tế mỗi model (lượt/giờ) | toàn bộ bảng ngân sách trên |
| 2 | Trung bình bao nhiêu câu/review, tỉ lệ bị lọc ở B2 | con số 5.9 units của B2 (§5 B2) — khối lượng này **không** co giãn theo mục tiêu số cặp |
| 3 | Tỉ lệ claim bị loại ở guard-rail B2, tách riêng tầng lexical và tầng entailment | chất lượng prompt B2; tỉ lệ loại cao bất thường = prompt cần chỉnh |
| 4 | Nhãn đa số n=3 lệch nhãn đa số n=5 bao nhiêu %? (tính ngay trên 5 mẫu đã sinh) | có hạ n=5 → n=3 ở B4 không (§5 B4) |
| 5 | Judge ~32B AWQ so với Qwen3-14B: phán quyết khác nhau ra sao, throughput thế nào | có nâng cấp Judge không (xem ngay dưới) |

### Model đã chốt

| Vai trò | Model | Ghi chú |
|---|---|---|
| Labeler 1 (+ chạy B2, xem §5) | `Qwen3-14B-AWQ` | mạnh nhất trong 3 labeler — dùng thêm cho B2 vì bước đó không có ensemble để tự sửa sai |
| Labeler 2 | `Gemma-2-9B-it` | fp16 |
| Labeler 3 | `SeaLLMs-v3-7B-Chat` | fp16, mạnh tiếng Việt/Đông Nam Á |
| **Judge** | `Qwen3-32B-AWQ` (hoặc tương đương ~32B quantized) — **nâng cấp riêng, xem quyết định dưới**, gate bằng pilot | chỉ chạy trên cặp debate (~400/2.000) — mỗi cặp 1 phán quyết, không phải 30 lượt như labeler nên chi phí tuyệt đối thấp dù model to |
| Dịch EN→VI | `vinai/vinai-translate-en2vi-v2` | 600M, rẻ |
| Backbone đích | `mDeBERTa-v3-base-xnli` | cross-encoder + head 6 lớp |

Model **chạy tuần tự**, không đồng thời — gán hết ~2.000 cặp bằng model 1, swap, model 2, swap, model 3, rồi swap sang Judge chỉ cho tập cặp debate. Bộ nhớ không phải ràng buộc; mỗi model chỉ cần vừa một mình GPU tại thời điểm nó chạy.

Serving bằng vLLM. **Constrained decoding bắt buộc** cho B4/Judge (`outlines` / `xgrammar`) — output bị ép vào đúng tập nhãn, không parse free text. B2 dùng **JSON-schema-constrained** thay vì enum thuần, vì có trường `claim` là free text (chi tiết ở §5 B2).

### Quyết định về kích thước model (A100 40GB) — vì sao không nâng cả 3 labeler lên >32B

Giảm mục tiêu Track B từ ~5.000 xuống ~1.000–2.000 cặp giải phóng thêm ~18 units dự phòng (xem bảng ngân sách trên). Điều đó **không** có nghĩa nên đổ hết vào việc nâng size model, vì ba lý do:

1. **VRAM**: A100 40GB. Một model 32B ở fp16 cần ~64GB — không vừa, bắt buộc quantize (AWQ 4-bit, ~16-18GB) mới chạy được. Phần lợi thế "tham số nhiều hơn" đã bị bào mòn một phần bởi quantize.
2. **Ngân sách tính theo unit cố định, không phải theo giờ rảnh**: model to hơn xử lý chậm hơn/cặp. Nâng cả 3 labeler tăng chi phí trên **toàn bộ** ~2.000 cặp × 30 lượt — ăn hết phần vừa giải phóng mà lợi ích không chắc tương xứng.
3. **Bài toán không cần model lớn để có lợi ích lớn**: phân loại 6 lớp trên câu ngắn qua constrained decoding không phải suy luận nhiều bước. Thiết kế ensemble + order-swap + debate (§5 B4-B5) đã tồn tại chính để bù đắp cho model tầm trung không hoàn hảo — đây là đòn bẩy hiệu quả hơn việc dùng 1 model to cho tất cả.

**Nơi đáng đầu tư hơn: nâng cấp riêng Judge.** Debate chỉ chạm ~400 cặp (20% × 2.000, dưới cap 1.200 nên cap không còn ràng buộc), và mỗi cặp Judge chỉ ra **1 phán quyết** — không phải 30 lượt như labeler. Đây là quyết định khó nhất, đáng giá nhất để đầu tư năng lực suy luận mạnh hơn, với chi phí tuyệt đối thấp vì số lượt gọi bị chặn trên bởi số cặp debate.

**Điều kiện áp dụng — bắt buộc qua pilot trước khi chốt:**
- Chạy pilot với Judge ứng viên (~32B AWQ) trên cùng tập debate mẫu đã dùng cho Qwen3-14B, so sánh phán quyết và đo throughput thực tế.
- Chỉ chốt nâng cấp nếu VRAM vừa khi load riêng Judge (không cần đồng thời với labeler vì chạy tuần tự), **và** dự phòng sau khi trừ chi phí Judge mới vẫn giữ **≥ 37 units** (sàn an toàn gốc) — tuyệt đối không tiêu hết phần 18 units mới giải phóng.

---

## 3. Thứ tự thực thi

```
Bước 0  Chuẩn bị few-shot (~40 cặp)  → fewshot.jsonl        [xem quyết định treo ở §1]
Bước 1  Pilot 100 cặp                → trả lời 5 câu hỏi §2  [gate ngân sách]
Bước 2  Track B — toàn bộ pipeline   → trackB_silver.jsonl
Bước 3  Spot-check thủ công          → reports/spotcheck.md  [thay cho bridge đã bỏ]
Bước 4  Dịch + hợp nhất              → train_{en,vi}.jsonl
Bước 5  Manifest SHA-256             → train_validation_manifest.json  [Hiếu đang bị chặn, xem §10]
```

**Bước 3 — spot-check thay cho bridge.** Bridge đã bị bỏ, nhưng train mù hoàn toàn trên ~2.000 cặp là rủi ro không cần thiết khi cách kiểm rẻ nhất chỉ tốn thời gian người, không tốn GPU: rút ngẫu nhiên có seed **~50 cặp** từ `trackB_silver.jsonl` (stratified theo nhãn và theo `route`, để nhìn được cả ca `unanimous` lẫn ca `debate`), đọc tay, ghi tỉ lệ đồng ý vào `reports/spotcheck.md`.

Đây **không** phải κ và không thay thế được bridge — cỡ mẫu nhỏ, không có gold set độc lập, người chấm cũng là người viết rubric. Giá trị của nó là bắt lỗi hệ thống thô: nhãn lệch hàng loạt ở một ranh giới, claim bị trích hỏng lọt qua guard-rail B2, một aspect toàn nhãn rác. Nếu tỉ lệ đồng ý thấp rõ rệt → sửa rubric và chạy lại B4 (dự phòng ~61 units đủ cho một lần chạy lại) thay vì train trên silver set hỏng.

---

## 4. Nguyên tắc bắt buộc

### R-1 — Few-shot không bao giờ nằm trong tập train ~~(cũ: đọc lại Track A trước khi chấm)~~

Bridge đã bị bỏ nên 4 bước xác minh gold cũ không còn áp dụng. Phần **vẫn còn hiệu lực**: các cặp dùng làm few-shot phải bị loại khỏi `trackB_silver.jsonl` theo `pair_id`. Model đã nhìn thấy đáp án của một cặp trong prompt thì cặp đó không còn là dữ liệu học hợp lệ.

### R-2 — Chỉnh rubric trước, không chỉnh theo kết quả spot-check từng cặp

Sửa rubric dựa trên **mẫu lỗi hệ thống** nhìn thấy ở spot-check (VD "ranh giới `PARTIAL_CONTRADICTION`/`CONTRADICTION` bị lệch đều một hướng") là hợp lệ và đúng mục đích. Sửa rubric để ép đúng từng cặp cụ thể đã đọc thì không — đó là tune trên chính tập dùng để kiểm, và giờ không còn κ độc lập nào để phát hiện việc đó nữa.

### R-3 — Ghi lại mọi seed

Seed của: chọn cặp few-shot, phân vai debate, thứ tự transcript trong prompt judge, sampling, rút mẫu spot-check. Ghi vào `configs/seeds.json`. Không tái lập được thì không bảo vệ được kết quả.

### R-4 — Chỉ pairwise

Quan hệ luôn giữa **đúng 2 claims**. Không triplet, không nhóm 3. (Dataset này mọi bài đều đúng 2 review nên ràng buộc khớp sẵn.)

---

## 5. TRACK B — sinh silver từ toàn văn review

### B1. Tách câu

Trên toàn bộ `Review_*_full`. Bắt key bằng regex `Review_\d+_full` — trong file có cả `Review_3_full`, `Review_6_full`, `Review_8_full`, `Review_10_full`, **không hardcode 1 và 2**.

Chuẩn hóa trước: sửa mojibake (`â`, `Ã¢ÂÂ`, `â`), gộp khoảng trắng thừa quanh dấu câu (`" . "` → `". "`).

### B2. Trích atomic claim + gán stance + gán aspect

**Model**: `Qwen3-14B-AWQ` — model mạnh nhất trong 3 cái, dùng cho bước này dù chỉ chạy 1 lượt/câu, không ensemble. Lý do: B2 là bước **nền, không có cơ chế tự kiểm** (không self-consistency, không cross-model, không debate) — lỗi trích `claim` ở đây lọt thẳng xuống B3-B6 mà không ai bắt được (B4 chỉ kiểm tra quan hệ giữa 2 claim đã có sẵn, không kiểm tra bản thân claim có trích đúng từ câu gốc hay không). Ưu tiên độ tin cậy hơn tốc độ ở đúng bước này. Về vận hành: chạy B2 ngay trước lượt gán nhãn B4 của chính Qwen (cùng một lần nạp model, tránh swap model thêm một lần).

**Một lượt LLM duy nhất cho mỗi câu**, trả về 3 trường cùng lúc — rẻ hơn chạy 3 pass:

```json
{"is_evaluative": true, "claim": "...", "stance": "CONCERN", "aspect": "soundness"}
```

**Decoding**: JSON-schema-constrained (`outlines`/`xgrammar`), **khác** với B4. `stance`/`aspect`/`is_evaluative` ép theo enum như B4, nhưng `claim` là free text nằm trong khuôn schema — không thể enum-constrain một chuỗi trích xuất tự do, nên cần grammar ở cấp cấu trúc JSON, không chỉ ở cấp token nhãn.

**Lọc — bỏ 5 nhóm, hai cơ chế khác nhau, không nên trộn lẫn:**

```
Rule-based, chạy TRƯỚC B2 (ở B1, không tốn lượt LLM):
✗ phần hậu-rebuttal

LLM-judged, qua field is_evaluative trong cùng lượt B2:
✗ tóm tắt bài báo ("This paper proposes a framework…")
✗ danh sách typo/grammar ("line 68: …", "Typos: …")
✗ danh sách tài liệu tham khảo
✗ câu hỏi thuần cho tác giả
✓ giữ: câu mang phán xét đánh giá về bài báo
```

Bốn nhóm còn lại cần phán đoán ngữ nghĩa nên phải qua LLM; riêng hậu-rebuttal là so khớp chuỗi thuần túy — làm ở B1 để không tốn ngân sách LLM cho những câu chắc chắn bị loại.

**Cắt hậu-rebuttal** bằng marker: `UPDATE AFTER`, `After reading the author`, `post-rebuttal`, `=== After rebuttal ==`, `I have read the author`, `-- I have read`. Bỏ qua bước này sẽ ghép claim **đã bị chính người viết rút lại** với claim của reviewer kia → nhãn sai không cách nào phát hiện.

**Guard-rail cho `claim` — 2 tầng, bắt buộc vì bước này không có ensemble:**

Fuzzy-match thuần có một lỗ hổng nghiêm trọng: nó chỉ đo **trùng lặp bề mặt từ ngữ**, không đo **ý nghĩa**. Một claim bị đảo cực tính so với câu gốc (VD nguồn *"does NOT improve robustness"* → claim bịa *"improves robustness"*) vẫn có thể đạt fuzzy score cao vì phần lớn từ trùng nhau — trong khi đây chính xác là loại lỗi tai hại nhất ở đây: input bị lật cực tính sẽ kéo sai toàn bộ nhãn quan hệ suy ra sau đó ở B4-B6. Vì vậy dùng 2 tầng thay vì 1:

```
claim ──► (1) lexical grounding ──► (2) semantic entailment ──► accept/reject
              (rẻ, lọc thô)              (NLI model nhỏ/local)
```

1. **Lexical grounding** (`rapidfuzz.partial_ratio`, ngưỡng 0.5): lọc thô, rẻ — bắt các ca claim gần như không liên quan gì tới câu nguồn (hallucination hoàn toàn). Không tốn thêm model, giữ đúng vai trò ban đầu của fuzzy-match.
2. **Semantic entailment check**: các claim qua được tầng 1 chạy tiếp qua **một NLI model nhỏ, local, không cần LLM** — câu nguồn là premise, `claim` là hypothesis:

   ```
   SOURCE: "The method improves robustness under noisy labels."
   CLAIM:  "The method improves robustness."
                       ↓
           premise=SOURCE, hypothesis=CLAIM
                       ↓
                 ENTAILMENT → accept
   ```

   Quyết định: `ENTAILMENT` → nhận; `CONTRADICTION` → loại (đúng ca nguy hiểm nêu trên mà fuzzy-match bỏ lọt); `NEUTRAL` → loại (nhất quán với triết lý fail-closed của contract — claim không được entailment rõ ràng thì chưa đủ tin cậy để vào training set, dù chưa chắc sai hẳn).

   **Model đề xuất**: một checkpoint NLI đa ngôn ngữ nhỏ (VD họ `mDeBERTa-v3-base-mnli-xnli`) — cùng họ kiến trúc với chính backbone đích của dự án (`mDeBERTa-v3-base-xnli`, §2), không phát sinh công cụ/hạ tầng mới. Đây là forward-pass phân loại thuần (không sinh text), chi phí không đáng kể so với 3 model LLM chính — không cần tính vào ngân sách units cùng thang đo với B2-B4.

Ghi số ca bị loại ở từng tầng vào `reports/` để theo dõi tỉ lệ — tỉ lệ loại cao bất thường (đặc biệt ở tầng entailment) là dấu hiệu prompt B2 cần chỉnh.

**Rủi ro khối lượng — độc lập với việc giảm mục tiêu Track B xuống 1.000–2.000 cặp:** B2 chạy trên toàn bộ câu thô của `Review_*_full`, khối lượng này do kích thước corpus quyết định, **không** tự động co giãn khi B3 giảm số cặp đích (5.9 units trong bảng ngân sách giữ nguyên vì lý do này). Pilot 100 cặp phải đo cụ thể trung bình số câu/review và tỉ lệ bị lọc để xác nhận hoặc điều chỉnh con số 0.5h/5.9 units.

**Stance** — 5 lớp, theo đúng quy ước của repo tại [`analyze_service.py:413-421`](../src/modules/synthesis/analyze_service.py#L413-L421):

```
"kiến nghị/đề nghị/cần/nên"        → RECOMMENDATION   ← ưu tiên cao nhất
"chưa/không phù hợp/hạn chế/thiếu" → CONCERN
"phù hợp/hợp lý/tốt/đầy đủ"        → POSITIVE
còn lại                             → NEUTRAL
```

Ba điều phải đưa vào prompt:
- **Thứ tự ưu tiên**: câu vừa khuyến nghị vừa chê → `RECOMMENDATION`, không phải `CONCERN`.
- `NEGATIVE` gần như không xuất hiện ở đường rule của production → đừng để silver set đầy `NEGATIVE`.
- Self-consistency n=3 là đủ (task dễ).

**Aspect** — 6 lớp + bucket `none`: `clarity`, `motivation`, `substance`, `soundness`, `originality`, `meaningful comparison`. Chỉ dùng để **ghép cặp**, không đưa vào input model.

### B3. Ghép cặp

Cùng bài + **cùng aspect** + khác review. Đúng 2 claim mỗi cặp.

`pair_id = {paper_id}:{aspect}:{claimA_id}:{claimB_id}`

Mục tiêu ~1.000–2.000 cặp (giảm từ ước tính ban đầu 5.000 — xem ngân sách §2). Nếu vượt, lấy mẫu ngẫu nhiên có seed, **giữ cân bằng số cặp trên mỗi aspect** để không bị một aspect chiếm hết.

### B4. Ensemble gán nhãn

**Bước 1 — sinh 30 lượt.** Mỗi cặp chạy qua **3 model × n=5 self-consistency × 2 chiều (A,B) và (B,A)** = 30 lượt. Constrained decoding vào đúng 6 nhãn. Temperature 0.5. Ba model chạy **tuần tự trên toàn bộ tập** (nạp model 1, chạy hết ~2.000 cặp, swap model 2, chạy hết, swap model 3) — không phải 30 lượt liên tiếp cho từng cặp riêng lẻ (xem §2).

Prompt chứa: 6 định nghĩa nguyên văn từ [contract](../specs/relation-classifier-contract.md), decision rules cho các ranh giới khó, và few-shot lấy từ `fewshot.jsonl` (nguồn chưa chốt — xem quyết định treo ở §1).

**Vì sao n=5, không phải 2:** ở temperature 0.5, mỗi lượt là 1 mẫu từ phân phối xác suất của model, không phải câu trả lời cố định. Với n=2, một cặp mà model thực sự phân vân (VD phân phối thật ~55/45 giữa 2 nhãn) có xấp xỉ 50% khả năng ra kết quả **hoà** (1-1, không có đa số) — phải thêm luật tie-break tuỳ tiện, tái tạo đúng loại nhiễu mà self-consistency sinh ra để loại bỏ. Số lẻ n=5 luôn cho ra đa số, không bao giờ hoà. Chi phí thêm không lớn vì mỗi lượt chỉ sinh 1 token (constrained decoding) và các lượt cùng prompt chia sẻ chi phí xử lý phần prefix.

> **Cân nhắc hạ xuống n=3 sau pilot.** n=3 vẫn là số lẻ (không bao giờ hoà phiếu — giữ nguyên lợi ích chính so với n=2) nhưng giảm 40% số lượt sinh ở B4. Quyết định dựa trên dữ liệu pilot, không quyết trước: nếu pilot cho thấy **tỉ lệ cặp mà nhãn đa số n=3 khác nhãn đa số n=5 là thấp** (tính ngay trên chính 5 mẫu đã sinh: so đa số của 3 mẫu đầu với đa số của cả 5), thì hạ n=3 là an toàn và phần units tiết kiệm được chuyển sang dự phòng hoặc mở rộng số cặp. Nếu tỉ lệ lệch cao — nghĩa là tập cặp này nhiều ca biên thật — giữ n=5. Đo được miễn phí vì không cần chạy thêm gì ngoài pilot vốn đã bắt buộc.

**Bước 2 — rút gọn self-consistency.** Với mỗi (model, chiều), lấy **nhãn đa số trong 5 mẫu** làm nhãn đại diện. Kết quả: 6 nhãn đại diện/cặp (3 model × 2 chiều).

**Bước 3 — kiểm order-invariance riêng từng model.** So nhãn đại diện chiều (A,B) với chiều (B,A) **của cùng một model**:
- Khớp → model "ổn định" trên cặp này, nhãn đó thành **phiếu** chính thức của model.
- Lệch → model bị đánh dấu **"lật nhãn"** trên cặp này (chỉ đánh dấu cho cặp cụ thể, không loại cả model).

**Order-swap là kiểm tra tin cậy rẻ nhất.** 5/6 nhãn đối xứng; chỉ `COMPLEMENTARY` có tính hướng nhẹ. Model đổi nhãn khi đảo thứ tự = không thực sự hiểu cặp đó — dấu hiệu thiên lệch vị trí (positional bias), khác với nhiễu sampling mà bước 2 đã lọc.

**Bước 4 — tổng hợp 3 phiếu**, đưa vào bảng routing B5 ngay dưới đây.

**Ví dụ minh hoạ** (cặp aspect `originality`):

| Model | Phiếu (A,B) | Phiếu (B,A) | Order-invariant? | Phiếu cuối |
|---|---|---|---|---|
| Qwen3-14B | CONTRADICTION | CONTRADICTION | ✅ | CONTRADICTION |
| Gemma-2-9B | CONTRADICTION | PARTIAL_CONTRADICTION | ❌ lật nhãn | (không ổn định) |
| SeaLLM-7B | CONTRADICTION | CONTRADICTION | ✅ | CONTRADICTION |

Đếm thô trông như "2/3 đồng thuận CONTRADICTION", nhưng Gemma lật nhãn khi đảo chiều → cặp này **không** được nhận thẳng theo luật "đa số, weight 0.7" ở B5 — nó bị đẩy sang debate theo đúng hàng "Lật nhãn khi đảo thứ tự".

### B5. Phân luồng

| Tình huống | Xử lý | Weight |
|---|---|---|
| Đồng thuận 3/3 **và** bất biến thứ tự | nhận | 1.0 |
| Đa số (≥ 2/3) và bất biến thứ tự | nhận | 0.7 |
| Lệch giữa nhãn **kề nhau** | → **debate** | 0.7 sau phán quyết |
| Lệch giữa nhãn **xa nhau** | **loại** | — |
| Lật nhãn khi đảo thứ tự | → **debate** | 0.7 sau phán quyết |

**Nhãn kề nhau** — trên trục quan hệ:

```
AGREEMENT — PARTIAL_AGREEMENT — COMPLEMENTARY — PARTIAL_CONTRADICTION — CONTRADICTION
                                  UNRELATED  ⊥ (trục khác: có cùng bàn một issue hay không)
```

Kề = cách nhau 1 bậc trên trục. `UNRELATED` kề với `COMPLEMENTARY` và không kề với gì khác.

Phân biệt kề/xa là chỗ đa số làm sai: lệch **kề** là ca biên thật, quý nhất; lệch **xa** là cặp ứng viên rác, giữ lại chỉ làm bẩn training set.

### B6. Nhãn mềm

Target là **phân bố thực nghiệm trên các labeler**, train bằng KL — không one-hot theo phe thắng.

Cặp mà 2/3 nói `PARTIAL_AGREEMENT`, 1/3 nói `AGREEMENT` → target `[0.33, 0.67, 0, 0, 0, 0]`. Đó là **thông tin**, không phải nhiễu, và chính các ca biên này là chỗ model sẽ yếu nhất.

---

## 6. Pipeline debate

Chỉ chạy trên các cặp bị B5 route sang — ước ~20%, **cap tuyệt đối 1.200 cặp**. Vượt cap thì ưu tiên theo mức bất đồng. Ở quy mô Track B hiện tại (~1.000–2.000 cặp), 20% ≈ 200–400 cặp — cap 1.200 không còn là ràng buộc thực tế, chỉ giữ làm trần an toàn nếu tỉ lệ bất đồng cao hơn dự kiến.

Debate đắt gấp ~50× một lượt gán nhãn (5 lượt sinh × ~150 token vs 1 lượt × 1 token). Chạy nó trên mọi cặp là đốt ngân sách để cải thiện gần 0 trên các cặp dễ.

**Số vòng: cố định đúng 2, không lặp thêm.** "5 lượt sinh" ở trên chính là: 2 advocate × Vòng 1 (mở đầu) + 2 advocate × Vòng 2 (phản biện) + 1 lượt Judge phán quyết = 5. Sau Vòng 2, Judge **bắt buộc** ra quyết định (`L1`/`L2`/`ABSTAIN`) dù hai advocate chưa đồng thuận — **không có Vòng 3**, không lặp đến khi hội tụ.

Đây là lựa chọn có chủ đích, khác với kiểu debate mở lặp nhiều vòng cho tới khi hội tụ (như trong `Data_Generated_Using_IMPACT.json` — có cặp lặp tới 4 vòng giữa 2 agent trước khi judge can thiệp): ngân sách ở đây cố định, không có API dự phòng để chạy bù nếu số vòng phình ra ngoài dự kiến, nên chi phí mỗi cặp debate phải là **hằng số biết trước** (đúng 5 lượt), không phụ thuộc việc hai bên có đồng thuận hay không. Nếu qua 2 vòng vẫn không thuyết phục được Judge, đó chính là lý do tồn tại nhánh `ABSTAIN` (loại cặp) thay vì lặp thêm vòng.

### Phân vai

Với cặp bị lệch giữa hai nhãn kề `L1` và `L2`:

| Vai | Model |
|---|---|
| Advocate-1 (bảo vệ `L1`) | Gemma-2-9B hoặc SeaLLM-7B |
| Advocate-2 (bảo vệ `L2`) | model còn lại trong hai cái trên |
| **Judge** | **Qwen3-14B-AWQ** — không bao giờ tranh luận |

**Phân vai được chỉ định, không cho model tự chọn phe.** Cho tự chọn thì chỉ nhận lại đúng prior của nó.

**Hoán đổi ngẫu nhiên phe theo từng cặp** (seed ghi lại): Gemma bảo vệ `L1` ở cặp này thì bảo vệ `L2` ở cặp khác. Không làm vậy sẽ sinh thiên lệch hệ thống — ví dụ Gemma luôn là "luật sư của CONTRADICTION".

### Vòng tranh luận

**Vòng 1 — Mở đầu** (hai bên viết độc lập, không thấy nhau)

Mỗi advocate ≤ 120 từ, bắt buộc:
- trích **nguyên văn đoạn** trong từng claim làm căn cứ;
- dẫn **điều khoản rubric cụ thể** mình dựa vào.

**Vòng 2 — Phản biện** (mỗi bên thấy bài mở đầu của bên kia)

Bắt buộc phản bác trực tiếp **đoạn trích mạnh nhất** của đối phương. Cấm đưa ra nhãn thứ ba.

### Phán quyết

Judge nhận:
- cặp claim + stance;
- **rubric hẹp** — chỉ ranh giới `L1` vs `L2`, không đưa cả 6 nhãn;
- 2–4 ví dụ biên từ `fewshot.jsonl` đúng ranh giới đó (nguồn chưa chốt — xem §1);
- hai transcript, **đã xóa danh tính model**, **thứ tự ngẫu nhiên**.

Trả về `L1` | `L2` | `ABSTAIN`, kèm **đoạn trích quyết định**.

`ABSTAIN` → loại cặp. Judge bị ép chọn sẽ sinh nhiễu.

### Bảy biện pháp chống thiên lệch

1. Phân vai chỉ định, không tự chọn phe.
2. Hoán đổi phe ngẫu nhiên giữa các cặp.
3. Judge **không bao giờ** là advocate — tránh self-preference bias, model thiên vị có hệ thống với lập luận của chính nó.
4. Xóa danh tính model khỏi transcript.
5. Ngẫu nhiên thứ tự transcript trong prompt judge — chống position bias.
6. Judge cầm **rubric**, không chỉ cầm transcript. Model bị thuyết phục bởi lập luận nghe tự tin bất kể đúng sai; rubric là mỏ neo.
7. Judge phải trích đoạn quyết định — ép grounding, đồng thời tạo audit trail.

Lưu toàn bộ transcript vào `data/interim/debate_transcripts/` để truy vết.

> **Thứ ăn tiền hơn cả debate là chất lượng rubric.** Rubric sắc thì tỉ lệ đồng thuận ở B5 vọt lên và số cặp phải debate giảm hẳn. Một giờ đầu tư vào `rubric/relation_6label.md` tiết kiệm nhiều giờ compute.

---

## 7. Spot-check thủ công — thay cho bridge đã bỏ

Bridge scoring (κ, confusion matrix, cổng quyết định 0.70/0.50) **không còn trong phạm vi**. Mục này là thứ thay thế rẻ nhất còn giữ được, và phải hiểu đúng nó **không tương đương**.

### Cách làm

1. Rút ngẫu nhiên **~50 cặp** từ `trackB_silver.jsonl`, seed ghi vào `configs/seeds.json`.
2. Stratified theo **hai trục**: nhãn `relation` (để mọi lớp đều có mặt, kể cả lớp hiếm) và `provenance.route` (`unanimous` / `majority` / `debate` — ca `debate` là ca biên khó nhất, phải xem riêng, không để mẫu ngẫu nhiên bỏ sót).
3. Đọc tay, đánh dấu đồng ý / không đồng ý với nhãn pipeline. Ghi kết quả + các mẫu lỗi quan sát được vào `reports/spotcheck.md`.

### Đọc kết quả thế nào

Cái cần tìm là **mẫu lỗi hệ thống**, không phải tỉ lệ phần trăm:

- Một ranh giới bị lệch đều một hướng (VD `PARTIAL_CONTRADICTION` liên tục bị gán thành `CONTRADICTION`) → siết đúng ranh giới đó trong `rubric/relation_6label.md`, chạy lại B4.
- Claim trích hỏng lọt qua guard-rail B2 → chỉnh prompt B2 hoặc ngưỡng entailment.
- Một aspect toàn nhãn rác → xem lại bước ghép cặp B3 cho aspect đó.
- Nhóm `route=debate` sai nhiều hơn hẳn nhóm `unanimous` → rubric hẹp đưa cho Judge chưa đủ sắc.

Dự phòng ~61 units đủ cho **một lần chạy lại B4 toàn tập** nếu spot-check lộ ra lỗi hệ thống — đây chính là công dụng đã tính trước của phần dự phòng đó.

### Giới hạn — bắt buộc khai trong model card

Spot-check **không phải** thước đo độc lập và không được trình bày như thể là:

- cỡ mẫu ~50, không đủ để tính κ có ý nghĩa thống kê;
- không có gold set độc lập — người chấm cũng chính là người viết rubric và prompt, nên thiên lệch xác nhận (confirmation bias) là có thật và không khử được;
- không phát hiện được lỗi mà cả rubric lẫn người chấm cùng mắc.

Nói cách khác: nó bắt được lỗi **thô và hệ thống**, không chứng minh được silver set đúng. Model card phải ghi rõ dataset này **chưa từng được đối chiếu với nhãn người trên một tập kiểm độc lập**.

---

## 8. Dịch sang tiếng Việt

Gán nhãn ở **tiếng Anh trước**, dịch sau. Ba labeler đều mạnh hơn ở tiếng Anh, và corpus nguồn là tiếng Anh.

- Công cụ: `vinai/vinai-translate-en2vi-v2`
- Tỉ lệ: **giữ 60% nguyên EN / dịch 40% sang VI** — tận dụng cross-lingual transfer của mDeBERTa, giảm nhiễu dịch, tăng robust với code-switch.

### Test giữ nhãn — bắt buộc

Trục `PARTIAL_*` hoàn toàn là **trục hedging**. Đọc justification trong dataset sẽ thấy:

> *"the softener 'a bit' does not remove the core accusation of being misleading"* — `NIPS_2019_175`
> *"framed as conditional on a particular confounder and does not amount to a blanket rejection"* — `ICLR_2019_206`

MT chuyên dụng có xu hướng **chuẩn hóa ngôn ngữ rào đón cho gọn**, làm `PARTIAL_CONTRADICTION` trượt thành `CONTRADICTION`. Nên phải đo:

1. Chạy labeler trên bản EN và bản VI của **cùng một mẫu ~100 cặp** rút từ `trackB_silver.jsonl`, ưu tiên cặp có hedging và cặp nhãn `PARTIAL_*`.
2. Báo cáo tỉ lệ lật nhãn, tách riêng cho nhóm `PARTIAL_*`.
3. Ghi vào `reports/translation_fidelity.md`.

**Ngưỡng leo thang:** nếu tỉ lệ lật trên `PARTIAL_*` **> 15%**, chuyển riêng nhóm câu có hedging sang dịch bằng LLM local (SeaLLM/Qwen3 với prompt giữ tình thái). Chỉ là một tập con nên chi phí thấp.

Bộ dò hedging bằng từ khóa: `may`, `might`, `somewhat`, `a bit`, `arguably`, `not necessarily`, `could`, `partly`, `to some extent`, `rather`, `slightly`, `seems`, `appears`.

---

## 9. Schema output

Một dòng training data:

```json
{
  "pair_id": "ICLR_2019_206:clarity:r1s07:r2s12",
  "source": "trackB",
  "lang": "en",
  "left":  {"text": "The paper is clearly written.", "stance": "POSITIVE"},
  "right": {"text": "I found the presentation of the proposed measure overly confusing.", "stance": "CONCERN"},
  "relation": "CONTRADICTION",
  "soft_label": {
    "AGREEMENT": 0.0, "PARTIAL_AGREEMENT": 0.0, "CONTRADICTION": 0.8,
    "PARTIAL_CONTRADICTION": 0.2, "COMPLEMENTARY": 0.0, "UNRELATED": 0.0
  },
  "weight": 1.0,
  "provenance": {
    "paper_id": "ICLR_2019_206",
    "aspect": "clarity",
    "route": "unanimous",
    "labelers": ["qwen3-14b-awq", "gemma-2-9b", "seallm-v3-7b"]
  }
}
```

**`criterion_id` / `aspect` không nằm trong input model.** Contract đảm bảo mọi cặp cùng criterion nên nó không mang tín hiệu phân biệt nhãn; tập giá trị lại nhỏ nên chỉ tạo tương quan giả. Giữ trong `provenance` để truy vết.

Format input cho model:

```
[PB-01|CONCERN] {left.text} [SEP] [PB-02|POSITIVE] {right.text}
```

Phải **phá shortcut stance**: chèn mẫu có stance ngược nhau nhưng nhãn là `COMPLEMENTARY`/`UNRELATED`, nếu không model học tắt "stance lệch ⇒ CONTRADICTION".

---

## 10. Manifest — làm sớm

SHA-256 mọi file train/validation → `train_validation_manifest.json`.

Đây là **input bắt buộc Hiếu phải nhận trước khi tạo gold** ([eval-contract §2](../specs/relation-evaluation-contract.md#L40)), và manifest của Hiếu có field `train_validation_hash_source` chỉ có thể đến từ đây. Đừng để cuối — Hiếu đang bị chặn.

---

## 11. Rủi ro và giới hạn phải khai

| # | Rủi ro | Giảm thiểu |
|---|---|---|
| 1 | Dịch MT làm chết hedging → hỏng `PARTIAL_*` | test giữ nhãn, ngưỡng leo thang 15% |
| 2 | **Không còn cổng chất lượng nào** — bridge/κ đã bị bỏ, train mù trên silver set | spot-check ~50 cặp (§7); giữ dự phòng đủ chạy lại B4 một lần; khai rõ trong model card |
| 3 | Domain gap: review ML tiếng Anh ≠ phiếu C4 hội đồng VN | khai trong model card; register khác nhưng cấu trúc quan hệ giống |
| 4 | Silver label kế thừa thiên lệch của 3 model, **không còn κ để đo** | order-swap + debate + nhãn mềm giữ lại bất đồng; spot-check bắt lỗi thô; báo cáo trung thực về việc không đo được |
| 5 | Ngân sách compute không có API dự phòng | pilot 100 cặp trước; giữ ≥ 30 units dự phòng |
| 6 | `NEGATIVE` hiếm ở production nhưng có thể nhiều trong silver | kiểm phân bố stance, cân lại nếu lệch |

---

## 12. Bản đồ aspect → tiêu chí C4

Chỉ dùng khi dịch, để câu tiếng Việt nằm đúng ngữ cảnh tiêu chí. Không đưa vào input.

| Aspect | Tiêu chí C4 |
|---|---|
| `soundness` | C2.3 Mức độ phù hợp của phương pháp nghiên cứu |
| `substance` | C3.1 Tính đầy đủ của kết quả và sản phẩm dự kiến |
| `meaningful comparison` | C1.2 Tổng quan tình hình nghiên cứu |
| `originality` | C1.1 Ý nghĩa khoa học và thực tiễn |
| `motivation` | C1.1 Ý nghĩa khoa học và thực tiễn |
| `clarity` | C7 Đánh giá tổng hợp |

**Bảy tiêu chí không có nguồn ngoài:** C3.2 (sở hữu trí tuệ), C4.2 (chuyển giao), C5.1 (kế hoạch), C5.2 (kinh phí), C6.1 và C6.2 (năng lực), C8.1 (kiến nghị). Peer review quốc tế không bàn về những thứ này.

Vì `criterion_id` không nằm trong input model nên đây là **known limitation cần khai**, không phải blocker — model chỉ học hình dạng ngôn ngữ của quan hệ, không biết đang xét tiêu chí nào.

---

## 13. Cấu trúc thư mục

```
phase-2/
├── PHASE2_PLAN.md                    ← file này
├── configs/
│   ├── models.yaml                   model id, quantization, vLLM params
│   ├── seeds.json                    R-3
│   └── budget.md                     theo dõi units đã tiêu
├── data/
│   ├── raw/Human_Annotated_Data.json
│   ├── interim/
│   │   ├── claims.jsonl              sau B2
│   │   ├── pairs.jsonl               sau B3
│   │   ├── labels_raw.jsonl          30 lượt/cặp, chưa gộp
│   │   └── debate_transcripts/
│   └── processed/
│       ├── fewshot.jsonl            Bước 0 — nguồn chưa chốt (§1)
│       ├── trackB_silver.jsonl       Bước 2
│       └── train_{en,vi}.jsonl       Bước 4
├── pipeline/
│   ├── common/                       normalize, fuzzy match, io, vllm wrapper
│   ├── track_b/                      b1_split … b6_soft_label, debate/
├── prompts/
│   ├── claim_extraction/  stance/  aspect/  labeler/  debate/
├── rubric/
│   ├── relation_6label.md            6 định nghĩa + decision rules + ví dụ biên
│   ├── stance_5label.md
│   └── aspect_6label.md
├── reports/
│   ├── spotcheck.md
│   └── translation_fidelity.md
└── train_validation_manifest.json
```

---

## 14. Ghi chú phạm vi

`phase-2/` nằm **ngoài** allowlist `experiments/relation_classifier/**` của MODEL-001. Scope guard trong CI sẽ chặn nếu PR này target `feat/gtq-upgrade` ([00-shared §6](../docs/plans/00-shared-contract-and-conflict-rules.md#L118)).

Nếu cần merge vào integration branch, phải mirror sang `experiments/relation_classifier/` và giữ PR chỉ chứa vùng đó cộng một dòng `WORKLOG.md`.
