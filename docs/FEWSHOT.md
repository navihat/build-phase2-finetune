# few-shot cho Track B — nguồn gốc và giới hạn

`fewshot.jsonl` (39 ví dụ) là thứ lấp vào **quyết định treo D1** ở [`PHASE2_PLAN.md`](PLAN.md) §1.
File này ghi lại nó được dựng thế nào, để không ai — kể cả chính nhóm sáu tháng sau — phải
đoán xem nhãn ở đâu ra.

## Nguồn

`data/raw/Human_Annotated_Data.json`: 75 paper, 197 mục `analysis` do **annotator người** dựng.
Mỗi mục có 2 evidence quote nguyên văn từ hai review khác nhau, `aspect`, và `intensity.score`
kèm justification.

Plan §1 viết rằng nguồn few-shot "mất chỗ dựa" vì `trackA_fewshot.jsonl` không còn. Điều đó
đúng ở thời điểm viết, nhưng file human ở trên cung cấp đúng thứ Track A từng cung cấp:
**cặp claim đã được người xác nhận là nói về cùng một chuyện.**

## Ai gán nhãn — đọc kỹ chỗ này

Quy trình tách làm hai phần, và hai phần có mức tin cậy KHÁC NHAU:

| Việc | Ai làm | Tin cậy |
|---|---|---|
| Xác định "hai claim này nói về cùng một issue" | **người** (annotator gốc) | nhãn người |
| Trích evidence quote nguyên văn | **người** (annotator gốc) | nhãn người |
| Ánh xạ sang 6 nhãn của contract | **Claude**, đọc tay theo RUBRIC | **KHÔNG phải nhãn người** |

Nghĩa là đây **không phải phương án (a)** trong plan §1 (người tự gán ~40 cặp). Nó là phương
án thứ ba, nằm giữa (a) và (b): nền người, nhãn LLM.

`intensity.score` **không** dùng làm nhãn. Đã thử và đo: ánh xạ score→nhãn lệch **58% (21/36)**
so với đọc tay, vì `intensity` đo *độ tương phản cảm nhận* còn contract hỏi *hai claim có loại
trừ nhau không*. Hai construct khác nhau. Xem `data/fewshot/draft.jsonl`, trường `_agrees_with_machine`.

### Phải khai vào manifest

Thêm vào `known_limitations` trong cell Manifest của notebook:

```
"Few-shot (39 ví dụ) lấy cặp và evidence từ nhãn người (Human_Annotated_Data.json),
 nhưng nhãn 6-lớp theo contract do LLM gán, không có người xác nhận lại.
 Neo hiệu chuẩn của B4/Judge vì vậy yếu hơn phương án 'người tự gán 40 cặp' ở PHASE2_PLAN §1.",
```

Muốn nâng lên phương án (a) thật: đọc lại 39 dòng, sửa nhãn nào thấy sai, rồi xoá đoạn
`known_limitations` trên. Chi phí ~20 phút, và đây là neo hiệu chuẩn người duy nhất còn lại
của cả pipeline nên nó đáng.

## R-1 (few-shot không nằm trong tập train) — bảo đảm bằng cấu trúc

75 paper của `Human_Annotated_Data.json` **không trùng paper nào** trong 263 paper của
`Data_Generated_Using_IMPACT.json`. Đã kiểm thêm ở mức text: **0/78** vế của few-shot xuất
hiện trong corpus IMPACT.

Mạnh hơn hẳn R-1 ở cell B3, vốn khớp text chính xác nên không bắt được trường hợp B2 viết
lại câu gốc thành claim gần giống.

> ⚠️ **Chỉ upload `Data_Generated_Using_IMPACT.json` vào `{ROOT}/raw/`.**
> Upload cả file human là tự phá đảm bảo trên: câu nguồn của few-shot chảy vào tập train
> dưới dạng claim đã viết lại, và R-1 không bắt được. Đổi lại chỉ mất ~3.3k trong ~27k câu.

## Chọn cặp thế nào

Bỏ 20/197 mục lệch phân phối — quote không thể sinh ra từ pipeline nên làm few-shot là dạy
model một kiểu input không tồn tại:

| Loại | Bỏ | Vì |
|---|---|---|
| câu hỏi thuần | 10 | B2 đặt `is_evaluative=false` cho câu hỏi không kèm phán xét |
| quote quá ngắn, thiếu ngữ cảnh | 6 | không qua được B2-guard (claim phải đứng một mình đọc hiểu) |
| nội dung hậu-rebuttal | 4 | B1 `cut_rebuttal()` cắt sạch trước khi tới B4 |

Cũng bóc tiền tố `"Review 1: '...'"` bị nhúng sẵn trong một số quote, vì claim thật từ B2
không bao giờ có tiền tố đó.

## Phân bố

| Nhãn | Số ví dụ |
|---|---|
| AGREEMENT | 6 |
| PARTIAL_AGREEMENT | 7 |
| COMPLEMENTARY | 8 |
| PARTIAL_CONTRADICTION | 8 |
| CONTRADICTION | 7 |
| UNRELATED | 3 |

Mỗi ranh giới liền kề trên trục quan hệ có ≥11 ví dụ, dư cho `k=4` mà prompt Judge dùng.

**UNRELATED chỉ có 3** và cả ba là cặp **ghép chéo**: cùng paper, cùng aspect (đúng ràng buộc
ghép cặp của B3), nhưng lấy từ hai mục `analysis` khác nhau nên khác issue. Không rút được từ
cặp gốc, vì theo định nghĩa mọi cặp trong dataset này đã là "cùng một issue". Chấp nhận được:
trong `is_adjacent()` của notebook, UNRELATED chỉ kề COMPLEMENTARY nên nó cần ít ví dụ hơn các
nhãn nằm giữa trục. Muốn thêm thì rút từ `interim/pairs.jsonl` sau khi chạy pilot.

## Một phát hiện về domain

Trong 36 cặp mà annotator người dán nhãn "contradiction", đọc theo contract chỉ **5 là
CONTRADICTION**, còn **10 là COMPLEMENTARY**. Cái mà người quen gọi là "hai reviewer mâu thuẫn"
phần lớn là hai người soi hai khía cạnh khác nhau của cùng một vấn đề.

Hệ quả cần theo dõi ở pilot: phân bố nhãn của `trackB_silver.jsonl` nhiều khả năng lệch nặng
về PARTIAL_CONTRADICTION/COMPLEMENTARY, và **ranh giới #4 của rubric** ("stance đối nghịch
KHÔNG tự động là mâu thuẫn") mới là điều khoản ăn tiền nhất, chứ không phải trục hedging.

## Thay đổi kèm theo trong notebook

`fewshot_block()` ở cell 0.5 có docstring ghi "labels=None -> trải đều mọi nhãn" nhưng cài đặt
lại chỉ shuffle rồi lấy `k` phần tử đầu. Đo thực tế: bốc 6 từ 39 ví dụ chỉ phủ **3/6 nhãn** —
prompt B4 thiên lệch đúng ở chỗ cần cân nhất. Đã sửa thành round-robin theo nhãn, và nâng B4
từ `k=6` lên `k=12` (2 ví dụ/nhãn, ~1.1k token).

Sau khi sửa: `k=6` phủ 6/6, `k=12` phủ 2 mỗi nhãn, debate `k=4` cân 2+2 mỗi ranh giới, vẫn tái
lập được theo seed. Bản gốc ở `notebooks/track_b_pipeline.ipynb.bak`.

## Dựng lại

```powershell
python src/fewshot/build_candidates.py   # 36 ứng viên + nhãn máy suy từ intensity
python src/fewshot/apply_draft.py        # pass đọc tay, kèm cờ lệch với nhãn máy
python src/fewshot/build_final.py        # -> data/fewshot/fewshot.jsonl
```

## Chạy

1. Upload `fewshot.jsonl` -> `MyDrive/phase2_trackb/processed/fewshot.jsonl`
2. Upload **chỉ** `Data_Generated_Using_IMPACT.json` -> `raw/`
3. Xoá `interim/` nếu có output cũ (B1/B2/B2-guard/B3 không có `force=True`, sẽ `[SKIP]`)
4. Chạy cell 0.5, xác nhận in `[OK] Nạp 39 ví dụ few-shot.` — thấy `[CẢNH BÁO]` là chưa nạp được
5. Chạy pilot, đối chiếu tỉ lệ đồng thuận 3/3 ở B5 với lần chạy zero-shot. Đó là toàn bộ lý do
   bỏ công vào việc này: few-shot phải kéo đồng thuận lên và kéo số cặp vào debate xuống.
