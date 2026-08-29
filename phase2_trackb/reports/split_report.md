# Báo cáo chia dữ liệu — trackB_silver

> Sinh tự động bởi `src/train/split.py` · seed `20260828` · 989 cặp / 210 nhóm paper

Chia theo **nhóm paper** (chống rò rỉ) **và** stratified theo nhãn (để lớp hiếm không dồn cục).

**Ghim vào train:** 39 cặp thuộc `fewshot_human_pairs` — đó chính là các ví dụ few-shot đã dùng để gán nhãn phần còn lại, xem `PIN_TRAIN_SOURCES` trong `split.py`.


## 1 · Phân bố nhãn

| Nhãn | toàn bộ | % | train | % | lệch | val | % | lệch | test | % | lệch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AGREEMENT | 76 | 7.7% | 61 | 7.7% | +0.1 | 7 | 7.0% | -0.7 | 8 | 7.9% | +0.2 |
| PARTIAL_AGREEMENT | 217 | 21.9% | 173 | 22.0% | +0.0 | 22 | 22.0% | +0.1 | 22 | 21.8% | -0.2 |
| COMPLEMENTARY | 153 | 15.5% | 122 | 15.5% | +0.0 | 16 | 16.0% | +0.5 | 15 | 14.9% | -0.6 |
| PARTIAL_CONTRADICTION | 168 | 17.0% | 134 | 17.0% | +0.0 | 17 | 17.0% | +0.0 | 17 | 16.8% | -0.2 |
| CONTRADICTION | 29 | 2.9% | 25 | 3.2% | +0.2 | 2 | 2.0% | -0.9 | 2 | 2.0% | -1.0 |
| UNRELATED | 346 | 35.0% | 273 | 34.6% | -0.3 | 36 | 36.0% | +1.0 | 37 | 36.6% | +1.6 |
| **tổng** | 989 | 100% | 788 | 79.7% |  | 100 | 10.1% |  | 101 | 10.2% |  |

Lệch lớn nhất so với phân bố toàn cục: **1.6 điểm %**.


## 2 · Nguồn dữ liệu × phần

| `source` | toàn bộ | train | % | val | % | test | % |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fewshot_human_pairs` | 39 | 39 | 4.9% | 0 | 0.0% | 0 | 0.0% |
| `full_batch00_manual` | 200 | 151 | 19.2% | 23 | 23.0% | 26 | 25.7% |
| `full_batch01_manual` | 200 | 161 | 20.4% | 16 | 16.0% | 23 | 22.8% |
| `full_batch02_manual` | 200 | 157 | 19.9% | 21 | 21.0% | 22 | 21.8% |
| `full_batch03_manual` | 200 | 156 | 19.8% | 24 | 24.0% | 20 | 19.8% |
| `mined_stance_opposition` | 150 | 124 | 15.7% | 16 | 16.0% | 10 | 9.9% |

Cần đọc cùng muc 4.6 của `docs/TRAIN.md`: `synthetic_cross_paper` sinh bằng luật, rất dễ đoán, nên phải đo tách chứ không để nó làm đẹp macro-F1 tổng.


## 3 · Kiểm tra rò rỉ

| Mức | Cặp phần | Số trùng |
|---|---:|---:|
| paper (đã `assert`) | mọi cặp | 0 |
| text của claim | train ∩ val | 0 |
| text của claim | train ∩ test | 0 |
| text của claim | val ∩ test | 0 |

Mức claim chặt hơn mức paper. Phần dư đến từ cặp UNRELATED chéo paper (`paper_id` dạng `A|B`), vì `group_of()` chỉ lấy paper bên trái — đánh đổi có ý thức, xem docstring của hàm.


## 4 · 5-fold (nhóm paper + stratified)

| Nhãn | chấm điểm | ghim→train | f0 | f1 | f2 | f3 | f4 | min–max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AGREEMENT | 70 | 6 | 14 | 12 | 14 | 15 | 15 | 12–15 |
| PARTIAL_AGREEMENT | 210 | 7 | 43 | 39 | 43 | 42 | 43 | 39–43 |
| COMPLEMENTARY | 149 | 4 | 32 | 28 | 30 | 30 | 29 | 28–32 |
| PARTIAL_CONTRADICTION | 160 | 8 | 34 | 32 | 32 | 30 | 32 | 30–34 |
| CONTRADICTION | 22 | 7 | 4 | 5 | 4 | 5 | 4 | 4–5 |
| UNRELATED | 339 | 7 | 66 | 70 | 65 | 69 | 69 | 65–70 |
| **tổng** | 950 | 39 | 193 | 186 | 188 | 191 | 192 | 186–193 |

Cột **ghim→train** nhận `fold = -1`: có mặt trong train của mọi fold, không bao giờ bị chấm điểm. Đây mới là bảng để đọc kết quả (muc 2 của `docs/TRAIN.md`) — test chỉ vài mẫu CONTRADICTION nên F1 lớp đó trên test vẫn không đọc được.


## 5 · Trọng số lớp (inverse-frequency, trên train)

| Nhãn | n train | w |
|---|---:|---:|
| AGREEMENT | 61 | 2.15 |
| PARTIAL_AGREEMENT | 173 | 0.76 |
| COMPLEMENTARY | 122 | 1.08 |
| PARTIAL_CONTRADICTION | 134 | 0.98 |
| CONTRADICTION | 25 | 5.25 |
| UNRELATED | 273 | 0.48 |

⚠ `train.py` phải tính lại trọng số trên train của **từng fold**, file này chỉ dùng cho lần train một-lượt.
