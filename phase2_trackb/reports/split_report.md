# Báo cáo chia dữ liệu — trackB_silver

> Sinh tự động bởi `src/train/split.py` · seed `20260828` · 1099 cặp / 222 nhóm paper

Chia theo **nhóm paper** (chống rò rỉ) **và** stratified theo nhãn (để lớp hiếm không dồn cục).

**Ghim vào train:** 39 cặp thuộc `fewshot_human_pairs` — đó chính là các ví dụ few-shot đã dùng để gán nhãn phần còn lại, xem `PIN_TRAIN_SOURCES` trong `split.py`.


## 1 · Phân bố nhãn

| Nhãn | toàn bộ | % | train | % | lệch | val | % | lệch | test | % | lệch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AGREEMENT | 76 | 6.9% | 61 | 7.0% | +0.0 | 7 | 6.2% | -0.7 | 8 | 7.2% | +0.3 |
| PARTIAL_AGREEMENT | 217 | 19.7% | 172 | 19.6% | -0.1 | 22 | 19.6% | -0.1 | 23 | 20.7% | +1.0 |
| COMPLEMENTARY | 496 | 45.1% | 393 | 44.9% | -0.3 | 53 | 47.3% | +2.2 | 50 | 45.0% | -0.1 |
| PARTIAL_CONTRADICTION | 168 | 15.3% | 134 | 15.3% | +0.0 | 17 | 15.2% | -0.1 | 17 | 15.3% | +0.0 |
| CONTRADICTION | 29 | 2.6% | 25 | 2.9% | +0.2 | 2 | 1.8% | -0.9 | 2 | 1.8% | -0.8 |
| UNRELATED | 113 | 10.3% | 91 | 10.4% | +0.1 | 11 | 9.8% | -0.5 | 11 | 9.9% | -0.4 |
| **tổng** | 1099 | 100% | 876 | 79.7% |  | 112 | 10.2% |  | 111 | 10.1% |  |

Lệch lớn nhất so với phân bố toàn cục: **2.2 điểm %**.


## 2 · Nguồn dữ liệu × phần

| `source` | toàn bộ | train | % | val | % | test | % |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fewshot_human_pairs` | 39 | 39 | 4.5% | 0 | 0.0% | 0 | 0.0% |
| `full_batch00_manual` | 200 | 154 | 17.6% | 26 | 23.2% | 20 | 18.0% |
| `full_batch01_manual` | 200 | 157 | 17.9% | 24 | 21.4% | 19 | 17.1% |
| `full_batch02_manual` | 200 | 160 | 18.3% | 20 | 17.9% | 20 | 18.0% |
| `full_batch03_manual` | 200 | 151 | 17.2% | 22 | 19.6% | 27 | 24.3% |
| `mined_stance_opposition` | 150 | 127 | 14.5% | 9 | 8.0% | 14 | 12.6% |
| `synthetic_cross_paper` | 110 | 88 | 10.0% | 11 | 9.8% | 11 | 9.9% |

Cần đọc cùng muc 4.6 của `docs/TRAIN.md`: `synthetic_cross_paper` sinh bằng luật, rất dễ đoán, nên phải đo tách chứ không để nó làm đẹp macro-F1 tổng.


## 3 · Kiểm tra rò rỉ

| Mức | Cặp phần | Số trùng |
|---|---:|---:|
| paper (đã `assert`) | mọi cặp | 0 |
| text của claim | train ∩ val | 1 |
| text của claim | train ∩ test | 0 |
| text của claim | val ∩ test | 0 |

Mức claim chặt hơn mức paper. Phần dư đến từ cặp UNRELATED chéo paper (`paper_id` dạng `A|B`), vì `group_of()` chỉ lấy paper bên trái — đánh đổi có ý thức, xem docstring của hàm.


## 4 · 5-fold (nhóm paper + stratified)

| Nhãn | chấm điểm | ghim→train | f0 | f1 | f2 | f3 | f4 | min–max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AGREEMENT | 70 | 6 | 14 | 14 | 14 | 14 | 14 | 14–14 |
| PARTIAL_AGREEMENT | 210 | 7 | 42 | 42 | 42 | 42 | 42 | 42–42 |
| COMPLEMENTARY | 488 | 8 | 98 | 91 | 103 | 98 | 98 | 91–103 |
| PARTIAL_CONTRADICTION | 160 | 8 | 33 | 31 | 31 | 32 | 33 | 31–33 |
| CONTRADICTION | 22 | 7 | 4 | 5 | 4 | 5 | 4 | 4–5 |
| UNRELATED | 110 | 3 | 21 | 23 | 22 | 22 | 22 | 21–23 |
| **tổng** | 1060 | 39 | 212 | 206 | 216 | 213 | 213 | 206–216 |

Cột **ghim→train** nhận `fold = -1`: có mặt trong train của mọi fold, không bao giờ bị chấm điểm. Đây mới là bảng để đọc kết quả (muc 2 của `docs/TRAIN.md`) — test chỉ vài mẫu CONTRADICTION nên F1 lớp đó trên test vẫn không đọc được.


## 5 · Trọng số lớp (inverse-frequency, trên train)

| Nhãn | n train | w |
|---|---:|---:|
| AGREEMENT | 61 | 2.39 |
| PARTIAL_AGREEMENT | 172 | 0.85 |
| COMPLEMENTARY | 393 | 0.37 |
| PARTIAL_CONTRADICTION | 134 | 1.09 |
| CONTRADICTION | 25 | 5.84 |
| UNRELATED | 91 | 1.60 |

⚠ `train.py` phải tính lại trọng số trên train của **từng fold**, file này chỉ dùng cho lần train một-lượt.
