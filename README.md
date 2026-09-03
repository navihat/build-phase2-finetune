# Phase 2 — Track B: Relation Classifier

Sinh tập dữ liệu và fine-tune bộ phân loại **quan hệ 6 lớp** giữa hai atomic claim của
hai reviewer về cùng một bài báo:

```
AGREEMENT — PARTIAL_AGREEMENT — COMPLEMENTARY — PARTIAL_CONTRADICTION — CONTRADICTION
                                      ⊥
                                  UNRELATED
```

## Cấu trúc

```
docs/          PLAN.md · CHECKLIST.md · FEWSHOT.md · TRAIN.md
notebooks/     01_data_pipeline.ipynb (B1→B6 sinh dữ liệu, Colab)
               02_train_relation_classifier.ipynb (fine-tune + đánh giá, Colab)
src/fewshot/   dựng tập few-shot từ nhãn người
src/train/     split.py (chia theo paper) · train.py (fine-tune)
data/raw/      hai file dữ liệu gốc
data/fewshot/  fewshot.jsonl và các bản trung gian
pipeline_data/ mọi output của pipeline sinh dữ liệu: raw/interim/processed/reports/configs
```

> **`pipeline_data/` là bản đồng bộ thủ công của thư mục `MyDrive/phase2_trackb` trên Google
> Drive** (tên trên Drive giữ nguyên `phase2_trackb`, chỉ đổi tên bản trong repo này cho dễ
> đọc). Script `src/train/*.py` đọc/ghi trực tiếp vào `pipeline_data/`, nên **đừng đổi tên nó
> lần nữa** nếu không sửa lại các script đó.
> `pipeline_data/raw/` không được track (trùng hệt `data/raw/`, 32MB).

## Bắt đầu từ đâu

| Bạn muốn | Đọc |
|---|---|
| Hiểu toàn bộ kế hoạch | [`docs/PLAN.md`](docs/PLAN.md) |
| Chạy pipeline sinh dữ liệu | [`docs/CHECKLIST.md`](docs/CHECKLIST.md) |
| **Fine-tune và đánh giá model** | [`docs/TRAIN.md`](docs/TRAIN.md) |
| Nhãn few-shot ở đâu ra | [`docs/FEWSHOT.md`](docs/FEWSHOT.md) |

## Trạng thái

Tập silver: **1099 cặp**, cả 6 lớp đều có mẫu
(`pipeline_data/processed/trackB_silver.jsonl`). Đã chia train/val/test + 5-fold theo
nhóm paper. Script train và notebook Colab đã sẵn sàng, **chưa chạy end-to-end**.

Chi tiết và việc còn lại: [`docs/TRAIN.md`](docs/TRAIN.md) mục 7.
