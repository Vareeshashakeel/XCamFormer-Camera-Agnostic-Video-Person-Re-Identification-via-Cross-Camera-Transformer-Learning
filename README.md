# XCamFormer: Camera-Agnostic Video Person Re-Identification via Cross-Camera Transformer Learning

This repository contains the implementation of **XCamFormer**, a camera-agnostic video person re-identification framework based on cross-camera transformer learning.

XCamFormer removes camera/view embeddings from the transformer input and uses camera labels only during training to construct cross-camera positive relationships. During inference, the model uses only visual embeddings for query-gallery retrieval.

## Key Features

- Vision Transformer-based video person Re-ID framework.
- Camera/view metadata is **not** injected into the input representation.
- Camera labels are used only during training for cross-camera supervision.
- Intermediate Cross-Camera Supervision is applied at transformer block indices `5` and `8`.
- Block indices follow zero-based implementation indexing, so indices `5` and `8` correspond to the 6th and 9th transformer encoder blocks.
- Part-Level Cross-Camera Supervision is applied to four local VPPF part features.
- Same-identity cross-camera pairs receive the primary positive weight, while same-identity same-camera pairs are retained with a lower weight.
- Inference remains camera-agnostic and uses only the final visual embedding.

## Repository Structure

```text
.
├── Dataloader.py
├── Datasets/
│   ├── MARS_dataset.py
│   ├── PRID_dataset.py
│   └── iLDSVID.py
├── Loss_fun.py
├── VID_Trans_ReID.py       # training entry point
├── VID_Test.py             # evaluation entry point
├── VID_Test_Final.py       # optional/final evaluation script
├── VID_Trans_model.py      # XCamFormer model definition
├── vit_ID.py               # ViT backbone components
├── utility.py
├── loss/
│   ├── center_loss.py
│   ├── softmax_loss.py
│   ├── triplet_loss.py
│   └── xcamera_supcon.py
├── requirements.txt
└── README.md
```

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

The code was developed and tested in a GPU-enabled PyTorch environment.

## Dataset Paths

The dataset paths are currently defined inside the dataset loader files under `Datasets/`.

Before running the code on your own system, update the dataset root paths in:

```text
Datasets/MARS_dataset.py
Datasets/PRID_dataset.py
Datasets/iLDSVID.py
```

The code expects the standard video Re-ID dataset organization for MARS, PRID2011, and iLIDS-VID.

## Pretrained ViT Weights

Set the path to the ImageNet-pretrained ViT model before training. Example:

```bash
export VIT=/path/to/jx_vit_base_p16_224-80ecf9dd.pth
```

## Training Commands

### MARS

```bash
python VID_Trans_ReID.py \
  --Dataset_name Mars \
  --model_path "$VIT" \
  --output_dir ./outputs_partxcam_bs64 \
  --epochs 120 \
  --eval_every 10 \
  --batch_size 64 \
  --num_workers 4 \
  --seq_len 4 \
  --center_w 0.0005 \
  --attn_w 1.0 \
  --xcam_w 0.15 \
  --xcam_temp 0.07 \
  --xcam_same_cam_w 0.25 \
  --xcam_blocks 5,8 \
  --part_xcam_w 0.10 \
  --part_xcam_temp 0.07 \
  --part_xcam_same_cam_w 0.10
```

### PRID2011

```bash
python VID_Trans_ReID.py \
  --Dataset_name PRID \
  --model_path "$VIT" \
  --output_dir ./outputs_partxcam_prid_BS16 \
  --epochs 160 \
  --eval_every 20 \
  --batch_size 16 \
  --num_workers 2 \
  --seq_len 4 \
  --center_w 0.0005 \
  --attn_w 1.0 \
  --xcam_w 0.10 \
  --xcam_temp 0.07 \
  --xcam_same_cam_w 0.25 \
  --xcam_blocks 5,8 \
  --part_xcam_w 0.08 \
  --part_xcam_temp 0.07 \
  --part_xcam_same_cam_w 0.10
```

### iLIDS-VID

```bash
python VID_Trans_ReID.py \
  --Dataset_name iLIDSVID \
  --model_path "$VIT" \
  --output_dir ./outputs_partxcam_ilids_tuned_16batchsize \
  --epochs 160 \
  --eval_every 20 \
  --batch_size 16 \
  --num_workers 2 \
  --seq_len 4 \
  --center_w 0.0005 \
  --attn_w 1.0 \
  --xcam_w 0.10 \
  --xcam_temp 0.07 \
  --xcam_same_cam_w 0.25 \
  --xcam_blocks 5,8 \
  --part_xcam_w 0.08 \
  --part_xcam_temp 0.07 \
  --part_xcam_same_cam_w 0.10
```

## Evaluation

### Standard Evaluation

```bash
python VID_Test.py \
  --Dataset_name Mars \
  --model_path ./outputs_partxcam_bs64/Mars_partxcam_best.pth
```

Replace `Mars` and the checkpoint path with the corresponding dataset and saved checkpoint for PRID2011 or iLIDS-VID.

### MARS Flip-Test Evaluation

```bash
python VID_Test.py \
  --Dataset_name Mars \
  --model_path ./outputs_partxcam_bs64/Mars_partxcam_best.pth \
  --flip_test \
  --rerank
```

The main MARS result reported in the paper uses the original distance result from flip-test evaluation, not the re-ranked Rank-1 result.

## Reported Results

| Dataset | Rank-1 (%) | Rank-5 (%) | mAP (%) | Notes |
|---|---:|---:|---:|---|
| MARS | 89.4 | 96.8 | 86.4 | Flip-test original distance result |
| PRID2011 | 96.6 | 98.9 | 97.6 | Best checkpoint result |
| iLIDS-VID | 90.0 | 100.0 | 94.4 | Best checkpoint result |

## Method Summary

XCamFormer uses a ViT-based video representation pipeline with global video-level features and local patch-part features. The model removes camera embeddings from the input sequence. To compensate for this removal, it introduces two training-only objectives:

1. **Intermediate Cross-Camera Supervision (ICCS):** applies cross-camera supervised contrastive learning to selected intermediate transformer block features.
2. **Part-Level Cross-Camera Supervision (PLCCS):** applies cross-camera supervision to four local part features.

Camera labels are used only to construct training-time positive relationships. During inference, no identity labels, camera labels, or camera metadata are provided to the model.

## Notes

- `--xcam_blocks 5,8` uses zero-based transformer block indexing.
- Same-camera same-identity pairs are not discarded; they are kept with a smaller weight than cross-camera same-identity pairs.
- Dataset paths are intentionally left inside dataset loader files and should be updated by users according to their local setup.

## Citation

If you use this repository, please cite the corresponding XCamFormer paper when available.
