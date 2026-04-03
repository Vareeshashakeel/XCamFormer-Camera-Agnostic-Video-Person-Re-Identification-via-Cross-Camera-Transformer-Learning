# VID-Trans-ReID camera-agnostic intermediate + part-XCam version

This repo extends the strongest intermediate XCam version with **part-level cross-camera supervised contrastive learning**.

## Main idea
The previous strongest repo aligned same-ID / different-camera features at intermediate ViT blocks. This version keeps that idea and adds the same cross-camera supervision on the **four local VPPF part features**.

## What changed
- no camera metadata is used as model input
- camera labels are used only to build training positives in the loss
- intermediate transformer tokens from selected blocks are aggregated into sequence-level features
- cross-camera same-ID features are pulled closer with supervised contrastive loss at two levels:
  - intermediate ViT sequence features
  - final local part features (4 VPPF parts)
- inference remains camera-agnostic

## Default strong setting
- xcam blocks: `5,8`
- xcam weight: `0.15`
- xcam temperature: `0.07`
- same-camera fallback weight: `0.25`
- part xcam weight: `0.10`
- part xcam temperature: `0.07`
- part same-camera fallback weight: `0.10`

## Train example
```bash
python VID_Trans_ReID.py \
  --Dataset_name Mars \
  --model_path /path/to/jx_vit_base_p16_224-80ecf9dd.pth \
  --batch_size 16 \
  --epochs 40 \
  --eval_every 10 \
  --num_workers 2 \
  --output_dir ./outputs_partxcam \
  --xcam_blocks 5,8 \
  --xcam_w 0.15 \
  --part_xcam_w 0.10
```
