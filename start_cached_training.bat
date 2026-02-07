@echo off
REM Start training with cached targets (100x speedup vs parsing SVG)
REM Prerequisites: run precompute_floorplancad_targets.py first

cd /d E:\dv\DeepV

REM Wait for cache to finish (optional, just for clarity)
echo.
echo ===============================================================
echo Training with Cached Targets
echo ===============================================================
echo.
echo Model: Non-Autoregressive Transformer Decoder
echo Dataset: FloorPlanCAD (train:8344, val:1281)
echo Batch Size: 4 (adjusted to fit RTX 3060 12GB VRAM)
echo Max Primitives: 20
echo Epochs: 20
echo GPU: NVIDIA RTX 3060
echo.
echo Expected epoch time: 1-2 minutes (vs 27 hours with SVG parsing)
echo Expected total time: 20-40 minutes for 20 epochs
echo.
echo ===============================================================
echo.

python scripts/train_floorplancad.py ^
  --model-spec vectorization/models/specs/non_autoregressive_resnet18_blocks1_bn_64__c2h__trans_heads4_feat256_blocks8_ffmaps512__h2o__out6.json ^
  --cache-dir data/cache/floorplancad ^
  --epochs 20 ^
  --batch-size 4 ^
  --max-primitives 20 ^
  --num-workers 2
