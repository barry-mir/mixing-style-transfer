#!/bin/bash
#
# Run validation without using cache (recompute all embeddings)
# Use this for debugging or if you've updated the model
#

python ../inference/validate_retrieval.py \
  --checkpoint /nas/mixing-representation/checkpoints_baseline/best_model.pt \
  --data_path /ssd2/barry/fma_full/ \
  --separated_path /ssd2/barry/fma_separated_cropped/ \
  --test_dir /nas/music2preset_testcase/ \
  --output_dir validation_results/ \
  --cache_dir validation_results/embeddings_cache/ \
  --val_split 0.1 \
  --seed 42 \
  --device cuda \
  --scnet_model Music-Source-Separation-Training/model_scnet_masked_ep_111_sdr_9.8286.ckpt \
  --scnet_config Music-Source-Separation-Training/configs/config_musdb18_scnet_xl_ihf.yaml

# Note: --use_cache is removed to force recomputation
