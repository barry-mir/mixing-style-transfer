#!/bin/bash
#
# Run in-domain validation only (validation set retrieval accuracy)
# Faster if you don't need out-of-domain results
#

python ../inference/validate_retrieval.py \
  --checkpoint /nas/mixing-representation/checkpoints_baseline/best_model.pt \
  --data_path /ssd2/barry/fma_full/ \
  --separated_path /ssd2/barry/fma_separated_cropped/ \
  --test_dir /tmp/empty_test_dir \
  --output_dir validation_results/ \
  --cache_dir validation_results/embeddings_cache/ \
  --use_cache \
  --val_split 0.1 \
  --seed 42 \
  --device cuda \
  --scnet_model Music-Source-Separation-Training/model_scnet_masked_ep_111_sdr_9.8286.ckpt \
  --scnet_config Music-Source-Separation-Training/configs/config_musdb18_scnet_xl_ihf.yaml
