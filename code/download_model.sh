#!/bin/bash
wget https://chunylcus.blob.core.windows.net/machines/msrdl/optimus/output/pretrain/philly_rr3_vc4_g8_base_vae_wikipedia_pretraining_beta_schedule_beta0.5_d1.0_ro0.5_ra0.25_"$LATENT_SIZE"_v2/checkpoint-508523.zip 
mkdir ./optimus_model
unzip checkpoint-508523.zip -d ./optimus_model
rm checkpoint-508523.zip 
