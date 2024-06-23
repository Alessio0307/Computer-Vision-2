python tools/train.py --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml &&
mv emissions/emissions.csv emissions/w32_256x192_adam_lr1e-3_emissions.csv &&
python tools/train.py --cfg experiments/coco/hrnet+edsr/s2_w32_256x192_adam_lr1e-3.yaml &&
mv emissions/emissions.csv emissions/s2_w32_256x192_adam_lr1e-3_emissions.csv &&
python tools/train.py --cfg experiments/coco/hrnet+edsr/s3_w32_256x192_adam_lr1e-3.yaml &&
mv emissions/emissions.csv emissions/s3_w32_256x192_adam_lr1e-3_emissions.csv &&
echo "all trainings done" > done
