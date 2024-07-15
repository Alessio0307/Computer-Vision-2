python tools/train.py --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-4.yaml &&
mv emissions/emissions.csv emissions/w32_256x192_adam_lr1e-4_emissions.csv &&
echo "First training done" > done1 &&
python tools/train.py --cfg experiments/coco/hrnet+edsr/s2_l1_w32_256x192_adam_lr1e-4.yaml &&
mv emissions/emissions.csv emissions/s2_l1_w32_256x192_adam_lr1e-4_emissions.csv &&
rm done1 &&
echo "Second training done" > done2 &&
python tools/train.py --cfg experiments/coco/hrnet+edsr/s2_l2_w32_256x192_adam_lr1e-4.yaml &&
mv emissions/emissions.csv emissions/s2_l2_w32_256x192_adam_lr1e-4_emissions.csv &&
rm done2 &&
echo "Third training done" > done3 &&
python tools/train.py --cfg experiments/coco/hrnet+edsr/s3_l1_w32_256x192_adam_lr1e-4.yaml &&
mv emissions/emissions.csv emissions/s3_l1_w32_256x192_adam_lr1e-4_emissions.csv &&
rm done3 &&
echo "Fourth training done" > done4 &&
python tools/train.py --cfg experiments/coco/hrnet+edsr/s3_l2_w32_256x192_adam_lr1e-4.yaml &&
mv emissions/emissions.csv emissions/s3_l2_w32_256x192_adam_lr1e-4_emissions.csv &&
rm done4 &&
echo "Fifth training done" > done5 &&
python tools/train.py --cfg experiments/coco/hrnet+edsr/s3_l3_w32_256x192_adam_lr1e-4.yaml &&
mv emissions/emissions.csv emissions/s3_l3_w32_256x192_adam_lr1e-4_emissions.csv &&
rm done5 &&
echo "all trainings done" > done
