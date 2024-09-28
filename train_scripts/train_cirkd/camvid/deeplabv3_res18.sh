python3 train_cirkd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --dataset camvid \
    --batch-size 8 \
    --crop-size 360 480 \
    --data data/CamVid/ \
    --save-dir save/ \
    --log-dir save/ \
    --teacher-pretrained deeplabv3_resnet101_cirkd.pth \
    --student-pretrained-base resnet18-imagenet.pth \
    --resume save/kd_deeplabv3_resnet18_camvid.pth


python3 eval.py \
  --model deeplabv3 \
  --backbone resnet18 \
  --dataset camvid \
  --data data/CamVid/ \
  --save-dir save/ \
  --pretrained save/kd_deeplabv3_resnet18_camvid_best_model.pth