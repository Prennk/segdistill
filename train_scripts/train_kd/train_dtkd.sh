python3 train_dtkd.py \
    --max-iterations 20000 \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --kd-temperature 5.0 \
    --lambda-kd 0.9 \
    --crop-size 360 480 \
    --dataset camvid \
    --data data/CamVid/ \
    --save-dir save/ \
    --log-dir save/ \
    --teacher-pretrained deeplabv3_resnet101_cirkd.pth \
    --student-pretrained-base resnet18-imagenet.pth

python3 eval.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --dataset camvid \
    --data data/CamVid/ \
    --data-list data/CamVid/test.txt \
    --crop-size 480 360 \
    --pretrained save/dtkd_deeplabv3_resnet18_camvid_best_model.pth