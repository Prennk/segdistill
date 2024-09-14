python3 train_kd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --max-iterations 15000 \
    --crop-size 480 360 \
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
    --pretrained save/kd_deeplabv3_resnet18_camvid_best_model.pth \