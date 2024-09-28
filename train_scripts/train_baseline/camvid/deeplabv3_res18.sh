python3 train_baseline.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --aux \
    --dataset camvid \
    --crop-size 360 480 \
    --data data/CamVid/ \
    --save-dir save/ \
    --log-dir save/ \
    --pretrained-base resnet18-imagenet.pth


python3 eval.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --dataset camvid \
    --data data/CamVid/ \
    --data-list data/CamVid/test.txt \
    --crop-size 480 360 \
    --pretrained save/deeplabv3_resnet18_camvid_best_model.pth \


# python3 eval.py --model deeplabv3 --backbone resnet101 --dataset camvid --data ../../semantic-seg-distillation/data/CamVid/ --data-list ../../semantic-seg-distillation/data/CamVid/test.txt --crop-size 480 360 --pretrained ../../semantic-seg-distillation/deeplabv3_resnet101_cirkd.pth