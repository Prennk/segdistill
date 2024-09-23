python3 eval.py \
    --model deeplabv3 \
    --backbone resnet101 \
    --dataset camvid \
    --data data/CamVid/ \
    --data-list data/CamVid/test.txt \
    --crop-size 480 360 \
    --pretrained save/deeplabv3_resnet18_camvid_best_model.pth \