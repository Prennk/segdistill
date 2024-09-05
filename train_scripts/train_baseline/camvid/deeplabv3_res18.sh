python3 train_baseline.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --dataset camvid \
    --crop-size 480 360 \
    --data data/CamVid/ \
    --save-dir save/ \
    --log-dir save/ \
    --pretrained resnet18-imagenet.pth


# CUDA_VISIBLE_DEVICES=0 \
#   python -m torch.distributed.launch --nproc_per_node=1 \
#   eval.py \
#   --model deeplabv3 \
#   --backbone resnet18 \
#   --dataset camvid \
#   --data [your dataset path]/CamVid/ \
#   --save-dir [your directory path to store checkpoint files] \
#   --pretrained [your pretrained model path]

# python3 eval.py --model deeplabv3 --backbone resnet101 --dataset camvid --data ../../semantic-seg-distillation/data/CamVid/ --data-list ../../semantic-seg-distillation/data/CamVid/test.txt --crop-size 480 360 --pretrained ../../semantic-seg-distillation/deeplabv3_resnet101_cirkd.pth