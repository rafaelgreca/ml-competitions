#! /usr/bin/python
input_dir="/media/greca/HD/Datasets/dogs-vs-cats"
output_dir="./checkpoints"
epochs=100
learning_rate=1e-03
batch_size=32
model_name="resnet"
scheduler_gamma=1e-03
tolerance=20
scheduler_step=10

python3 train.py --output_dir=$output_dir \
                 --input_dir=$input_dir \
                 --lr=$learning_rate \
                 --epochs=$epochs \
                 --batch_size=$batch_size \
                 --model_name=$model_name \
                 --scheduler_gamma=$scheduler_gamma \
                 --tolerance=$tolerance \
                 --scheduler_step=$scheduler_step
