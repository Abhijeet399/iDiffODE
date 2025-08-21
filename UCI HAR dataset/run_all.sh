#!/bin/bash

for model in mlp rnn transformer
do
    echo "Training $model..."
    python train.py --model $model --epochs 50 --batch-size 64 --learning-rate 0.001
done

