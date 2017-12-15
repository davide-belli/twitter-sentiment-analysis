#!/bin/bash

source activate nlp1

echo "--epochs 100 --lr 0.01 --lamb 0.1 --model RAN --batch_size 100 1"
python main.py --plot --epochs 100 --lr 0.01 --lamb 0.1 --model RAN --batch_size 20
echo "--epochs 100 --lr 0.01 --lamb 0.1 --model RAN --batch_size 100 2"
python main.py --plot --epochs 100 --lr 0.01 --lamb 0.1 --model RAN_BIDIR --batch_size 20 
echo "--epochs 100 --lr 0.01 --lamb 0.1 --model RAN --batch_size 100 3"
python main.py --plot --epochs 100 --lr 0.01 --lamb 0.1 --model RAN --batch_size 100 
echo "--epochs 100 --lr 0.01 --lamb 0.1 --model RAN --batch_size 100 4"
python main.py --plot --epochs 100 --lr 0.01 --lamb 0.1 --model RAN_BIDIR --batch_size 100 
echo "--epochs 100 --lr 0.01 --lamb 0.1 --model RAN --batch_size 100 5"
python main.py --plot --epochs 400 --lr 0.001 --lamb 0.1 --model RAN --batch_size 100 
echo "--epochs 100 --lr 0.01 --lamb 0.1 --model RAN --batch_size 100 6"
python main.py --plot --epochs 400 --lr 0.001 --lamb 0.1 --model RAN_BIDIR --batch_size 100 
