#!/bin/bash

source activate nlp1

echo "--epochs 100 --lr 0.01 --lamb 0.5 --model LSTM_REV --batch_size 20"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.5 --model LSTM_REV --batch_size 20
echo "--epochs 100 --lr 0.01 --lamb 0.5 --model LSTM --batch_size 20"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.5 --model LSTM --batch_size 20
echo "--epochs 100 --lr 0.01 --lamb 0.5 --model LSTM_BIDIR --batch_size 20"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.5 --model LSTM_BIDIR --batch_size 20
echo "--epochs 100 --lr 0.1 --lamb 0.5 --model LSTM_REV --batch_size 20"
python main.py --cuda --plot --epochs 100 --lr 0.1 --lamb 0.5 --model LSTM_REV --batch_size 20
echo "--epochs 100 --lr 0.1 --lamb 0.5 --model LSTM --batch_size 20"
python main.py --cuda --plot --epochs 100 --lr 0.1 --lamb 0.5 --model LSTM --batch_size 20
echo "second experiment --epochs 100 --lr 0.1 --lamb 0.5 --model LSTM --batch_size 20"
python main.py --cuda --plot --epochs 100 --lr 0.1 --lamb 0.5 --model LSTM --batch_size 20
echo "--epochs 100 --lr 0.1 --lamb 0.5 --model LSTM_BIDIR --batch_size 20"
python main.py --cuda --plot --epochs 100 --lr 0.1 --lamb 0.5 --model LSTM_BIDIR --batch_size 20
echo "--epochs 200 --lr 0.002 --lamb 0.5 --model LSTM --batch_size 20"
python main.py --cuda --plot --epochs 200 --lr 0.002 --lamb 0.5 --model LSTM --batch_size 20
echo "--last --epochs 1000 --lr 0.1 --lamb 0.5 --model LSTM --batch_size 100"
python main.py --cuda --plot --last --epochs 1000 --lr 0.1 --lamb 0.5 --model LSTM_REV --batch_size 100
echo "--last --epochs 1000 --lr 0.1 --lamb 0.5 --model LSTM --batch_size 100"
python main.py --cuda --plot --last --epochs 1000 --lr 0.1 --lamb 0.5 --model LSTM_BIDIR --batch_size 100
echo "--last --epochs 1000 --lr 0.1 --lamb 0.5 --model LSTM --batch_size 100"
python main.py --cuda --plot --last --epochs 1000 --lr 0.1 --lamb 0.5 --model LSTM --batch_size 100