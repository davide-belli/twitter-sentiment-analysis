#!/bin/bash

source activate nlp1

echo "--epochs 100 --lr 0.1 --lamb 0.5 --model LSTM --batch_size 20"
python main.py --cuda --plot --epochs 100 --lr 0.1 --lamb 0.5 --model LSTM --batch_size 20
echo "--epochs 100 --lr 0.01 --lamb 0 --model LSTM --batch_size 20"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0 --model LSTM --batch_size 20
echo "--epochs 100 --lr 1 --lamb 0.1 --model LSTM --batch_size 20"
python main.py --cuda --plot --epochs 100 --lr 1 --lamb 0.1 --model LSTM --batch_size 20
echo "--epochs 100 --lr 0.1 --lamb 1 --model LSTM --batch_size 20"
python main.py --cuda --plot --epochs 100 --lr 0.1 --lamb 1 --model LSTM --batch_size 20
echo "--epochs 100 --lr 5 --lamb 0.5 --model LSTM --batch_size 20"
python main.py --cuda --plot --epochs 100 --lr 5 --lamb 0.5 --model LSTM --batch_size 20
echo "--epochs 100 --lr 0.1 --lamb 0.5 --model LSTM --batch_size 40"
python main.py --cuda --plot --epochs 100 --lr 0.1 --lamb 0.5 --model LSTM --batch_size 40
echo "--epochs 100 --lr 0.1 --lamb 0.5 --model LSTM --batch_size 100"
python main.py --cuda --plot --epochs 100 --lr 0.1 --lamb 0.5 --model LSTM --batch_size 100
echo "--epochs 100 --lr 0.1 --lamb 0.5 --model GRU --batch_size 20"
python main.py --cuda --plot --epochs 100 --lr 0.1 --lamb 0.5 --model GRU --batch_size 20
echo "--epochs 100 --lr 0.1 --lamb 0 --model GRU --batch_size 20"
python main.py --cuda --plot --epochs 100 --lr 0.1 --lamb 0 --model GRU --batch_size 20
echo "--epochs 100 --lr 0.1 --lamb 0.5 --model RNN_TANH --batch_size 20"
python main.py --cuda --plot --epochs 100 --lr 0.1 --lamb 0.5 --model RNN_TANH --batch_size 20
echo "--epochs 100 --lr 0.1 --lamb 0 --model RNN_TANH --batch_size 20"
python main.py --cuda --plot --epochs 100 --lr 0.1 --lamb 0 --model RNN_TANH --batch_size 20
echo "--epochs 800 --lr 5 --lamb 0 --model LSTM --batch_size 100"
python main_last.py --cuda --plot --epochs 800 --lr 0.5 --lamb 0 --model LSTM --batch_size 100
echo "--epochs 800 --lr 5 --lamb 1 --model LSTM --batch_size 100"
python main_last.py --cuda --plot --epochs 800 --lr 0.1 --lamb 1 --model LSTM --batch_size 100
echo "--epochs 800 --lr 20 --lamb 0.5 --model LSTM --batch_size 20"
python main_last.py --cuda --plot --epochs 800 --lr 0.1 --lamb 0.5 --model LSTM --batch_size 20
echo "--epochs 100 --lr 0.1 --lamb 0.6 --model LSTM --batch_size 20 --hidden 4"
python main.py --cuda --plot --epochs 100 --lr 0.1 --lamb 0.5 --model LSTM --batch_size 20