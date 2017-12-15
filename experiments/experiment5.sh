#!/bin/bash

source activate nlp1


echo "--last --epochs 1000 --lr 0.1 --lamb 0.1 --model LSTMrev --batch_size 100"
python main.py --cuda --plot --last --epochs 1000 --lr 0.1 --lamb 0.1 --model LSTM_REV --batch_size 100 --pre
echo "--last --epochs 1000 --lr 0.1 --lamb 0.01 --model LSTMrev --batch_size 100"
python main.py --cuda --plot --last --epochs 1000 --lr 0.01 --lamb 0.1 --model LSTM_REV --batch_size 100 --pre
echo "--last --epochs 1000 --lr 0.1 --lamb 0.1 --model LSTMbidir --batch_size 100"
python main.py --cuda --plot --last --epochs 1000 --lr 0.1 --lamb 0.1 --model LSTM_BIDIR --batch_size 100 --pre
echo "--last --epochs 1000 --lr 0.1 --lamb 0.1 --model LSTM --batch_size 100"
python main.py --cuda --plot --last --epochs 1000 --lr 0.1 --lamb 0.1 --model LSTM --batch_size 100 --pre

echo "--last --epochs 1000 --lr 0.1 --lamb 0.1 --model LSTMbidir --batch_size 100"
python main.py --cuda --plot --last --epochs 1000 --lr 0.01 --lamb 0.1 --model LSTM_BIDIR --batch_size 100 --pre
echo "--last --epochs 1000 --lr 0.1 --lamb 0.1 --model LSTM --batch_size 100"
python main.py --cuda --plot --last --epochs 1000 --lr 0.01 --lamb 0.1 --model LSTM --batch_size 100 --pre
echo "--last --epochs 1000 --lr 0.1 --lamb 0.1 --model LSTMrev --batch_size 100"
python main.py --cuda --plot --last --epochs 1500 --lr 0.005 --lamb 0.1 --model LSTM_REV --batch_size 100 --pre
echo "--last --epochs 1000 --lr 0.1 --lamb 0.01 --model LSTMrev --batch_size 100"
python main.py --cuda --plot --last --epochs 2000 --lr 0.002 --lamb 0.1 --model LSTM_REV --batch_size 100 --pre