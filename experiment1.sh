#!/bin/bash

source activate nlp1
python main.py --epochs 100 --cuda --plot --lr 0.01 --lamb 0.1
python main.py --epochs 100 --cuda --plot --lr 0.05 --lamb 0.1
python main.py --epochs 100 --cuda --plot --lr 0.1 --lamb 0.1
python main.py --epochs 100 --cuda --plot --lr 0.01 --lamb 0
python main.py --epochs 100 --cuda --plot --lr 0.05 --lamb 0
python main.py --epochs 100 --cuda --plot --lr 0.1 --lamb 0
python main.py --epochs 100 --cuda --plot --lr 0.01 --lamb 0.5
python main.py --epochs 100 --cuda --plot --lr 0.05 --lamb 0.5
python main.py --epochs 100 --cuda --plot --lr 0.1 --lamb 0.5
python main_last.py --epochs 400 --cuda --plot --lr 0.5 --lamb 0.1
python main_last.py --epochs 400 --cuda --plot --lr 0.05 --lamb 0.1
python main_last.py --epochs 1000 --cuda --plot --lr 0.1 --lamb 0.1
python main_last.py --epochs 400 --cuda --plot --lr 0.5 --lamb 0
python main_last.py --epochs 400 --cuda --plot --lr 0.05 --lamb 0
python main_last.py --epochs 400 --cuda --plot --lr 0.1 --lamb 0
python main_last.py --epochs 400 --cuda --plot --lr 0.5 --lamb 0.5
python main_last.py --epochs 400 --cuda --plot --lr 0.05 --lamb 0.5
python main_last.py --epochs 400 --cuda --plot --lr 0.1 --lamb 0.5