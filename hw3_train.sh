#!/bin/bash

if [ $# -ne 1 ]; then
  echo -e "usage:\tbash hw3_train.sh data_directory"
  exit
fi

DIRECTORY='data/'
if [ ! -d "$DIRECTORY" ]; then
  mkdir "$DIRECTORY"
fi

python3 process_data.py $1

echo "start training..."
echo "model 0"
python3 hw3_train_0.py
echo "model 1"
python3 hw3_train_1.py
echo "model 2"
python3 hw3_train_2.py
echo "model 3"
python3 hw3_train_3.py
echo "model 4"
python3 hw3_train_4.py
echo "model 5"
python3 hw3_train_5.py
echo "model 6"
python3 hw3_train_6.py
