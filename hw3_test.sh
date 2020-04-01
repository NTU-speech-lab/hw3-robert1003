#!/bin/bash

if [ $# -ne 2 ]; then
  echo -e "usage:\tbash hw3_test.sh data_directory prediction_file"
  exit
fi

m0=140.112.90.197:10297/hw3/model0.ckpt
m1=140.112.90.197:10297/hw3/model1.ckpt
m2=140.112.90.197:10297/hw3/model2.ckpt
m3=140.112.90.197:10297/hw3/model3.ckpt
m4=140.112.90.197:10297/hw3/model4.ckpt
m5=140.112.90.197:10297/hw3/model5.ckpt
m6=140.112.90.197:10297/hw3/model6.ckpt

wget "${m0}" -O ./model0.ckpt
wget "${m1}" -O ./model1.ckpt
wget "${m2}" -O ./model2.ckpt
wget "${m3}" -O ./model3.ckpt
wget "${m4}" -O ./model4.ckpt
wget "${m5}" -O ./model5.ckpt
wget "${m6}" -O ./model6.ckpt

python3 hw3_test.py $1 $2
