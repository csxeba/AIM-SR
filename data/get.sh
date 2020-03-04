#! /bin/bash

wget https://aimotive.com/share/traindata/traffic_signs/train-dataset-52x52.zip
echo "Unzipping data..."
unzip train-dataset-52x52.zip > /dev/null
rm train-dataset-52x52.zip
