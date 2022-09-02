# Decompress Ali Tianchi Street View Character Detection Dataset
tar -xvf ./VOC0712/VOCtrainval_06-Nov-2007.tar -C ./VOC0712/
tar -xvf ./VOC0712/VOCtrainval_11-May-2012.tar -C ./VOC0712/
tar -xvf ./VOC0712/VOCtest_06-Nov-2007.tar -C ./VOC0712/

# Extract the tag information in the Json file and form a character detection dataset
mkdir -p ./VOC0712/images/train
mkdir -p ./VOC0712/images/test
mkdir -p ./VOC0712/labels/train
mkdir -p ./VOC0712/labels/test

# shellcheck disable=SC2164
cd ./VOC0712
python3 voc_label.py
# shellcheck disable=SC2103
cd ..

# Delete residual files in the process of making datasets
rm -rf ./VOC0712/2007_test.txt
# shellcheck disable=SC2035
rm -rf ./VOC0712/2007_train.txt
# shellcheck disable=SC2035
rm -rf ./VOC0712/2007_val.txt
# shellcheck disable=SC2035
rm -rf ./VOC0712/2012_train.txt
# shellcheck disable=SC2035
rm -rf ./VOC0712/2012_val.txt
# shellcheck disable=SC2035
rm -rf ./VOC0712/*.tar
rm -rf ./VOC0712/VOCdevkit
