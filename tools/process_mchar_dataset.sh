# Decompress Ali Tianchi Street View Character Detection Dataset
unzip ./TIANCHI-mchar/mchar_train.zip -d ./TIANCHI-mchar
unzip ./TIANCHI-mchar/mchar_val.zip -d ./TIANCHI-mchar
unzip ./TIANCHI-mchar/mchar_test_a.zip -d ./TIANCHI-mchar

# Extract the tag information in the Json file and form a character detection dataset
mkdir -p ./TIANCHI-mchar/images/train
mkdir -p ./TIANCHI-mchar/images/valid
mkdir -p ./TIANCHI-mchar/labels/train
mkdir -p ./TIANCHI-mchar/labels/valid

cp -r ./TIANCHI-mchar/mchar_train/* ./TIANCHI-mchar/images/train
cp -r ./TIANCHI-mchar/mchar_val/* ./TIANCHI-mchar/images/valid

# shellcheck disable=SC2164
cd ./TIANCHI-mchar
python3 json2yolo.py --train_images_path images/train --valid_images_path images/valid --train_labels_path labels/train --valid_labels_path labels/valid --train_json_path mchar_train.json --valid_json_path mchar_val.json --train_txt_path train.txt --valid_txt_path valid.txt
# shellcheck disable=SC2103
cd ..

# Delete residual files in the process of making datasets
rm -rf ./TIANCHI-mchar/__MACOSX
# shellcheck disable=SC2035
rm -rf ./TIANCHI-mchar/*.csv
# shellcheck disable=SC2035
rm -rf ./TIANCHI-mchar/*.json
# shellcheck disable=SC2035
rm -rf ./TIANCHI-mchar/*.zip
rm -rf ./TIANCHI-mchar/mchar_train
rm -rf ./TIANCHI-mchar/mchar_val
rm -rf ./TIANCHI-mchar/mchar_test_a
