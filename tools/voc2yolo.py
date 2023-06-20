import os
import os.path
import xml.etree.ElementTree as ET

class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

dirpath = r'./train/labels'  # 原来存放xml文件的目录
newdir = r'./train/labels'  # 修改label后形成的txt目录

if not os.path.exists(newdir):
    os.makedirs(newdir)

for fp in os.listdir(dirpath):
    file_path = os.path.join(dirpath, fp)

    root = ET.parse(file_path).getroot()

    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    sz = root.find('size')
    width = float(sz[0].text)
    height = float(sz[1].text)
    filename = root.find('filename').text
    for child in root.findall('object'):  # 找到图片中的所有框
        name = child.find('name').text  # 找到类别名
        class_num = class_names.index(name)  #

        sub = child.find('bndbox')  # 找到框的标注值并进行读取
        xmin = float(sub[0].text)
        xmax = float(sub[1].text)
        ymin = float(sub[2].text)
        ymax = float(sub[3].text)
        try:  # 转换成yolov3的标签格式，需要归一化到（0-1）的范围内
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
        except ZeroDivisionError:
            print(f"{filename} width error")

        with open(os.path.join(newdir, fp[:-4] + '.txt'), 'a+') as f:
            f.write(' '.join([str(class_num), str(x_center), str(y_center), str(w), str(h) + '\n']))

    os.remove(file_path)



# Write TXT
train_images_path = "./train/images"

train_txt_path = "./train.txt"

train_files = os.listdir(train_images_path)
with open(train_txt_path, "w") as f:
    for image in train_files:
        f.write(f"{train_images_path}/{image}" + "\n")
