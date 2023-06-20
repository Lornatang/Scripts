import os
import os.path
import xml.dom.minidom
 
path="./labels"

for xmlFile in os.listdir(path): #遍历文件夹
    xml_file_path = os.path.join(path,xmlFile)
	#将获取的xml文件名送入到dom解析
    dom=xml.dom.minidom.parse(xml_file_path)  #最核心的部分,路径拼接,输入的是具体路径
    root=dom.documentElement
    #获取标签对name/pose之间的值
    file_name=root.getElementsByTagName('filename')
    #原始信息
    info=file_name[0]
	#修改
    info.firstChild.data=xmlFile[:-4] + ".jpg"
    with open(xml_file_path[:-4] + ".xml",'w') as fh:
        dom.writexml(fh)
    
for file in os.listdir(path): # 遍历文件夹
    # 删除txt文件
    if file[-4:] == ".txt":
        os.remove(os.path.join(path, file))
