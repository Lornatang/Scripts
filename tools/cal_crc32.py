import zlib
import os

def crc(fileName):
    prev = 0
    for eachLine in open(fileName,"rb"):
        prev = zlib.crc32(eachLine, prev)
    return "%X"%(prev & 0xFFFFFFFF)


for file_name in os.listdir("."):
    if file_name.endswith(".tar"):
        os.rename(file_name, file_name[:-8] + "-" +crc(file_name).lower() + ".pth.tar")
