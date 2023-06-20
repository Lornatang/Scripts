import zlib
import os
import torch

def crc(fileName):
    prev = 0
    for eachLine in open(fileName,"rb"):
        prev = zlib.crc32(eachLine, prev)
    return "%X"%(prev & 0xFFFFFFFF)


for file_name in os.listdir("."):
    if file_name.endswith(".tar"):
        os.rename(file_name, file_name[:-8] + "-" +crc(file_name).lower() + ".pth.tar")


for file_name in os.listdir("."):
    if file_name.endswith(".tar"):
        torch.save({"state_dict": torch.load(file_name)["state_dict"]}, file_name)