import os
from shutil import copyfile

src_path = './trainval/'
store_path = './src_result/'
maker_name = 'groundtruth.txt'

vedios = os.listdir(src_path)
for vedio in vedios:
    copyfile(src_path + vedio + "/" + maker_name, store_path + vedio + ".txt")

