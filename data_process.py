import os


def pre_process():
    test_path = './trainval/'
    marker_file = 'groundtruth.txt'
    root_dirs = os.listdir(test_path)
    name = [item for item in root_dirs]
    size = []
    locators = []
    root_dirs = [test_path + item + "/" for item in root_dirs]
    for dir in root_dirs:
        marker = dir + marker_file
        with open(marker, encoding='utf-8', mode='r') as f:
            data = f.readline()
            f.close()
        locator = [int(float(item)) for item in data.strip("\n").split(",")]
        locators.append(locator)
        files = os.listdir(dir)
        size.append(len(files) - 1)
    #     预处理文件，得到每一个视频的名字、帧数和标志框体
    return name, root_dirs, size, locators
