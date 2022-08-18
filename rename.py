import os

def rename():
    # 原始图片路径
    # path = r'E:\04-Data\02-robot\05-Xingqiao0708\test1(2022-07-08 161940)'
    path=r'E:\04-Data\02-robot\07-Xingqiao-26A\0725\26A(2022-07-25 160923)\2D'
    # 获取该路径下所有图片
    filelist = os.listdir(path)
    a = 1
    for files in filelist:
        # 原始路径
        Olddir = os.path.join(path, files)

        # if os.path.isdir(Olddir):
        #	continue
        # 将图片名切片,比如 xxx.bmp 切成xxx和.bmp
        # xxx
        print(files)
        filename = files.split()[0][:-4]
        print(filename)
        # 需要存储的路径 a 是需要定义修改的文件名
        Newdir = os.path.join(path, filename + '.jpg')
        os.rename(Olddir, Newdir)
        a += 1

rename()
