import os
import shutil


# 创建文件夹
def Create_folder(filename):
    filename = filename.strip()
    filename = filename.rstrip("\\")
    isExists = os.path.exists(filename)

    if not isExists:
        os.makedirs(filename)
        print(filename + "创建成功")
        return True
    else:
        print(filename + "已存在")
        return False


def Copyfile(srcfile, dstpath):  # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, dstpath + "/" + fname)  # 复制文件
        print("copy %s to %s" % (srcfile, dstpath + "/" + fname))
