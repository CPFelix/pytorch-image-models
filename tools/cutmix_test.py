from PIL import Image
import os
from timm.data.auto_augment import rotate
from tqdm import tqdm
import csv
import random

# 读取csv获取烟区域
def read_csv(imgpath, savepath, csvfile):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    with open(csvfile, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)
            imgfile = os.path.join(imgpath, row[0])
            img = Image.open(imgfile)
            crop_xy = (int(row[1]), int(row[2]), int(row[3]), int(row[4]))
            crop_img = img.crop(crop_xy)
            newImgname = row[0].split(".")[0] + "_crop.jpg"
            savefile = os.path.join(savepath, newImgname)
            crop_img.save(savefile, quality=100)


def cutmix(baseImgPath, smokeImgPath, savePath):
    radio = 5
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    for imgname in tqdm(os.listdir(baseImgPath)):
        baseimgfile = os.path.join(baseImgPath, imgname)
        baseImg = Image.open(baseimgfile)
        size1 = baseImg.size

        smoke_list = os.listdir(smokeImgPath)
        select_index = random.randint(0, len(smoke_list) - 1)
        smokeimgfile = os.path.join(smokeImgPath, smoke_list[select_index])
        smoke_img = Image.open(smokeimgfile)
        size2 = smoke_img.size

        center = (int(size1[0] / 2), int(size1[1] / 2))

        nums = 0
        flag = False
        while((size1[0] < size2[0]) or (size1[1] < size2[1]) or (size2[0] < 20 or size2[1] < 20)):
            nums += 1
            if (nums > 100):
                flag = True
                break
            select_index = random.randint(0, len(smoke_list) - 1)
            smokeimgfile = os.path.join(smokeImgPath, smoke_list[select_index])
            smoke_img = Image.open(smokeimgfile)
            size2 = smoke_img.size
        if (flag):
            print(imgname)
            continue

        crop_x1 = center[0] - int(size2[0] / 2)
        crop_y1 = center[1] - int(size2[1] / 2)
        crop_x2 = center[0] + (size2[0] - int(size2[0] / 2))
        crop_y2 = center[1] + (size2[1] - int(size2[1] / 2))
        baseImg.paste(smoke_img, (crop_x1, crop_y1))

        # baseImg.show()

        newFilename = imgname.split(".")[0] + "+" + smoke_list[select_index]
        saveImgPath = os.path.join(savePath, newFilename)
        baseImg.save(saveImgPath, quality=100)







if __name__ == "__main__":
    # imgpath = "E:\\pycharm-projects\\dataset\\smake_cover\\sigurate_shujutang\\part1"
    # savepath = "E:\\pycharm-projects\\dataset\\DSMhands6\\smokeArea\\sigurate_shujutang"
    # csvfile = "E:\\pycharm-projects\\dataset\\smake_cover\\sigurate_shujutang\\sigurate_shujutang.csv"
    #
    # read_csv(imgpath, savepath, csvfile)
    baseImgPath = "/home/chenpengfei/dataset/DSMhands5/val/fake_sigurate"
    smokeImgPath = "/home/chenpengfei/dataset/DSMhands6/smokeArea"
    savePath = "/home/chenpengfei/dataset/DSMhands6/val/sigurate_cutmix"
    cutmix(baseImgPath, smokeImgPath, savePath)

    '''
    # 旋转
    degree = -15
    img_r = rotate(img, degree)  #, expand=True
    newImgname = imgname.split(".")[0] + "_r" + str(degree) + ".jpg"
    saveImgpath = os.path.join(savepath, newImgname)
    img_r.save(saveImgpath, quality=100)
    '''