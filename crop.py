from PIL import Image
import os.path, sys

path = "set5/Set5/"
path1 = "set5/Set5_cropped/"
dirs = os.listdir(path)
print(dirs)
def crop():
    for item in dirs:
        fullpath = os.path.join(path,item)         #corrected
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            imCrop = im.crop((18, 32, 210, 224)) #corrected
            imCrop.save(path1 + item, "PNG", quality=100)

crop()