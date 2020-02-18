import os
from skimage import data, io, filters
import imageio
import numpy as np
from numpy import array
from skimage.transform import rescale, resize
from scipy.misc import imresize
from keras.models import load_model

model = load_model('gan_model15000.h5')


def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path,elem)):
            directories = directories + load_path(os.path.join(path,elem))
            directories.append(os.path.join(path,elem))
    return directories

def load_data_from_dirs(dirs, ext):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d): 
            if f.endswith(ext):
                image = imageio.imread(os.path.join(d,f))
                if len(image.shape) > 2:
                    files.append(image)
                    file_names.append(os.path.join(d,f))
                count = count + 1
    return files     
            

def load_data(directory, ext):

    files = load_data_from_dirs(load_path(directory), ext)
    return files


files = load_data("set5/Set5_cropped", ".png")
x_test = files[:5]

print("data loaded")

def hr_images(images):
    images_hr = array(images)
    return images_hr

def lr_images(images_real , downscale):    
    images = []
    for img in  range(len(images_real)):
        images.append(imresize(images_real[img], [images_real[img].shape[0]//downscale,images_real[img].shape[1]//downscale], interp='bicubic', mode=None))
    images_lr = array(images)
    return images_lr

def lr_images_scaled(images_real_lr , downscale):    
    images = []
    for img in  range(len(images_real_lr)):
        images.append(imresize(images_real_lr[img], [images_real_lr[img].shape[0]*downscale,images_real_lr[img].shape[1]*downscale], interp='bicubic', mode=None))
    images_lr_scaled = array(images)
    return images_lr_scaled

def preprocess_HR(x):
    return np.divide(x.astype(np.float32), 127.5) - np.ones_like(x,dtype=np.float32)


def deprocess_HR(x):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8) 


def preprocess_LR(x):
    return np.divide(x.astype(np.float32), 255.)


def deprocess_LR(x):
    x = np.clip(x*255, 0, 255)
    return x

def normalize(input_data):

    return (input_data.astype(np.float32) - 127.5)/127.5 
    
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8) 

def deprocess_LRS(x):
    x = np.clip(x*255, 0, 255)
    return x.astype(np.uint8)


x_test_hr = hr_images(x_test)
x_test_hr = normalize(x_test_hr)

x_test_lr = lr_images(x_test, 4)
x_test_lr = normalize(x_test_lr)

print("data processed")
print(len(x_test_lr))


def plot_generated_images(model, dim=(1, 3), figsize=(15, 5)):
    
    for i in range(0,5):
        image_batch_hr = denormalize(x_test_hr[i])
        image_batch_lr = x_test_lr[i]
        gen_img = model.predict(image_batch_lr)
        generated_image = denormalize(gen_img)
        image_batch_lr = denormalize(image_batch_lr)
        
        #generated_image = deprocess_HR(generator.predict(image_batch_lr))
        
        plt.figure(figsize=figsize)
        
        plt.subplot(dim[0], dim[1], 1)
        plt.imshow(image_batch_lr[1], interpolation='nearest')
        plt.axis('off')
            
        plt.subplot(dim[0], dim[1], 2)
        plt.imshow(generated_image[1], interpolation='nearest')
        plt.axis('off')
        
        plt.subplot(dim[0], dim[1], 3)
        plt.imshow(image_batch_hr[1], interpolation='nearest')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('output1/gan_generated_image_epoch_%d.png' % i)

plot_generated_images(model)