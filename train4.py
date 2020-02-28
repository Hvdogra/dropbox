from Network3 import Generator, Discriminator, Attention, ResidueAdd

import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from keras.applications.vgg19 import VGG19
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Model
from keras.layers import add, multiply, subtract, Dense, Reshape, Flatten
from keras.optimizers import SGD, Adam, RMSprop
import keras
import keras.backend as K
from keras.layers import Lambda, Input
import tensorflow as tf
import skimage.transform
from skimage import data, io, filters
import imageio
import numpy as np
from numpy import array
from skimage.transform import rescale, resize
from scipy.misc import imresize
import os

np.random.seed(10)
image_shape = (192,192,3)



def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303


def up_sampling_block(model, kernal_size, filters, strides):
    
    # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
    # Even we can have our own function for deconvolution (i.e one made in Utils.py)
    #model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = UpSampling2D(size = 2)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model

def vgg_loss(y_true, y_pred):
    
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

def get_gan_network(discriminator, shape, generator, optimizer, residue):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    # res_input = Input(shape=shape)
    # x = generator(gan_input)
    x = Model(inputs=generator.input, outputs=generator.get_layer('add_17_all').output)(gan_input)
    x1 = Model(inputs=residue.input, outputs=residue.get_layer('conv2d_46_all').output)(gan_input)
    y = Model(inputs=residue.input, outputs=residue.get_layer('leaky_re_lu_22_all').output)(gan_input)

    diff1 = subtract([x1,y])

    z = add([diff1, x])
    print(z.shape)
    # Using 2 UpSampling Blocks
    for index in range(2):
        z = up_sampling_block(z, 3, 256, 1)
        
    z = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(z)
    z = Activation('tanh')(z)
    gan_output = discriminator(z)
    gan = Model(inputs=gan_input, outputs=[z,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3], metrics=[PSNR, 'accuracy'],
                optimizer=optimizer)

    return gan


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
            
def load_data_from_dirs_resize(dirs, ext, size):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d): 
            if f.endswith(ext):
                files.append(resize(data.imread(os.path.join(d,f)), size))
                file_names.append(os.path.join(d,f))
                count = count + 1
    return files     
                        
          
def load_data(directory, ext):

    files = load_data_from_dirs(load_path(directory), ext)
    return files


files = load_data("data", ".png")
x_train = files[:600]
x_test = files[600:900]

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

x_train_hr = hr_images(x_train)
x_train_hr = normalize(x_train_hr)

x_train_lr = lr_images(x_train, 4)
x_train_lr_scaled = lr_images_scaled(x_train_lr, 4)
x_train_lr = normalize(x_train_lr)
x_train_lr_scaled = normalize(x_train_lr_scaled)


x_test_hr = hr_images(x_test)
x_test_hr = normalize(x_test_hr)

x_test_lr = lr_images(x_test, 4)
x_test_lr_scaled = lr_images_scaled(x_test_lr, 4)
x_test_lr = normalize(x_test_lr)
x_test_lr_scaled = normalize(x_test_lr_scaled)

print("data processed")
print(len(x_train_lr_scaled))

def plot_generated_images(epoch,generator, attention, examples=3 , dim=(1, 3), figsize=(15, 5)):
    
    rand_nums = np.random.randint(0, x_test_hr.shape[0], size=examples)
    image_batch_hr = denormalize(x_test_hr[rand_nums])
    image_batch_lr = x_test_lr[rand_nums]
    [gen_img,gan_output] = generator.predict(image_batch_lr)
    # image_batch_lr_scaled = x_test_lr_scaled[rand_nums]
    # gen_img = generator.predict(image_batch_lr)
    # attention_images = attention.predict(image_batch_lr_scaled)
    # generated_images_sr_scaled = gen_img*attention_images+image_batch_lr_scaled
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
    plt.savefig('output3/gan_generated_image_epoch_%d.png' % epoch)
    

def train(epochs=1, batch_size=128, lr_val=1):

    downscale_factor = 4
    
    batch_count = int(x_train_hr.shape[0] / batch_size)
    shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, image_shape[2])
    
    generator = Generator(shape).generator()
    # print(generator.summary())
    discriminator = Discriminator(image_shape).discriminator()
    # attention = Attention(image_shape).attention()
    residue_data = ResidueAdd(shape).residueadd()
    # print(residue_data.summary())

    lr_dat = lr_val*1E-4
    adam = Adam(lr=lr_dat, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss=vgg_loss, optimizer=adam)
    discriminator.compile(loss="binary_crossentropy", optimizer=adam)
    # attention.compile(loss="mean_squared_error", optimizer=adam)
    residue_data.compile(loss=vgg_loss, optimizer=adam)
    
    shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, 3)
    gan = get_gan_network(discriminator, shape, generator, adam, residue_data)
    print(gan.summary())
    epoch_array = []
    loss_array = []



    for e in range(1, epochs+1):
        epoch_array.append(e)
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in range(batch_count):
            
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]
            # image_batch_lr_scaled = x_train_lr_scaled[rand_nums]
            [final,gan_output] = gan.predict(image_batch_lr)
            
    
            # res_images = res_data.predict(image_batch_lr)
            generated_images_sr_scaled = final

            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            fake_data_Y = np.random.random_sample(batch_size)*0.2
            
            discriminator.trainable = True
            
            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images_sr_scaled, fake_data_Y)
            #d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]
            # image_batch_lr_scaled = x_train_lr_scaled[rand_nums]

            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            discriminator.trainable = False

            loss_gan = gan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])
            
        # print("Loss HR, Loss LR, Loss GAN, Loss Generator, Loss Disc, PSNR Generator, Accuracy Generator, Accuracy Disc")
        # print(gan.metrics_names)
        # print(d_loss_real, d_loss_fake, loss_gan[0], loss_gan[1], loss_gan[2], loss_gan[3], loss_gan[4], loss_gan[6])
        print("Loss HR : %f, Loss LR : %f, Loss GAN : %f, Loss Generator : %f, Loss Disc : %f, PSNR Generator : %f, Accuracy Generator : %f, Accuracy Disc : %f" %(d_loss_real, d_loss_fake, loss_gan[0], loss_gan[1], loss_gan[2], loss_gan[3], loss_gan[4], loss_gan[6]))
        loss_array.append(loss_gan[0])
        if e == 1 or e % 5 == 0:
            # print("hi")
            plot_generated_images(e, gan, residue_data)
        if e % 2000 == 0:
            # generator.save('./output/gen_model%d.h5' % e)
            discriminator.save('./output3/dis_model%d.h5' % e)
            gan.save('./output3/gan_model%d.h5' % e)

    plt.figure()
    plt.plot(epoch_array, loss_array)
    plt.savefig('new_2000.png')

train(2000, 8)
# x = [1,7,8,9]
# for each in x:
#     train(40,4,each)

