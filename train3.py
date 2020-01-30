from Network3 import Generator, Discriminator, Attention, ResidueAdd

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.applications.vgg19 import VGG19
from keras.layers.convolutional import UpSampling2D
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
    x = generator(gan_input)
    model1 = Model(inputs=residue.input, outputs=residue.get_layer('pool_24').output)(gan_input)
    model2 = Model(inputs=residue.input, outputs=residue.get_layer('pool_12').output)(gan_input)
    model3 = Model(inputs=residue.input, outputs=residue.get_layer('pool_6').output)(gan_input)
    model4 = Model(inputs=residue.input, outputs=residue.get_layer('pool_3').output)(gan_input)
    model5 = Model(inputs=residue.input, outputs=residue.get_layer('sample_3').output)(gan_input)
    model6 = Model(inputs=residue.input, outputs=residue.get_layer('sample_6').output)(gan_input)
    model7 = Model(inputs=residue.input, outputs=residue.get_layer('sample_12').output)(gan_input)
    model8 = Model(inputs=residue.input, outputs=residue.get_layer('sample_24').output)(gan_input)
    
    # model1 = residue(gan_input).get_layer("pool_24")
    # model2 = residue(gan_input).get_layer("pool_12")
    # model3 = residue(gan_input).get_layer("pool_6")
    # model4 = residue(gan_input).get_layer("pool_3")
    # model5 = residue(gan_input).get_layer("sample_3")
    # model6 = residue(gan_input).get_layer("sample_6")
    # model7 = residue(gan_input).get_layer("sample_12")
    # model8 = residue(gan_input).get_layer("sample_24")
    diff1 = subtract([model1, model8])#24x24x64
    diff1 = Flatten()(diff1)
    diff2 = subtract([model2, model7])#12x12x128
    diff2 = Flatten()(diff2)
    diff3 = subtract([model3, model6])#6x6x256
    diff3 = Flatten()(diff3)
    diff4 = subtract([model4, model5])#3x3x512
    diff4 = Flatten()(diff4)
    d1 = Dense(1024, activation='relu')(diff1)
    d2 = Dense(1024, activation='relu')(diff2)
    d3 = Dense(1024, activation='relu')(diff3)
    d4 = Dense(1024, activation='relu')(diff4)
    d = add([d1,d2])
    d = add([d,d3])
    d = add([d,d4])
    d = Dense(110592, activation='relu')(d)
    d = Reshape((192, 192, 3))(d)

    final = add([d, x])

    gan_output = discriminator(final)
    gan = Model(inputs=gan_input, outputs=[final,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
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
    image_batch_lr_scaled = x_test_lr_scaled[rand_nums]
    gen_img = generator.predict(image_batch_lr)
    attention_images = attention.predict(image_batch_lr_scaled)
    generated_images_sr_scaled = gen_img*attention_images+image_batch_lr_scaled
    generated_image = denormalize(generated_images_sr_scaled)
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
    plt.savefig('output/gan_generated_image_epoch_%d.png' % epoch)
    

def train(epochs=1, batch_size=128):

    downscale_factor = 4
    
    batch_count = int(x_train_hr.shape[0] / batch_size)
    shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, image_shape[2])
    
    generator = Generator(shape).generator()
    discriminator = Discriminator(image_shape).discriminator()
    # attention = Attention(image_shape).attention()
    residue_data = ResidueAdd(shape).residueadd()

    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss=vgg_loss, optimizer=adam)
    discriminator.compile(loss="binary_crossentropy", optimizer=adam)
    # attention.compile(loss="mean_squared_error", optimizer=adam)
    residue_data.compile(loss=vgg_loss, optimizer=adam)
    
    shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, 3)
    gan = get_gan_network(discriminator, shape, generator, adam, residue_data)

    for e in range(1, epochs+1):
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
            
        print("Loss HR , Loss LR, Loss GAN")
        print(d_loss_real, d_loss_fake, loss_gan)

        if e == 1 or e % 5 == 0:
            plot_generated_images(e, generator, residue_data)
        if e % 300 == 0:
            generator.save('./output/gen_model%d.h5' % e)
            discriminator.save('./output/dis_model%d.h5' % e)
            gan.save('./output/gan_model%d.h5' % e)

train(200, 4)

