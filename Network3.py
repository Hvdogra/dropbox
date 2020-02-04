#!/usr/bin/env python
#title           :Network.py
#description     :Architecture file(Generator and Discriminator)
#author          :Deepak Birla
#date            :2018/10/30
#usage           :from Network import Generator, Discriminator
#python_version  :3.5.4 

# Modules
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input, concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add


# Residual block
def res_block_gen(model, kernal_size, filters, strides):
    
    gen = model
    
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
        
    model = add([gen, model])
    
    return model

def dense_factor(model, kernal_size, filters, strides):

    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = LeakyReLU(alpha = 0.2)(model)

    return model

def dense_block(model, kernal_size, filters, strides):

    gen = model
    concatenated_inputs = model
    for i in range(4):
        x = dense_factor(gen, kernal_size, filters, strides)
        concatenated_inputs = concatenate([concatenated_inputs, x], axis=3)

    return concatenated_inputs
    
    
def up_sampling_block(model, kernal_size, filters, strides):
    
    # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
    # Even we can have our own function for deconvolution (i.e one made in Utils.py)
    #model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = UpSampling2D(size = 2)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model


def discriminator_block(model, filters, kernel_size, strides):
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model

# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Generator(object):

    def __init__(self, noise_shape):
        
        self.noise_shape = noise_shape

    def generator(self):
        
	    gen_input = Input(shape = self.noise_shape)
	    
	    model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(gen_input)
	    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
	    
	    gen_model = model
        
        # Using 16 Residual Blocks
	    for index in range(16):
	        model = res_block_gen(model, 3, 64, 1)
	    
	    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
	    model = BatchNormalization(momentum = 0.5)(model)
	    model = add([gen_model, model])
	    
	    # Using 2 UpSampling Blocks
	    for index in range(2):
	        model = up_sampling_block(model, 3, 256, 1)
	    
	    model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)
	    model = Activation('tanh')(model)
	   
	    generator_model = Model(inputs = gen_input, outputs = model)
        
	    return generator_model

# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Discriminator(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape
    
    def discriminator(self):
        
        dis_input = Input(shape = self.image_shape)
        
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)
        model = LeakyReLU(alpha = 0.2)(model)
        
        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)
        
        model = Flatten()(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha = 0.2)(model)
       
        model = Dense(1)(model)
        model = Activation('sigmoid')(model) 
        
        discriminator_model = Model(inputs = dis_input, outputs = model)
        
        return discriminator_model


# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Attention(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape
    
    def attention(self):
        
        dis_input = Input(shape = self.image_shape)
        
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)
        model = LeakyReLU(alpha = 0.2)(model)

        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
        model = LeakyReLU(alpha = 0.2)(model)

        stop_one = model

        model = MaxPooling2D(pool_size=(2,2))(model)

        model = dense_block(model, 3, 64, 1)
        stop_two = model
        model = AveragePooling2D(pool_size=(2,2))(model)

        model = dense_block(model, 3, 64, 1)
        stop_three = model
        model = AveragePooling2D(pool_size=(2,2))(model)

        model = dense_block(model, 3, 64, 1)
        model = up_sampling_block(model, 3, 576, 1)
        model = add([stop_three, model])

        model = dense_block(model, 3, 64, 1)
        model = up_sampling_block(model, 3, 320, 1)
        model = add([stop_two, model])

        model = dense_block(model, 3, 64, 1)
        model = up_sampling_block(model, 3, 64, 1)
        model = add([stop_one, model])

        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
        model = LeakyReLU(alpha = 0.2)(model)

        model = Conv2D(filters = 3, kernel_size = 3, strides = 1, padding = "same")(model)
        model = LeakyReLU(alpha = 0.2)(model)

        attention_model = Model(inputs = dis_input, outputs = model)
        
        return attention_model


# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class ResidueAdd(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape
    
    def residueadd(self):
        
        dis_input = Input(shape = self.image_shape) #48
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)
        model = LeakyReLU(alpha = 0.2)(model)
        model = MaxPooling2D((2, 2), padding='same', name="pool_24")(model)#24
        model_24 = model
        model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same")(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = MaxPooling2D((2, 2), padding='same', name="pool_12")(model)#12
        model_12 = model
        model = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = "same")(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = MaxPooling2D((2, 2), padding='same', name="pool_6")(model)#6
        model_6 = model
        model = Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = "same")(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = MaxPooling2D((2, 2), padding='same', name="pool_3")(model)#3
        model_3 = model
        model = Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = "same")(model)
        model = LeakyReLU(alpha = 0.2)(model)
        encoded = MaxPooling2D((3, 3), padding='same', name="pool_1")(model)#1


        model = Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = "same")(encoded)
        model = LeakyReLU(alpha = 0.2)(model)
        model = UpSampling2D((3, 3), name="sample_3")(model)#3
        model = add([model_3, model])

        model = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = "same")(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = UpSampling2D((2, 2), name="sample_6")(model)
        model = add([model_6, model])

        model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same")(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = UpSampling2D((2, 2), name="sample_12")(model)
        model = add([model_12, model])

        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = UpSampling2D((2, 2), name="sample_24")(model)
        model = add([model_24, model])

        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = UpSampling2D((2, 2))(model)

        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = UpSampling2D((2, 2))(model)   

        model = Conv2D(filters = 3, kernel_size = 3, strides = 1, padding = "same")(model)
        model = LeakyReLU(alpha = 0.2)(model)
        decoded = UpSampling2D((2, 2))(model)

        
        residue_model = Model(inputs = dis_input, outputs = decoded)
        
        return residue_model
