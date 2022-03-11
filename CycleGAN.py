# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 05:44:20 2020
CycleGAN
@author: User
"""
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation, LeakyReLU
from keras.initializers import RandomNormal
from tensorflow.keras.layers import Concatenate
from tensorflow_addons.layers import InstanceNormalization
from keras.utils.vis_utils import plot_model
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
from data_loader import DataLoader

class CycleGAN(object):
    
    def __init__(self, width=28, height=28, channels=3, n_residual=2):
        
        self.n_residual = n_residual
        self.image_shape = (width, height, channels)
        self.D_A = self.__define_discriminator()
        self.D_B = self.__define_discriminator()
        self.G_A2B = self.__define_generator()
        self.G_B2A = self.__define_generator()
        self.C_A2B = self.__define_composite_model(self.G_A2B, self.D_B, self.G_B2A)
        self.C_B2A = self.__define_composite_model(self.G_B2A, self.D_A, self.G_A2B)
        
        self.dataset_name = 'apple2orange'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(height, width))
        
        
    # define the discriminator model    
    def __define_discriminator(self):
        
        init = RandomNormal(stddev=0.02)
        in_image = Input(shape=self.image_shape)  
        #C64
        d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = InstanceNormalization(gamma_initializer=init)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256
        d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = InstanceNormalization(gamma_initializer=init)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C512
        d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = InstanceNormalization(gamma_initializer=init)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # second last output layer
        d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
        d = InstanceNormalization(gamma_initializer=init)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # patch output
        patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
        
        model = Model(in_image, patch_out)
        model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
        return model
    
    # generator a residual block
    def __residual_block(self, n_filters, input_layer):
        
        init = RandomNormal(stddev=0.02)
        #first layer
        g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
        InstanceNormalization(gamma_initializer=init)(g)
        g = Activation('relu')(g)
        #second layer
        g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(gamma_initializer=init)(g)
        # concatenate merge channel-wise with input layer
        g = Concatenate()([g, input_layer])
        return g
    
    #define the standalone generator model
    def __define_generator(self):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        #image input
        in_image = Input(shape=self.image_shape)
        #C7s1-64
        g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
        g = InstanceNormalization(gamma_initializer=init)(g)
        g = Activation('relu')(g)
        # C128
        g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(gamma_initializer=init)(g)
        g = Activation('relu')(g)
        # d256
        g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(gamma_initializer=init)(g)
        g = Activation('relu')(g)
        # R256
        for _ in range(self.n_residual):
            g = self.__residual_block(256, g)
        # u128
        g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(gamma_initializer=init)(g)
        g = Activation('relu')(g)
        # u64
        g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(gamma_initializer=init)(g)
        g = Activation('relu')(g)
        # c7s1-3
        g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(gamma_initializer=init)(g)
        out_image = Activation('tanh')(g)
        
        model = Model(in_image, out_image)
        return model
    
    #define a composite model 
    def __define_composite_model(self, g_1, d, g_2):

        g_1.trainable = True
        d.trainable = False
        g_2.trainable = False
        # discriminator element
        input_gen = Input(shape=self.image_shape)
        gen1_out = g_1(input_gen)
        output_d = d(gen1_out)
        # identity element
        input_id = Input(shape=self.image_shape)
        output_id = g_1(input_id)
        # forward cycle
        output_f = g_2(gen1_out)
        # backward cycle
        gen2_out = g_2(input_id)
        output_b = g_1(gen2_out)
        # define model graph
        model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
        #define optimization algorithm configuration
        opt = Adam(lr=0.0002, beta_1=0.5) 
        model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
        model.summary()
        return model
    
    def __generate_real_samples(self, dataset, n_samples, patch_shape):
            
        # choose random instances
        ix = np.random.randint(0, dataset.shape[0], size=n_samples)
        # retrieve selected images
        X = dataset[ix]
        # generate 'real' class labels (1)
        y = np.ones((n_samples, patch_shape, patch_shape, 1))
        return X, y
    
    def __generate_fake_samples(self, g_model, dataset, patch_shape):
        
        #generate fake instance
        X = g_model.predict(dataset)
        # create 'fake' class labels (0)
        y = np.zeros((len(X), patch_shape, patch_shape, 1))
        return X, y
    
    #update image pool for fake images
    def __update_image_pool(self, pool, images, max_size=50):
        
        selected=list()
        for image in images:
            if len(pool) < max_size:
                #stock the pool
                pool.append(image)
                selected.append(image)
            elif random() < 0.5:
                #use image, but don't add it to the pool
                selected.append(image)
            else:
                #replace an existing image and use replaced image
                ix = randint(0, max_size)
                selected.append(pool[ix])
                pool[ix] = image
        return np.asarray(selected)
    
    
    """
    train() unfinished
    """
    #train cyclegan models
    def train(self, n_epochs, n_batch, sample_interval=50):
        
        #determine the output square shape of the discriminator
        n_patch = self.D_A.output_shape[1]
        #prepare image pool for fakes
        poolA, poolB = list(), list()
        
        for epoch in range(n_epochs):
            for batch_i, (trainA, trainB) in enumerate(self.data_loader.load_batch(n_batch)):
                #select a batch of real sample
                x_realA, y_realA = self.__generate_real_samples(trainA, n_batch, n_patch)
                x_realB, y_realB = self.__generate_real_samples(trainB, n_batch, n_patch)
                #generate a batch of fake samples
                x_fakeA, y_fakeA = self.__generate_fake_samples(self.G_B2A, x_realB, n_patch)
                x_fakeB, y_fakeB = self.__generate_fake_samples(self.G_A2B, x_realA, n_patch)
                #update fakes from pool
                x_fakeA = self.__update_image_pool(poolA, x_fakeA)
                x_fakeB = self.__update_image_pool(poolB, x_fakeB)
                #update generator B->A via addversarial and cycle loss
                g_loss2, _, _, _, _ = self.C_B2A.train_on_batch([x_realB, x_realA], [y_realA, x_realA, x_realB, x_realA])
                #update discriminator for A->[real/fake]
                dA_loss1 = self.D_A.train_on_batch(x_realA, y_realA)
                dA_loss2 = self.D_A.train_on_batch(x_fakeA, y_fakeB)
                #update generator A->B via adversarial and cycle loss
                g_loss1, _, _, _, _ = self.C_A2B.train_on_batch([x_realA, x_realB], [y_realB, x_realB, x_realA, x_realB])
                #update discriminator for B->[real/fake]
                dB_loss1 = self.D_B.train_on_batch(x_realB, y_realB)
                dB_loss2 = self.D_B.train_on_batch(x_fakeB, y_fakeB)
                #summarize performance
                print(">>[Epoch %d/%d], [Batch %d/%d]: dA[%.3f, %.3f], dB[%.3f, %.3f], g[%.3f, %.3f]" %(
                          epoch, n_epochs,
                          batch_i, self.data_loader.n_batches,
                          dA_loss1,dA_loss2, 
                          dB_loss1,dB_loss2, 
                          g_loss1,g_loss2))
     
                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
                    
        #save weight and model
        if not os.path.exists("./saved_model_weights"):
            os.makedirs('saved_model_weights', exist_ok=True)
            self.G.save_weights('saved_model_weights/generator_weights.h5')
            self.D.save_weights('saved_model_weights/discriminator_weights.h5')
            self.stacked_generator_discriminator.save_weights('saved_model_weights/combined_weights.h5')
                    
        
    
    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 3

        imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)

        # Demo (for GIF)
        #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
        #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

        # Translate images to the other domain
        fake_B = self.G_A2B.predict(imgs_A)
        fake_A = self.G_B2A.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.G_B2A.predict(fake_B)
        reconstr_B = self.G_A2B.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()
     
    
if __name__ == '__main__':
    
    cyclegan = CycleGAN(width=28, height=28, channels=3, n_residual=2)
    cyclegan.train(n_epochs=200, n_batch=1)
    
    
    
    
    