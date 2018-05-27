"""
Hierarchical GAN with InfoGAN structure and losses

Author(s): Wei Chen (wchen459@umd.edu)
"""

import os.path
import pickle
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Reshape, Input, Lambda
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D, Cropping2D
from keras.layers import LeakyReLU, Dropout, GaussianNoise
from keras.layers import BatchNormalization, RepeatVector
from keras.layers import concatenate, dot, multiply
from keras.regularizers import l2
from keras.optimizers import Adam, Adamax, RMSprop
from keras import backend as K
from keras.initializers import RandomNormal
from keras.layers.pooling import GlobalAveragePooling2D
from keras.utils import plot_model
from keras.models import load_model


def label_flipping(y, p):
    h = np.random.binomial(1, p)
    if h:
        y[:, [0, 1]] = y[:, [1, 0]]
    return y

def gaussian_loss(y_true, y_pred):
    
    Q_C_mean = y_pred[:, 0, :]
    Q_C_logstd = y_pred[:, 1, :]

    y_true = y_true[:, 0, :]

    epsilon = (y_true - Q_C_mean) / (K.exp(Q_C_logstd) + K.epsilon())
    loss_Q_C = (Q_C_logstd + 0.5 * K.square(epsilon))
    loss_Q_C = K.mean(loss_Q_C)

    return loss_Q_C


class HGAN(object):
    def __init__(self, X_train, X_test, latent_dim=2, noise_dim=100):

        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.D = None   # discriminator
        self.G1 = None   # first generator
        self.G2 = None   # second generator
        self.DM = None  # discriminator model
        self.AM1 = None  # first adversarial model
        self.AM2 = None  # second adversarial model
        
        self.x_train = X_train
        self.x_test = X_test
        n_points = X_train.shape[1]/2
        self.input_shape = (n_points, 2, 1)
        self.conc_shape = X_train.shape[1:]

    # (W-F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        
        kernel_height = 5
        depth = 64
        dropout = 0.4
        weight_decay = 1e-5
        
        x = Input(shape=self.conc_shape)
        
        y = Conv2D(depth*1, (kernel_height,2), strides=2, padding='same', 
                   kernel_regularizer=l2(weight_decay))(x)
        y = BatchNormalization(momentum=0.9)(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Dropout(dropout)(y)

        y = Conv2D(depth*2, (kernel_height,2), strides=2, padding='same', 
                   kernel_regularizer=l2(weight_decay))(y)
        y = BatchNormalization(momentum=0.9)(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Dropout(dropout)(y)

        y = Conv2D(depth*4, (kernel_height,2), strides=2, padding='same', 
                   kernel_regularizer=l2(weight_decay))(y)
        y = BatchNormalization(momentum=0.9)(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Dropout(dropout)(y)

        y = Conv2D(depth*8, (kernel_height,2), strides=2, padding='same', 
                   kernel_regularizer=l2(weight_decay))(y)
        y = BatchNormalization(momentum=0.9)(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Dropout(dropout)(y)
        
        y = Flatten()(y)
        y = Dense(1024)(y)
        y = BatchNormalization(momentum=0.9)(y)
        y = LeakyReLU(alpha=0.2)(y)
        
        d = Dense(2, activation='softmax', name="D_out")(y)
        
        def linmax(x):
            return K.maximum(x, -16)
    
        def linmax_shape(input_shape):
            return input_shape
            
        # Auxiliary Q
        q1 = Dense(128)(y)
        q1 = BatchNormalization(momentum=0.9)(q1)
        q1 = LeakyReLU(alpha=0.2)(q1)
        q_mean1 = Dense(self.latent_dim, activation='linear', name="Q_mean1")(q1)
        q_logstd1 = Dense(self.latent_dim, name="Q_logstd1")(q1)
        q_logstd1 = Lambda(linmax, output_shape=linmax_shape)(q_logstd1)
        # Reshape Q to nbatch, 1, latent_dim
        q_mean1 = Reshape((1, self.latent_dim))(q_mean1)
        q_logstd1 = Reshape((1, self.latent_dim))(q_logstd1)
        q1 = concatenate([q_mean1, q_logstd1], name="Q_out1", axis=1)
        
        q2 = Dense(128)(y)
        q2 = BatchNormalization(momentum=0.9)(q2)
        q2 = LeakyReLU(alpha=0.2)(q2)
        q_mean2 = Dense(self.latent_dim, activation='linear', name="Q_mean2")(q2)
        q_logstd2 = Dense(self.latent_dim, name="Q_logstd2")(q2)
        q_logstd2 = Lambda(linmax, output_shape=linmax_shape)(q_logstd2)
        # Reshape Q to nbatch, 1, latent_dim
        q_mean2 = Reshape((1, self.latent_dim))(q_mean2)
        q_logstd2 = Reshape((1, self.latent_dim))(q_logstd2)
        q2 = concatenate([q_mean2, q_logstd2], name="Q_out2", axis=1)
        
        self.D = Model(inputs=x, outputs=[d, q1, q2])
        self.D.summary()
        plot_model(self.D, to_file='airfoil/2g1d/D.png',show_shapes=True)
        return self.D

    def generator1(self):
        if self.G1:
            return self.G1
            
        kernel_height = 5
        depth = 32*16
        dim = (self.input_shape[0]+12)/16
        weight_decay = 1e-5
        noise_std = 0.01
        
        c1 = Input(shape=(self.latent_dim,), name="latent_input1")
        z1 = Input(shape=(self.noise_dim,), name="noise_input1")
        
        x = concatenate([c1, z1])
        
        x = Dense(dim*2*depth, kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((dim, 2, depth))(x)

        x = UpSampling2D((2,1))(x)
        x = Conv2DTranspose(int(depth/2), (kernel_height,2), padding='same', 
                            kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = GaussianNoise(noise_std)(x)

        x = UpSampling2D((2,1))(x)
        x = Conv2DTranspose(int(depth/4), (kernel_height,2), padding='same', 
                            kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = GaussianNoise(noise_std)(x)

        x = UpSampling2D((2,1))(x)
        x = Conv2DTranspose(int(depth/8), (kernel_height,2), padding='same', 
                            kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = GaussianNoise(noise_std)(x)

        x = UpSampling2D((2,1))(x)
        x = Conv2DTranspose(int(depth/16), (kernel_height,2), padding='same', 
                            kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = GaussianNoise(noise_std)(x)

        # Out: 100 x 2, xy coordinates, [-1.0,1.0] per coordinate
        x = Conv2DTranspose(1, (kernel_height,2), padding='same', 
                            kernel_regularizer=l2(weight_decay))(x)
        x = Activation('tanh')(x)
        x = Cropping2D((6, 0))(x)
        
        self.G1 = Model(inputs=[c1, z1], outputs=x)
        self.G1.summary()
        plot_model(self.G1, to_file='airfoil/2g1d/G1.png',show_shapes=True)
        return self.G1

    def generator2(self):
        if self.G2:
            return self.G2
            
        kernel_height = 5
        depth = 32*16
        dim = (self.input_shape[0]+12)/16
        weight_decay = 1e-5
        noise_std = 0.01
        dropout = 0.4
        
        c2 = Input(shape=(self.latent_dim,))
        z2 = Input(shape=(self.noise_dim,))
        
        cz = concatenate([c2, z2])
        
        f2 = Dense(dim*2*depth, kernel_regularizer=l2(weight_decay))(cz)
        f2 = BatchNormalization(momentum=0.9)(f2)
        f2 = LeakyReLU(alpha=0.2)(f2)
        f2 = Reshape((dim, 2, depth))(f2)

        f2 = UpSampling2D((2,1))(f2)
        f2 = Conv2DTranspose(int(depth/16), (kernel_height,2), padding='same', 
                            kernel_regularizer=l2(weight_decay))(f2)
        f2 = BatchNormalization(momentum=0.9)(f2)
        f2 = LeakyReLU(alpha=0.2)(f2)
        f2 = GaussianNoise(noise_std)(f2)

        f2 = UpSampling2D((2,1))(f2)
        f2 = Conv2DTranspose(int(depth/16), (kernel_height,2), padding='same', 
                            kernel_regularizer=l2(weight_decay))(f2)
        f2 = BatchNormalization(momentum=0.9)(f2)
        f2 = LeakyReLU(alpha=0.2)(f2)
        f2 = GaussianNoise(noise_std)(f2)
        
        x1 = Input(shape=self.input_shape)
        
        f1 = ZeroPadding2D((6, 0))(x1)
        
        f1 = Conv2D(depth/16, (kernel_height,2), strides=(2,1), padding='same', 
                   kernel_regularizer=l2(weight_decay))(f1)
        f1 = BatchNormalization(momentum=0.9)(f1)
        f1 = LeakyReLU(alpha=0.2)(f1)
        f1 = Dropout(dropout)(f1)

        f1 = Conv2D(depth/8, (kernel_height,2), strides=(2,1), padding='same', 
                   kernel_regularizer=l2(weight_decay))(f1)
        f1 = BatchNormalization(momentum=0.9)(f1)
        f1 = LeakyReLU(alpha=0.2)(f1)
        f1 = Dropout(dropout)(f1)

        f = concatenate([f1, f2])
        
        f = Conv2D(depth/4, (kernel_height,2), strides=(2,1), padding='same', 
                   kernel_regularizer=l2(weight_decay))(f)
        f = BatchNormalization(momentum=0.9)(f)
        f = LeakyReLU(alpha=0.2)(f)
        f = Dropout(dropout)(f)

        f = Conv2D(depth/2, (kernel_height,2), strides=(2,1), padding='same', 
                   kernel_regularizer=l2(weight_decay))(f)
        f = BatchNormalization(momentum=0.9)(f)
        f = LeakyReLU(alpha=0.2)(f)
        f = Dropout(dropout)(f)

        x = UpSampling2D((2,1))(f)
        x = Conv2DTranspose(int(depth/2), (kernel_height,2), padding='same', 
                            kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = GaussianNoise(noise_std)(x)

        x = UpSampling2D((2,1))(x)
        x = Conv2DTranspose(int(depth/4), (kernel_height,2), padding='same', 
                            kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = GaussianNoise(noise_std)(x)

        x = UpSampling2D((2,1))(x)
        x = Conv2DTranspose(int(depth/8), (kernel_height,2), padding='same', 
                            kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = GaussianNoise(noise_std)(x)

        x = UpSampling2D((2,1))(x)
        x = Conv2DTranspose(int(depth/16), (kernel_height,2), padding='same', 
                            kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = GaussianNoise(noise_std)(x)

        # Out: 100 x 2, xy coordinates, [-1.0,1.0] per coordinate
        x = Conv2DTranspose(1, (kernel_height,2), padding='same', 
                            kernel_regularizer=l2(weight_decay))(x)
        x = Activation('tanh')(x)
        x = Cropping2D((6, 0))(x)
        
        self.G2 = Model(inputs=[c2, z2, x1], outputs=x)
        self.G2.summary()
        plot_model(self.G2, to_file='airfoil/2g1d/G2.png',show_shapes=True)
        return self.G2
    
    def discriminator_model(self):
        if self.DM:
            return self.DM
        x = Input(shape=self.conc_shape) # parent with child
        dis = self.discriminator()
        dis.trainable = True
        d, q1, q2 = dis(x)
        self.DM = Model(inputs=x, outputs=[d, q1, q2])
#        optimizer = RMSprop(lr=1e-3)
        optimizer = Adam(lr=0.00005, beta_1=0.5)
        self.DM.compile(loss=['binary_crossentropy', gaussian_loss, gaussian_loss], 
                        loss_weights=[1, 1, 1], optimizer=optimizer)
        return self.DM

    def adversarial_model1(self):
        if self.AM1:
            return self.AM1
        c1 = Input(shape=(self.latent_dim,))
        z1 = Input(shape=(self.noise_dim,))
        gen1 = self.generator1()
        x1 = gen1([c1, z1])
        c2 = Input(shape=(self.latent_dim,))
        z2 = Input(shape=(self.noise_dim,))
        gen2 = self.generator2()
        gen2.trainable = False
        x2 = gen2([c2, z2, x1])
        x = concatenate([x1, x2], axis=1)
        dis = self.discriminator()
        dis.trainable = False
        d, q1, q2 = dis(x)
        self.AM1 = Model(inputs=[c1, z1, c2, z2], outputs=[d, q1, q2])
#        optimizer = RMSprop(lr=1e-3)
        optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.AM1.compile(loss=['binary_crossentropy', gaussian_loss, gaussian_loss], 
                         loss_weights=[1, 1, 0], optimizer=optimizer)
        return self.AM1

    def adversarial_model2(self):
        if self.AM2:
            return self.AM2
        c2 = Input(shape=(self.latent_dim,))
        z2 = Input(shape=(self.noise_dim,))
        x1 = Input(shape=self.input_shape)
        gen2 = self.generator2()
        gen2.trainable = True
        x2 = gen2([c2, z2, x1])
        x = concatenate([x1, x2], axis=1)
        dis = self.discriminator()
        dis.trainable = False
        d, q1, q2 = dis(x)
        self.AM2 = Model(inputs=[c2, z2, x1], outputs=[d, q1, q2])
#        optimizer = RMSprop(lr=1e-3)
        optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.AM2.compile(loss=['binary_crossentropy', gaussian_loss, gaussian_loss], 
                         loss_weights=[1, 0, 1], optimizer=optimizer)
        return self.AM2

    def train(self, train_steps=2000, batch_size=256, save_interval=0, mode='startover'):
        
        g1_fname = '../hgan_idetc2018_data/airfoil/2g1d/generator1.h5'
        g2_fname = '../hgan_idetc2018_data/airfoil/2g1d/generator2.h5'
        d_fname = '../hgan_idetc2018_data/airfoil/2g1d/discriminator.h5'
        
        if os.path.isfile(g1_fname) and os.path.isfile(g2_fname) and os.path.isfile(d_fname):
            trained_existed = True
        else:
            trained_existed = False
            
        if mode != 'startover' and trained_existed:
            self.dis = self.D = load_model(d_fname)
            self.gen1 = self.G1 = load_model(g1_fname)
            self.gen2 = self.G2 = load_model(g2_fname)
            
        else:
            self.dis = self.discriminator()
            self.gen1 = self.generator1()
            self.gen2 = self.generator2()
            
        self.dis_model = self.discriminator_model()
        self.adv_model1 = self.adversarial_model1()
        self.adv_model2 = self.adversarial_model2()
            
        if mode != 'evaluate' or not trained_existed:
    
            losses = dict()
            losses['d_loss_real'] = []
            losses['d_loss_fake'] = []
            losses['g_loss_parent'] = []
            losses['g_loss_child'] = []
            losses['q_loss_parent'] = []
            losses['q_loss_child'] = []
            
            for t in range(train_steps):
                
                sigma = np.exp(-t/1e4) # annealed noise scale
   
                # Train discriminator model and adversarial model
                ind = np.random.choice(self.x_train.shape[0], size=batch_size, replace=False)
                X_train = self.x_train[ind]
                X_train += np.random.normal(scale=sigma, size=X_train.shape)
                y_real = np.zeros((batch_size, 2), dtype=np.uint8)
                y_real[:, 1] = 1
#                y_real = label_flipping(y_real, .1)
                y_latent1 = np.random.uniform(size=(batch_size, self.latent_dim))
                y_latent_target1 = np.expand_dims(y_latent1, 1)
                y_latent_target1 = np.repeat(y_latent_target1, 2, axis=1)
                y_latent2 = np.random.uniform(size=(batch_size, self.latent_dim))
                y_latent_target2 = np.expand_dims(y_latent2, 1)
                y_latent_target2 = np.repeat(y_latent_target2, 2, axis=1)
                d_loss_real = self.dis_model.train_on_batch(X_train, [y_real, y_latent_target1, 
                                                                      y_latent_target2])
            
                noise1 = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim))
                noise2 = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim))
                y_latent1 = np.random.uniform(size=(batch_size, self.latent_dim))
                y_latent_target1 = np.expand_dims(y_latent1, 1)
                y_latent_target1 = np.repeat(y_latent_target1, 2, axis=1)
                y_latent2 = np.random.uniform(size=(batch_size, self.latent_dim))
                y_latent_target2 = np.expand_dims(y_latent2, 1)
                y_latent_target2 = np.repeat(y_latent_target2, 2, axis=1)
                X1_fake = self.gen1.predict_on_batch([y_latent1, noise1])
                X2_fake = self.gen2.predict_on_batch([y_latent2, noise2, X1_fake])
                X_fake = np.concatenate((X1_fake, X2_fake), axis=1)
                X_fake += np.random.normal(scale=sigma, size=X_fake.shape)
                y_fake = np.zeros((batch_size, 2), dtype=np.uint8)
                y_fake[:, 0] = 1
#                y_fake = label_flipping(y_fake, .1)
                d_loss_fake = self.dis_model.train_on_batch(X_fake, [y_fake, y_latent_target1, 
                                                                     y_latent_target2])
                
                a1_loss = self.adv_model1.train_on_batch([y_latent1, noise1, y_latent2, noise2], 
                                                         [y_real, y_latent_target1, y_latent_target2])
                
                a2_loss = self.adv_model2.train_on_batch([y_latent2, noise2, X1_fake], 
                                                         [y_real, y_latent_target1, y_latent_target2])
                
                log_mesg = "%d: [D] real %f fake %f latent1 %f latent2 %f" % (t+1, d_loss_real[1], 
                                d_loss_fake[1], d_loss_fake[2], d_loss_fake[3])
                log_mesg = "%s  [A1] fake %f latent1 %f" % (log_mesg, a1_loss[1], a1_loss[2])
                log_mesg = "%s  [A2] fake %f latent2 %f" % (log_mesg, a2_loss[1], a2_loss[3])
                print(log_mesg)
                
                losses['d_loss_real'].append(d_loss_real[1])
                losses['d_loss_fake'].append(d_loss_fake[1])
                losses['g_loss_parent'].append(a1_loss[1])
                losses['g_loss_child'].append(a2_loss[1])
                losses['q_loss_parent'].append(a1_loss[2])
                losses['q_loss_child'].append(a2_loss[3])
                
                if save_interval>0 and (t+1)%save_interval==0:
                    self.gen1.save(g1_fname)
                    self.gen2.save(g2_fname)
                    self.dis.save(d_fname)
                    with open('../hgan_idetc2018_data/airfoil/2g1d/losses.pkl', 'wb') as f:
                        pickle.dump(losses, f, pickle.HIGHEST_PROTOCOL)
                    print 'Plotting results ...'
                    from shape_plot import plot_grid
                    plot_grid(9, gen_func=self.synthesize_parent, d=self.latent_dim, 
                              scale=.95, save_path='airfoil/2g1d/parent.svg')
                    plot_grid(9, gen_func=self.synthesize_child, d=self.latent_dim, 
                              scale=.95, save_path='airfoil/2g1d/child.svg')

    def synthesize_parent(self, c1):
        ''' Generate parent without child '''
        if isinstance(c1, int):
            N = c1
            c1 = np.random.uniform(size=(N, self.latent_dim))
            noise1 = np.random.normal(scale=0.5, size=(N, self.noise_dim))
            X = self.gen1.predict([c1, noise1])
        else:
            N = c1.shape[0]
            X = []
            for i in range(N):
                m = 10
                noise1 = np.random.normal(scale=0.5, size=(m, self.noise_dim))
                c1i = np.tile(c1[i], (m, 1))
                Xi = self.gen1.predict([c1i, noise1])
                Xi = np.mean(Xi, axis=0) # mean output
                X.append(Xi)
            X = np.array(X)
        return np.squeeze(X)
    
    def synthesize_child(self, c2, airfoil=None):
        ''' Generate child give an airfoil '''
        if airfoil is None:
#            c1 = np.random.normal(scale=0.5, size=(1, self.latent_dim))
#            noise1 = np.random.normal(scale=0.5, size=(1, self.noise_dim))
#            airfoil = self.gen1.predict([c1, noise1])
            airfoil = self.x_test[np.random.choice(self.x_test.shape[0]), :self.x_test.shape[1]/2]
        X = []
        for i in range(c2.shape[0]):
            m = 10
            c2i = np.tile(c2[i], (m, 1))
            noise2 = np.random.normal(scale=0.5, size=(m, self.noise_dim))
            parent = np.tile(airfoil, (m,1,1,1))
            Xi = self.gen2.predict([c2i, noise2, parent])
            Xi = np.mean(Xi, axis=0) # mean output
            X.append(Xi)
        parent = np.tile(airfoil, (c2.shape[0],1,1,1))
        X = np.array(X)
        return np.squeeze(parent), np.squeeze(X)
    
    def synthesize_assembly(self, N):
        ''' Generate airfoil-hole combinations '''
        c1 = np.random.uniform(size=(N, self.latent_dim))
        noise1 = np.random.normal(scale=0.5, size=(N, self.noise_dim))
        parent = self.gen1.predict([c1, noise1])
        c2 = np.random.uniform(size=(N, self.latent_dim))
        noise2 = np.random.normal(scale=0.5, size=(N, self.noise_dim))
        child = self.gen2.predict([c2, noise2, parent])
        X = np.concatenate((parent, child), axis=1)
        return np.squeeze(X)
    
    def embed(self, X):
        _, c1, c2 = self.dis.predict(X)
        return c1[:,0,:], c2[:,0,:]
