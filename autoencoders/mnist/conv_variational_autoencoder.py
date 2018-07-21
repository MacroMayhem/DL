import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input,Conv2D, MaxPooling2D, Flatten,Dense,Lambda,Conv2DTranspose,Reshape,add
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist




# Hyperparameters
batch_size = 100
original_shape = (28,28,1)
latent_dim = 24
nb_epoch = 10

inputs = Input(batch_shape = (batch_size, 28,28,1))
cv_1   = Conv2D(filters=8,kernel_size=(3,3),padding='same',name='cv_1')(inputs)
ds_1   = MaxPooling2D(pool_size=(2,2),padding='same',name='ds_1')(cv_1)
cv_2   = Conv2D(filters=4,kernel_size=(3,3),padding='same',name='cv_2')(ds_1)
ds_2   = MaxPooling2D(pool_size=(2,2),padding='same',name='ds_2')(cv_2)
dn_1   = Dense(1,name='dn_1')(ds_2)
en_inp = Flatten()(dn_1)

mu = Dense(latent_dim,activation='relu')(en_inp)
sig= Dense(latent_dim,activation='relu')(en_inp)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size,latent_dim), mean=0.,stddev=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon

dec_input = Lambda(sampling)([mu,sig])

encoder = Model(inputs,[mu,sig])
encoder.summary()

layer_d_1 = Dense(24,activation='relu')
layer_d_2 = Dense(49,activation='relu')
layer_rs_d_1 = Reshape((7,7,-1))
layer_tc_1 = Conv2DTranspose(filters=4,kernel_size=(3,3),strides=(2,2),padding='same',name='tc_1')
layer_tc_2 = Conv2DTranspose(filters=8,kernel_size=(3,3),strides=(2,2),padding='same',name='tc_2')
layer_dec_output = Dense(1,activation='sigmoid',name='dec_output')

d_1 = layer_d_1(dec_input)
d_2 = layer_d_2(d_1)
rs_d_1 = layer_rs_d_1(d_2)
tc_1 = layer_tc_1(rs_d_1)
tc_2 = layer_tc_2(tc_1)
dec_output = layer_dec_output(tc_2)

exp_dec_input = Input(shape=(latent_dim,),name='dec_inp')
exp_d_1 = layer_d_1(exp_dec_input)
exp_d_2 = layer_d_2(exp_d_1)
exp_rs_d_1 = layer_rs_d_1(exp_d_2)
exp_tc_1 = layer_tc_1(exp_rs_d_1)
exp_tc_2 = layer_tc_2(exp_tc_1)
exp_dec_output = layer_dec_output(exp_tc_2)



#loss
def vae_loss(x, x_rec):
    minus_r2 = Lambda(lambda x:-x)(x_rec)
    subtracted = add([x,minus_r2])
    sq_err = Lambda(lambda x:x**2)(subtracted)
    rec_loss = K.sum(sq_err)
    kl_loss = - 0.5 * K.sum(1 + sig - K.square(mu) - K.exp(sig), axis=-1)
    return rec_loss + kl_loss

vae = Model(inputs, dec_output)
vae.compile(optimizer='rmsprop', loss=vae_loss)
vae.summary()



decoder = Model(exp_dec_input,exp_dec_output)
decoder.summary()

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), 28,28,1))
x_test = x_test.reshape((len(x_test),28,28,1))


vae.fit(x_train, x_train,
 shuffle=True,
 epochs=nb_epoch,
 batch_size=batch_size,
 validation_data=(x_test, x_test),verbose=1)

print('Learning Done!')

# build a model to project inputs on the latent space
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import random
from matplotlib.pyplot import get_cmap
# display a 2D plot of the digit classes in the latent space
x_test_encoded_mu,x_test_encoded_sig = encoder.predict(x_test, batch_size=batch_size)

lst = random.sample(range(0,len(y_test)),int(len(y_test)/5))

x_selected = [x_test_encoded_mu[i] for i in lst]
y_selected = [y_test[i] for i in lst]

tsne = TSNE(2,init='pca',random_state=0)
x_tsne = tsne.fit_transform(x_selected)

target_ids = range(len(np.unique(y_test)))

col_label = []
legends = []
cmp = get_cmap('tab20')
for tid in y_selected:
    col_label.append(cmp(tid/len(target_ids))[:3])
    legends.append(str(tid))


plt.figure(figsize=(6, 5))
plt.scatter(x_tsne[:,0], x_tsne[:,1],c=y_selected)
plt.colorbar()
plt.savefig('../output_img/conv_vae_latent_plot.png')
#plt.show()
 

n = 10
digit_size = 28
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian

randoms = []
for i in range(0,10):
    indices = np.where(y_test == i)
    u = np.mean(np.asarray([x_test_encoded_mu[i]for ind in indices]),axis=0)
    s = np.mean(np.asarray([x_test_encoded_sig[i] for ind in indices]),axis=0)
    randoms.append(u+np.random.normal(0, 1, latent_dim)*s)

fig=plt.figure(figsize=(15,4))

imgs = decoder.predict(np.asarray(randoms))
print(imgs.shape)
for i in range(1,n+1):
    ax = fig.add_subplot(1, n, i)
    ax.imshow(imgs[i-1].reshape(28,28),cmap='gray')


plt.savefig('../output_img/conv_vae_dream.png')

#plt.show()
