
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # for pretty plots
from scipy.stats import norm

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

mu,sigma=4,0.5
z_range = 8


xs=np.linspace(mu-z_range,mu+z_range,1000)
plt.plot(xs, norm.pdf(xs,loc=mu,scale=sigma))
plt.show()
#plt.savefig('fig0.png')

TRAIN_ITERS=1200
hidden_size = 4
M=12 # minibatch size


# MLP - used for D_pre, D1, D2, G networks
def linear(input, output_dim, scope=None, stddev=1.0):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b

def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    #h1 = tf.nn.softplus(linear(h0, h_dim, 'g1'))
    h1 = linear(h0, 1, 'g1')
    return h1

def discriminator(input, h_dim, minibatch_layer=False):
    h0 = tf.nn.tanh(linear(input, h_dim * 2, 'd0'))
    h1 = tf.nn.tanh(linear(h0, h_dim * 2, 'd1'))

    # without the minibatch layer, the discriminator needs an additional layer
    # to have enough capacity to separate the two distributions correctly
    if minibatch_layer:
        h2 = minibatch(h1)
    else:
        h2 = tf.nn.tanh(linear(h1, h_dim * 2, 'd2'))

    h3 = tf.sigmoid(linear(h2, 1, 'd3'))
    return h3

def minibatch(input, num_kernels=5, kernel_dim=3):
    x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input, minibatch_features], 1)

# re-used for optimizing all networks
#tf.train.exponential_decay: This function start with an certain learning rate,
#and then the function applies and exp decay.

def momentum_optimizer(loss,var_list):
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.001,                # Base learning rate.
        batch,  # Current index into the dataset.
        TRAIN_ITERS // 4,          # Decay step - this decays 4 times throughout training process.
        0.95,                # Decay rate.
        staircase=True)
    #optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=batch,var_list=var_list)
    optimizer=tf.train.MomentumOptimizer(learning_rate,0.6).minimize(loss,global_step=batch,var_list=var_list)
    return optimizer

def grad_optimizer(loss, var_list):
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.03,
        batch,
        150,
        0.95,
        staircase=True
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=batch,var_list=var_list)
    return optimizer

def adam_optimizer(loss, var_list):
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        learning_rate = 0.001,
        global_step = batch,
        decay_steps = 250,
        decay_rate = 0.95,
        staircase=True
    )
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss,global_step=batch,var_list=var_list)
    return optimizer


#Pre-train Decision Surface (Discriminator??)
with tf.variable_scope("D_pre"):
    input_node=tf.placeholder(tf.float32, shape=(None,1))
    train_labels=tf.placeholder(tf.float32,shape=(None,1))
    D = discriminator(input_node, hidden_size)
    loss=tf.reduce_mean(tf.square(D-train_labels)) #Loss is difference bet. GAN output and Gauss Density!

#optimizer=momentum_optimizer(loss,None)
optimizer = grad_optimizer(loss, None)
#optimizer = adam_optimizer(loss, None)


# plot decision surface
def plot_d0(D):
    f,ax=plt.subplots(1)
    # p_data
    xs=np.linspace(-z_range,z_range,1000)
    ax.plot(xs, norm.pdf(xs,loc=mu,scale=sigma), label='p_data')
    # decision boundary
    r=1000 # resolution (number of points)
    xs=np.linspace(-z_range,z_range,r)
    ds=np.zeros((r,1)) # decision surface
    # process multiple points in parallel in a minibatch
    for i in range(int(r/M)):
        x=np.reshape(xs[M*i:M*(i+1)],(M,1))
        ds[M*i:M*(i+1)]=sess.run(D,{input_node: x})


    ax.plot(xs, ds, label='Decision Boundary')
    ax.set_ylim(0,1.5)
    plt.legend()


# BUILD NET
# Now to build the actual generative adversarial network

##Neural Network for the generator:
with tf.variable_scope("Gen"):
    z_node=tf.placeholder(tf.float32, shape=(None,1)) # M uniform01 floats
    G=generator(z_node,hidden_size) # generate normal transformation of Z


##Neural network for the Discriminator:

with tf.variable_scope("Disc") as scope:
    # D(x)
    x_node=tf.placeholder(tf.float32, shape=(None,1)) # input M normally distributed floats
    D1=discriminator(x_node,hidden_size) # output likelihood of being normally distributed
    # make a copy of D that uses the same variables, but takes in G as input
    scope.reuse_variables()#Variables in scope D are reused.
    D2=discriminator(G,hidden_size) #G is the output of a graph in a different scope.


obj_d=tf.reduce_mean(-tf.log(D1)-tf.log(1-D2))
obj_g=tf.reduce_mean(-tf.log(D2))
#obj_g2=tf.reduce_mean(-tf.log(D2/(1-D2)))
# copy weights from pre-training over to new D network.
d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_pre')
d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')



# set up optimizer for G,D
opt_d=grad_optimizer(obj_d, d_params)
opt_g=grad_optimizer(obj_g, g_params) # maximize log(D(G(z)))
#opt_g=adam_optimizer(obj_g2, g_params) # maximize log(D(G(z)))

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
plot_d0(D)
plt.title('Initial Decision Boundary')
plt.show()
#plt.savefig('fig1.png')


# process multiple points in parallel in a minibatch

lh=np.zeros(1000)
for i in range(1000):
    #d=np.random.normal(mu,sigma,M)
    d=(np.random.random(M)-0.5) * 10.0 # Uniform between [-5,5]
    labels=norm.pdf(d,loc=mu,scale=sigma) # For each unif. we assign a label co
    lh[i],_=sess.run([loss,optimizer], {input_node: np.reshape(d,(M,1)), train_labels: np.reshape(labels,(M,1))})

##In this case to each z uniform [-5,5], the generator estimates a probability.

# training loss
plt.plot(lh)
plt.title('Training Loss')

plot_d0(D)
#D: Variable created by function MLP.
# input_node: z, [-5,5]

plt.show()
#plt.savefig('fig2.png')

# pre-training Parameters:
weightsD = sess.run(d_pre_params)

for i, v in enumerate(d_params):
    sess.run(v.assign(weightsD[i]))



def plot_fig():
    # plots pg, pdata, decision boundary
    f,ax=plt.subplots(1)
    # p_data
    xs=np.linspace(-z_range,z_range,1000)
    ax.plot(xs, norm.pdf(xs,loc=mu,scale=sigma), label='p_data')

    # decision boundary
    r=5000 # resolution (number of points)
    xs=np.linspace(-z_range,z_range,r)
    ds=np.zeros((r,1)) # decision surface
    # process multiple points in parallel in same minibatch
    for i in range(int(r/M)):
        x=np.reshape(xs[M*i:M*(i+1)],(M,1))
        ds[M*i:M*(i+1)]=sess.run(D1,{x_node: x})

    ax.plot(xs, ds, label='decision boundary')

    # distribution of inverse-mapped points
    zs=np.linspace(-z_range,z_range,r)
    gs=np.zeros((r,1)) # generator function
    for i in range(int(r/M)):
        z=np.reshape(zs[M*i:M*(i+1)],(M,1))
        gs[M*i:M*(i+1)]=sess.run(G,{z_node: z})
    histc, edges = np.histogram(gs, bins = 10)
    ax.plot(np.linspace(-z_range,z_range,10), histc/float(r), label='p_g')

    # ylim, legend
    ax.set_ylim(0,1.5)
    plt.legend()

# initial conditions
plot_fig()
plt.title('Before Training')
plt.show()
#plt.savefig('fig3.png')

# Algorithm 1 of Goodfellow et al 2014
k=1 #Times we train the discriminator for each run of the generator.
histd, histg= np.zeros(TRAIN_ITERS), np.zeros(TRAIN_ITERS)
M=1200
TRAIN_ITERS = 12000
#histd and histf save the history of the loss for the discriminator and the generator.
for i in range(TRAIN_ITERS):
    for j in range(k):
        x= np.random.normal(mu,sigma,M) # sampled m-batch from p_data
        x.sort()
        #z = np.random.normal(0,1,M)
        #z.sort()
        z= np.linspace(-z_range,z_range,M)+np.random.random(M)*0.01  # sample m-batch from noise prior
        histd[i],_=sess.run([obj_d, opt_d], {x_node: np.reshape(x,(M,1)), z_node: np.reshape(z,(M,1))})
    z= np.linspace(-z_range,z_range,M)+np.random.random(M)*0.01 # sample noise prior
    histg[i],_=sess.run([obj_g,opt_g], {z_node: np.reshape(z,(M,1))}) # update generator
    if i % (TRAIN_ITERS//10) == 0:
        print(float(i)/float(TRAIN_ITERS), histd[i], histg[i])

plot_fig()
plt.show()

#plt.savefig('fig5.png')

plt.plot(range(TRAIN_ITERS),histd, label='obj_d')
plt.plot(range(TRAIN_ITERS), 1-histg, label='obj_g')
plt.legend()
plt.show()
#plt.savefig('fig4.png')

sess.close()
