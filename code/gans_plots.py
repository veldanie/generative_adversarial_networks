
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # for pretty plots
from scipy.stats import norm
from matplotlib import rc
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = 'lightgray'

#plt.rcParams
D = np.arange(0,1,0.01)

fig = plt.figure()
plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="off", right="off", labelleft="on")
plt.plot(D, np.log(1-D))
plt.plot(D, -np.log(D))
plt.xlabel(r'$D(G(\mathbf{z}))$')
plt.ylabel('Loss')
plt.legend([r'$-\log D(G(\mathbf{z}))$', r'$\log (1-D(G(\mathbf{z})))$'], loc = 'upper center')
plt.show()
fig.savefig('images/gen_loss.png')


##Generator Loss - KL and Reverse KL
fig = plt.figure()
r = np.arange(-5,5,0.1)
plt.plot(D/(1-D), -1-np.log(D/(1-D)))
#plt.plot(D, -np.log(D))
plt.xlabel(r'$D(G(\mathbf{z}))$')
plt.ylabel('Loss')
#plt.legend([r'$-\log D(G(\mathbf{z}))$', r'$\log (1-D(G(\mathbf{z})))$'], loc = 'upper center')
plt.show()
#fig.savefig('images/gen_loss.png')
