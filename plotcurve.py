import matplotlib.pyplot as plt
import numpy as np

num_mnist = 60000


plt.figure(dpi=300)

for latent_dim in [3,5,10,20,200]:
    loss =  np.load('./loss_data/'+str(latent_dim)+'.npy')
    num_samples = num_mnist * np.arange(loss.shape[0])

    plt.plot(num_samples, -loss/100, label='latent_dim = '+ str(latent_dim))
    plt.xscale('log')

    print(loss)
plt.ylim([-150,-100])
plt.legend()

plt.savefig('lossplot.png')