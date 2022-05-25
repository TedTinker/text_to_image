#%%
from GAN import GAN

gan = GAN(d = 1)
gan.train(epochs = 100000, batch_size = 256)
# %%
