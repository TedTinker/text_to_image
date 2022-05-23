from GAN import GAN

gan = GAN(d = 3)
gan.train(epochs = 100000, batch_size = 256)