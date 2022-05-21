import torch
from torch import nn
from torch.optim import Adam

from random import shuffle

from utils import plot_losses, plot_images, plot_acc, texts_to_hot
from get_data import get_data, get_image
from gen_dis import Generator, Discriminator, seed_size

class GAN:
    def __init__(self, d = 3):
        self.layers = 1
        self.gen = Generator()
        self.gen_opt = Adam(self.gen.parameters(), .001)

        self.dis = [Discriminator() for _ in range(d)]
        self.dis_opts = [Adam(dis.parameters(), .001) for dis in self.dis]
        
        self.bce = nn.BCELoss()
        
        self.display_labels, self.display_texts, _ = \
            get_data(9, 256, test = True)
        self.display_seeds = \
            torch.zeros((9,seed_size)).uniform_(-1, 1)
        
        self.gen_train_losses = []; self.gen_test_losses  = []
        self.dis_train_losses = [[] for _ in range(d)]; self.dis_test_losses  = [[] for _ in range(d)]
        self.train_fakes_acc =  [[] for _ in range(d)]; self.test_fakes_acc =   [[] for _ in range(d)]
        self.train_reals_acc =  [[] for _ in range(d)]; self.test_reals_acc =   [[] for _ in range(d)]
        
    def bigger_gen(gen, transitioning = True):
        if(transitioning):
            new_gen = Generator(gen.layers+1, 1)
        else:
            new_gen = Generator(gen.layers, False)
        return(new_gen)
    
    def bigger_dis(dis, transitioning = True):
        if(transitioning):
            new_dis = Discriminator(dis.layers+1, 1)
        else:
            new_dis = Discriminator(dis.layers, False)
        return(new_dis)
        
    def gen_epoch(self, batch_size, test = False):
        self.gen.zero_grad()
        if(test): self.gen.eval()
        else:     self.gen.train()
        _, texts, images = get_data(batch_size, 2*(2**self.layers), test)
        texts_hot = texts_to_hot(texts)
        index = [i for i in range(len(texts))]
        shuffle(index)
        texts = texts[index]; images = images[index]
        seeds = torch.zeros((texts.shape[0],seed_size)).uniform_(-1, 1)
        gen_images = self.gen(texts_hot, seeds)
        judgements = []
        for dis in self.dis:
            dis.eval()
            judgements.append(dis(texts_hot, gen_images))
        judgements = torch.cat(judgements, 1)
        loss = self.bce(judgements, torch.ones(judgements.shape))
        if(not test):
            loss.backward()
            self.gen_opt.step()
            self.gen_train_losses.append(loss.cpu().detach())
        else:
            self.gen_test_losses.append(loss.cpu().detach())
            
    def dis_epoch(self, d, batch_size, test = False):
        dis = self.dis[d]
        dis.zero_grad()
        if(test): self.gen.eval();  dis.eval()
        else:     self.gen.train(); dis.train()
        _, texts, images = get_data(batch_size//2, 2*(2**self.layers), test)
        texts_hot = texts_to_hot(texts)
        index = [i for i in range(len(texts))]
        shuffle(index)
        texts = texts[index]; images = images[index]
        seeds = torch.zeros((texts.shape[0],seed_size)).uniform_(-1, 1)
        with torch.no_grad():
            gen_images = self.gen(texts_hot, seeds)
        texts_hot = torch.cat([texts_hot]*2, 0)
        images = torch.cat([images, gen_images], 0)
        noise = torch.normal(torch.zeros(images.shape), .05*torch.ones(images.shape))
        noisy_images = images + noise
        index = [i for i in range(len(texts_hot))]
        shuffle(index)
        texts_hot = texts_hot[index]; noisy_images = noisy_images[index]
        judgement = dis(texts_hot, noisy_images)
        correct = torch.cat([
            .9*torch.ones((batch_size//2,1)),
            .1*torch.ones((batch_size//2,1))])
        noisy_correct = correct
        noisy_correct[0] = .1; noisy_correct[-1] = .9
        correct = correct[index]
        noisy_correct = noisy_correct[index]
        loss = self.bce(judgement, noisy_correct)
        fakes_correct = [1 if round(judgement[i].item()) == round(correct[i].item()) else 0 for i in range(len(judgement)) if round(correct[i].item()) == 1]
        reals_correct = [1 if round(judgement[i].item()) == round(correct[i].item()) else 0 for i in range(len(judgement)) if round(correct[i].item()) == 0]        
        fakes_accuracy = sum(fakes_correct)/len(fakes_correct)
        reals_accuracy = sum(reals_correct)/len(reals_correct)
        if(not test):
            loss.backward()
            self.dis_opts[d].step()
            self.dis_train_losses[d].append(loss.cpu().detach())
            self.train_fakes_acc[d].append(fakes_accuracy)
            self.train_reals_acc[d].append(reals_accuracy)
        else:
            self.dis_test_losses[d].append(loss.cpu().detach())
            self.test_fakes_acc[d].append(fakes_accuracy)
            self.test_reals_acc[d].append(reals_accuracy)
            
            
                
    def train(self, epochs = 100, batch_size = 64):
        for e in range(epochs):
            self.gen_epoch(batch_size)
            self.gen_epoch(batch_size, test = True)
            for d in range(len(self.dis)):
                self.dis_epoch(d, batch_size)
                self.dis_epoch(d, batch_size, test = True)
            if(e%10 == 0):
                self.display()
            
            
            

    def display(self):
        plot_losses(self.gen_train_losses, self.gen_test_losses, "Generator Losses")
        for d in range(len(self.dis)):
            plot_losses(
                self.dis_train_losses[d], self.dis_test_losses[d], 
                "Discriminator {} Losses".format(d))
            plot_acc(d,
                self.train_fakes_acc[d], self.train_reals_acc[d], 
                self.test_fakes_acc[d],  self.test_reals_acc[d])
        plot_images(
            [get_image(l, 2*(2**self.layers)) for l in self.display_labels], 3, 3)
        plot_images(self.gen(
            texts_to_hot(self.display_texts), 
            self.display_seeds).cpu().detach(), 3, 3)
        print()
