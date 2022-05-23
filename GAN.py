import torch
from torch import nn
from torch.optim import Adam

from random import shuffle

from utils import device, plot_losses, plot_images, plot_acc, texts_to_hot
from get_data import get_data, get_image
from gen_dis import Generator, Discriminator, seed_size

class GAN:
    def __init__(self, d = 3):
        self.d = d
        self.layers = 1
        self.trans = False
        self.trans_level = 1
        self.trans_rate = .01
        self.non_trans_rate = .01
        self.gen = Generator()
        self.gen_opt = Adam(self.gen.parameters(), .001)

        self.dis = [Discriminator() for _ in range(d)]
        self.dis_opts = [Adam(dis.parameters(), .001) for dis in self.dis]
        
        self.bce = nn.BCELoss()
        
        self.display_labels, self.display_texts, _ = \
            get_data(9, 256, test = True)
        self.display_seeds = \
            torch.zeros((9,seed_size)).uniform_(-1, 1).to(device)
        
        self.gen_train_losses = []; self.gen_test_losses  = []
        self.dis_train_losses = [[] for _ in range(d)]; self.dis_test_losses  = [[] for _ in range(d)]
        self.train_fakes_acc =  [[] for _ in range(d)]; self.test_fakes_acc =   [[] for _ in range(d)]
        self.train_reals_acc =  [[] for _ in range(d)]; self.test_reals_acc =   [[] for _ in range(d)]
        
    def bigger_gen(self):
        if(self.trans):
            new_gen = Generator(self.layers, True)
        else:
            new_gen = Generator(self.layers, False)
        self.gen = new_gen
        self.gen_opt = Adam(self.gen.parameters(), .001)
 
    def bigger_dis(self):
        if(self.trans):
            new_dis = [Discriminator(self.layers, True) for _ in range(self.d)]
        else:
            new_dis = [Discriminator(self.layers, False) for _ in range(self.d)]
        self.dis = new_dis
        self.dis_opts = [Adam(dis.parameters(), .001) for dis in self.dis]
        
    def gen_epoch(self, seeds, texts_hot, test = False):
        self.gen.zero_grad()
        if(test): self.gen.eval()
        else:     self.gen.train()
        gen_images = self.gen(texts_hot, seeds)
        judgements = []
        for dis in self.dis:
            dis.eval()
            judgements.append(dis(texts_hot, gen_images))
        judgements = torch.cat(judgements, 1)
        loss = self.bce(judgements, torch.ones(judgements.shape).to(device))
        if(not test):
            loss.backward()
            self.gen_opt.step()
            self.gen_train_losses.append(loss.cpu().detach())
        else:
            self.gen_test_losses.append(loss.cpu().detach())
            
    def dis_epoch(self, seeds, d, texts_hot, images, noise, correct, noisy_correct, test = False):
        dis = self.dis[d]
        dis.zero_grad()
        if(test): self.gen.eval();  dis.eval()
        else:     self.gen.train(); dis.train()
        with torch.no_grad():
            gen_images = self.gen(texts_hot, seeds)
        texts_hot = torch.cat([texts_hot]*2, 0)
        images = torch.cat([images, gen_images], 0)
        noisy_images = images + noise
        judgement = dis(texts_hot, noisy_images)
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
            print("Epoch {}: {}x{} images. Transitioning: {} ({}).".format(
                e, 2**(self.layers+1), 2**(self.layers+1), 
                self.trans, round(self.trans_level,2)))
            _, train_texts, train_images = get_data(batch_size, 2**(self.layers+1), False)
            _, test_texts,  test_images  = get_data(batch_size, 2**(self.layers+1), True)
            train_texts_hot = texts_to_hot(train_texts)
            test_texts_hot = texts_to_hot(test_texts)
            seeds = torch.zeros((train_texts_hot.shape[0],seed_size)).uniform_(-1, 1).to(device)
            correct = torch.cat([
                .9*torch.ones((batch_size,1)),
                torch.zeros((batch_size,1))]).to(device)
            noisy_correct = correct; noisy_correct[0] = .1; noisy_correct[-1] = .9
            noise = torch.normal(
                torch.zeros((train_images.shape[0]*2,) + train_images.shape[1:]), 
                .05*torch.ones((train_images.shape[0]*2,) + train_images.shape[1:])).to(device)
            self.gen_epoch(seeds, train_texts_hot, test = False)
            self.gen_epoch(seeds, test_texts_hot,  test = True)
            for d in range(len(self.dis)):
                self.dis_epoch(seeds, d, train_texts_hot, train_images, noise, correct, noisy_correct, test = False)
                self.dis_epoch(seeds, d, test_texts_hot,  test_images,  noise, correct, noisy_correct, test = True)
            torch.cuda.synchronize()
            if(e%10 == 0):
                self.display()
            if(self.trans): 
                self.trans_level -= self.trans_rate 
                if(self.trans_level <= 0):
                    self.trans = False
                    self.trans_level = 1
                    self.bigger_gen()
                    self.bigger_dis()
            else:
                self.trans_level -= self.non_trans_rate
                if(self.trans_level <= 0):
                    self.trans = True
                    self.layers += 1
                    self.trans_level = 1
                    self.bigger_gen()
                    self.bigger_dis()
            if(self.layers >= 256):
                break

                    
            
            

    def display(self):
        plot_losses(self.gen_train_losses, self.gen_test_losses, "Generator Losses")
        for d in range(len(self.dis)):
            plot_losses(
                self.dis_train_losses[d], self.dis_test_losses[d], 
                "Discriminator {} Losses".format(d))
            plot_acc(d,
                self.train_fakes_acc[d], self.train_reals_acc[d], 
                self.test_fakes_acc[d],  self.test_reals_acc[d])
        print(self.display_texts)
        plot_images(
            [get_image(l, 2*(2**self.layers)) for l in self.display_labels], 3, 3)
        plot_images(self.gen(
            texts_to_hot(self.display_texts), 
            self.display_seeds).cpu().detach(), 3, 3)
        print()
