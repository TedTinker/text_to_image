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
        self.lr = .0002
        self.layers = 1
        self.trans = False
        self.trans_level = 1
        self.trans_rate = .01
        self.non_trans_rate = .01
        self.changes = []
        
        self.gen = Generator()
        self.gen_opt = Adam(self.gen.parameters(), self.lr)

        self.dis = [Discriminator() for _ in range(d)]
        self.dis_opts = [Adam(dis.parameters(), self.lr) for dis in self.dis]
        
        self.bce = nn.BCELoss()
        
        self.display_labels, self.display_texts, _ = \
            get_data(9, 256, test = True)
        self.display_seeds = self.get_seeds(9)
        
        self.gen_train_losses = []; self.gen_test_losses  = []
        self.dis_train_losses = [[] for _ in range(d)]; self.dis_test_losses  = [[] for _ in range(d)]
        self.train_fakes_acc =  [[] for _ in range(d)]; self.test_fakes_acc =   [[] for _ in range(d)]
        self.train_reals_acc =  [[] for _ in range(d)]; self.test_reals_acc =   [[] for _ in range(d)]
        
    def get_seeds(self, batch_size):
        return(torch.normal(0, 1, size = (batch_size, seed_size))).to(device)
    
    # Also, freeze old layers!
    def bigger_gen(self):
        old_state_dict = self.gen.state_dict()
        old_keys = old_state_dict.keys()
        if(self.trans):
            new_gen = Generator(self.layers, True)
        else:
            new_gen = Generator(self.layers, False)
        new_state_dict = new_gen.state_dict()
        for key in new_state_dict.keys():
            if key in old_keys:
                new_state_dict[key] = old_state_dict[key]
        new_gen.load_state_dict(new_state_dict)
        self.gen = new_gen
        self.gen_opt = Adam(self.gen.parameters(), self.lr)
        
    def bigger_dises(self):
        for d in range(self.d):
            new_dis, new_opts = self.bigger_dis(self.dis[d])
            self.dis[d] = new_dis 
            self.dis_opts[d] = new_opts
 
    def bigger_dis(self, dis):
        old_state_dict = dis.state_dict()
        old_keys = old_state_dict.keys()
        if(self.trans):
            new_dis = Discriminator(self.layers, True)
        else:
            new_dis = Discriminator(self.layers, False)
        new_state_dict = new_dis.state_dict()
        for key in new_state_dict.keys():
            if key in old_keys:
                new_state_dict[key] = old_state_dict[key]
        new_dis.load_state_dict(new_state_dict)
        dis_opts = Adam(new_dis.parameters(), self.lr)
        return(new_dis, dis_opts)
        
    def gen_epoch(self, seeds, texts_hot, test = False):
        self.gen.zero_grad()
        if(test): self.gen.eval()
        else:     self.gen.train()
        gen_images = self.gen(texts_hot, seeds, self.trans_level)
        judgements = []
        for dis in self.dis:
            dis.eval()
            judgements.append(dis(texts_hot, gen_images, self.trans_level))
        judgements = torch.cat(judgements, 1)
        loss = self.bce(judgements, torch.ones(judgements.shape).to(device))
        if(not test):
            loss.backward()
            self.gen_opt.step()
            self.gen_train_losses.append(loss.cpu().detach())
        else:
            self.gen_test_losses.append(loss.cpu().detach())
            
    def dis_epoch(self, d, gen_images, texts_hot, images, noise, real_correct, fake_correct, dis_batch, test = False):
        dis = self.dis[d]
        dis.zero_grad()
        if(test): dis.eval()
        else:     dis.train()
        texts_hot = torch.cat([texts_hot]*2, 0)
        images = torch.cat([images, gen_images], 0)
        noisy_images = images + noise
        judgement = dis(texts_hot, noisy_images, self.trans_level)
        correct = torch.cat([real_correct, fake_correct])
        loss = self.bce(judgement, correct)
        reals_correct = [1 if round(judgement[i].item()) == round(correct[i].item()) else 0 for i in range(len(judgement)) if round(correct[i].item()) == 1]        
        fakes_correct = [1 if round(judgement[i].item()) == round(correct[i].item()) else 0 for i in range(len(judgement)) if round(correct[i].item()) == 0]
        reals_accuracy = sum(reals_correct)/len(reals_correct)
        fakes_accuracy = sum(fakes_correct)/len(fakes_correct)
        if(not test):
            loss.backward()
            self.dis_opts[d].step()
            self.dis_train_losses[d].append(loss.cpu().detach())
            self.train_reals_acc[d].append(reals_accuracy)
            self.train_fakes_acc[d].append(fakes_accuracy)
        else:
            self.dis_test_losses[d].append(loss.cpu().detach())
            self.test_reals_acc[d].append(reals_accuracy)
            self.test_fakes_acc[d].append(fakes_accuracy)
            
            
                
    def train(self, epochs = 100, batch_size = 64):
        dis_batch = True
        for e in range(epochs):
            
            if(e%10 == 0 or e == 0):
                self.display()
                
            print("Epoch {}: {}x{} images. Transitioning: {} ({}).\n\tDiscriminator trains on {}.".format(
                e, 2**(self.layers+1), 2**(self.layers+1), 
                self.trans, round(self.trans_level,2), "real images" if dis_batch else "fakes"))
            
            _, train_texts, train_images = get_data(batch_size, 2**(self.layers+1), False)
            _, test_texts,  test_images  = get_data(batch_size, 2**(self.layers+1), True)
            train_texts_hot = texts_to_hot(train_texts)
            test_texts_hot  = texts_to_hot(test_texts)
            train_seeds = self.get_seeds(batch_size)
            test_seeds  = self.get_seeds(batch_size)
            
            self.gen_epoch(train_seeds, train_texts_hot, test = False)
            self.gen_epoch(test_seeds,  test_texts_hot,  test = True)
            
            with torch.no_grad():
                self.gen.train()
                train_gen_images = self.gen(train_texts_hot, train_seeds, self.trans_level)
                self.gen.eval()
                test_gen_images  = self.gen(test_texts_hot,  test_seeds,  self.trans_level)
            real_correct = .9*torch.ones((batch_size,1)).to(device)
            fake_correct = torch.zeros(  (batch_size,1)).to(device)
            noise = torch.normal(
                torch.zeros((train_images.shape[0]*2,) + train_images.shape[1:]), 
                .05*torch.ones((train_images.shape[0]*2,) + train_images.shape[1:])).to(device)
            
            for d in range(len(self.dis)):
                self.dis_epoch(d, train_gen_images, train_texts_hot, train_images, 
                               noise, real_correct, fake_correct, dis_batch, test = False)
                self.dis_epoch(d, test_gen_images,  test_texts_hot,  test_images,  
                               noise, real_correct, fake_correct, dis_batch, test = True)
            dis_batch = not dis_batch
            torch.cuda.synchronize()
            
            if(self.trans): 
                self.trans_level -= self.trans_rate 
                if(self.trans_level <= 0):
                    print("\n\nEnd of transitioning:")
                    self.just_pics_display()
                    self.changes.append(True)
                    self.trans = False
                    self.trans_level = 1
                    self.bigger_gen()
                    self.bigger_dises()
                    print("Begin not transitioning:")
                    self.just_pics_display()
                else:
                    self.changes.append(False)
            else:
                self.trans_level -= self.non_trans_rate
                if(self.trans_level <= 0):
                    print("\n\nEnd of not transitioning:")
                    self.just_pics_display()
                    self.changes.append(True)
                    self.trans = True
                    self.layers += 1
                    self.trans_level = 1
                    self.bigger_gen()
                    self.bigger_dises()
                    print("Begin transitioning:")
                    self.just_pics_display()
                else:
                    self.changes.append(False)
            if(self.layers >= 256):
                break



    def display(self):
        plot_losses(self.changes,self.gen_train_losses, self.gen_test_losses, "Generator Losses")
        for d in range(len(self.dis)):
            plot_losses(self.changes,
                self.dis_train_losses[d], self.dis_test_losses[d], 
                "Discriminator {} Losses".format(d))
            plot_acc(self.changes, d,
                self.train_fakes_acc[d], self.train_reals_acc[d], 
                self.test_fakes_acc[d],  self.test_reals_acc[d])
        print(self.display_texts)
        plot_images(
            [get_image(l, 2*(2**self.layers)) for l in self.display_labels], 3, 3)
        plot_images(self.gen(
            texts_to_hot(self.display_texts), 
            self.display_seeds, self.trans_level).cpu().detach(), 3, 3)
        print()
        
    def just_pics_display(self):
        plot_images(self.gen(
            texts_to_hot(self.display_texts), 
            self.display_seeds, self.trans_level).cpu().detach(), 3, 3)
        print()

