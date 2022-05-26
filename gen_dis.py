#%%
import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary as torch_summary
    
from utils import device, chars



def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass

class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)




seed_size = 256
class Generator(nn.Module):
    
    def __init__(self, layers = 1, transitioning = False):
        super().__init__()
        
        self.layers = layers
        self.trans = transitioning
        
        self.text_in = nn.Sequential(
            nn.Linear(len(chars), 128),
            nn.LeakyReLU()
            )
        
        self.lstm = nn.LSTM(
            input_size = 128,
            hidden_size = 256,
            batch_first = True
            )
        
        self.seed_in = nn.Sequential(
            nn.Linear(seed_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU()
            )
        
        self.lin = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(512 + 256, 2*2*128),
            nn.BatchNorm1d(2*2*128),
            nn.LeakyReLU()
            )
        
        self.cnn_list = nn.ModuleList()
        for i in range(layers):
            self.add_cnn()
            
        self.image_out = nn.Sequential(
                ConstrainedConv2d(
                    in_channels = 128, 
                    out_channels = 3, 
                    kernel_size = 3,
                    padding = (1,1),
                    padding_mode = "reflect"),
                nn.BatchNorm2d(3),
                nn.Tanh()
            )
        
        self.text_in.apply(init_weights).float()
        self.lstm.apply(init_weights).float()
        self.seed_in.apply(init_weights).float()
        self.lin.apply(init_weights).float()
        for cnn in self.cnn_list:
            cnn.apply(init_weights).float()
        self.image_out.apply(init_weights).float()
        self.to(device)
        
    def add_cnn(self):
        cnn = nn.Sequential(
            ConstrainedConv2d(
                in_channels = 128,
                out_channels = 128, 
                kernel_size = 3,
                padding = (1,1),
                padding_mode = "reflect"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True))
        cnn.apply(init_weights).float()
        self.cnn_list.append(cnn.to(device))        
        
    def freeze(self):
        keys = self.state_dict().keys()
        freezable_keys = [key for key in keys if any(map(key.__contains__, ["weight", "bias"]))]
        for i, (param, key) in enumerate(zip(self.parameters(), freezable_keys)):
            if(i > 5 and i < len(freezable_keys)-6):
                param.requires_grad = False
    
    def forward(self, text, seed, trans_level):
        x = self.text_in(text)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        seed = self.seed_in(seed)
        x = torch.cat([x, seed],-1)
        x = self.lin(x)
        x = x.reshape(x.shape[0], 128, 2, 2)
        for cnn in self.cnn_list[:-1]:
            x = cnn(x)
        if(not self.trans):
            x = self.cnn_list[-1](x)
        image = self.image_out(x)
        if(self.trans):
            x = self.cnn_list[-1](x)
            image_2 = self.image_out(x)
            image = F.interpolate(image, scale_factor = 2, mode = "nearest") #mode = "bilinear", align_corners = True)
            image = image*trans_level + image_2*(1-trans_level)
        image = (image + 1) / 2
        image = image.permute(0, 2, 3, 1)
        return(image)
    
if __name__ == "__main__":
    print("\n\n\n")
    layers = 2
    gen = Generator(layers = layers, transitioning = True)
    print(gen)
    print()
    print(torch_summary(gen, (
        (1, 1, len(chars)), 
        (1, seed_size),
        (1,1))))
    
    
    
    
class Discriminator(nn.Module):
    
    def __init__(self, layers = 1, transitioning = False):
        super().__init__()
        
        self.layers = layers
        self.trans = transitioning
        
        self.text_in = nn.Sequential(
            nn.Linear(len(chars), 128),
            nn.LeakyReLU()
            )
        
        self.lstm = nn.LSTM(
            input_size = 128,
            hidden_size = 256,
            batch_first = True
            )
        
        self.lin = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU()
            )
                
        self.image_in = nn.Sequential(
            ConstrainedConv2d(
                in_channels = 3,
                out_channels = 128, 
                kernel_size = 3,
                padding = (1,1),
                padding_mode = "reflect"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout(.2)
            ).to(device)
        
        self.cnn_list = nn.ModuleList()
        for i in range(layers):
            self.add_cnn()
            
        example = torch.zeros(1, 3, 2**(layers+1), 2**(layers+1)).to(device)
        example = self.image_in(example)
        for cnn in self.cnn_list:
            example = cnn(example)
        example = example.flatten(1)
        quantity = example.shape[1]
                    
        self.guess = nn.Sequential(
                    nn.Linear(quantity, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(),
                    nn.Dropout(.2),
                    nn.Linear(512, 1),
                    nn.BatchNorm1d(1),
                    nn.Tanh())
        
        self.text_in.apply(init_weights).float()
        self.lstm.apply(init_weights).float()
        self.lin.apply(init_weights).float()
        self.image_in.apply(init_weights).float()   
        for cnn in self.cnn_list:
            cnn.apply(init_weights).float()
        self.guess.apply(init_weights).float()
        self.to(device)
        
    def add_cnn(self):
        cnn = nn.Sequential(
            ConstrainedConv2d(
                in_channels = 128,
                out_channels = 128, 
                kernel_size = 3,
                stride = 2,
                padding = (1,1),
                padding_mode = "reflect"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout(.2))
        cnn.apply(init_weights).float()
        self.cnn_list.insert(0,cnn.to(device))
        
    def freeze(self):
        keys = self.state_dict().keys()
        freezable_keys = [key for key in keys if any(map(key.__contains__, ["weight", "bias"]))]
        for i, (param, key) in enumerate(zip(self.parameters(), freezable_keys)):
            if(i >= 12):
                param.requires_grad = False
        
    def forward(self, text, image, trans_level):
        x = self.text_in(text)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = self.lin(x)
        image = (image.permute(0, -1, 1, 2) * 2) - 1
        if(self.trans):
            image_2 = self.image_in(image)
            image_2 = self.cnn_list[0](image_2)
            image = F.interpolate(image, scale_factor = .5, mode = "bilinear", align_corners = True, recompute_scale_factor=False)
            image = self.image_in(image)
            image = image*trans_level + image_2*(1-trans_level)
            for cnn in self.cnn_list[1:]:
                image = cnn(image)
        else:
            image = self.image_in(image)
            for cnn in self.cnn_list:
                image = cnn(image)
        image = image.flatten(1)
        x = torch.cat([x, image], -1)
        x = (self.guess(image) + 1)/2
        return(x)
    
if __name__ == "__main__":
    print("\n\n\n")
    layers = 4
    dis = Discriminator(layers = layers, transitioning = True)
    print(dis)
    print()
    print(torch_summary(dis, (
        (1, 1, len(chars)), 
        (1,2*(2**layers), 2*(2**layers),3),
        (1,1))))
    
    


# %%
