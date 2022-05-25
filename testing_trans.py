#%%
from torch import nn 
from torchinfo import summary as torch_summary

class First(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.lin_1 = nn.Sequential(
            nn.Linear(3,2),
            nn.Linear(2,2)
        )
        
        self.lin_list = nn.ModuleList()
        for i in range(3):
            self.lin_list.append(
                nn.Linear(2,2)
            )
        
    def forward(self, x):
        x = self.lin_1(x)
        for lin in self.lin_list:
            x = lin(x)
        return(x)
    
first = First()
print(first)
print()
print(torch_summary(first, (1,3)))
# %%
class Second(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.lin_1 = nn.Sequential(
            nn.Linear(3,2),
            nn.Linear(2,2)
        )
        
        self.lin_list = nn.ModuleList()
        for i in range(3):
            self.lin_list.append(
                nn.Linear(2,2)
            )
            
        self.lin_2 = nn.Linear(2,1)
        
    def forward(self, x):
        x = self.lin_1(x)
        for lin in self.lin_list:
            x = lin(x)
        x = self.lin_2(x)
        return(x)
    
second = Second()
print(second)
print()
print(torch_summary(second, (1,3)))
# %%

def print_state_dict(string, state_dict):
    print("\n{}".format(string))
    for key in state_dict.keys():
        print("{} : {}".format(key, state_dict[key]))

def replace_paras(old, new):
    old_state_dict = old.state_dict()
    old_keys = old_state_dict.keys()
    new_state_dict = new.state_dict()
    print_state_dict("Old", old_state_dict)
    print_state_dict("To change", new_state_dict)
    print("\n\n")
    for key in new_state_dict.keys():
        print("Checking...", key)
        if key in old_keys:
            print("\tIt's here!")
            new_state_dict[key] = old_state_dict[key]
    new.load_state_dict(new_state_dict)
    print_state_dict("New", new.state_dict())

replace_paras(first, second)
# %%
