### A few utilities
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal



import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if(device.type == "cpu"):   print("\n\nAAAAAAAUGH! CPU! >:C\n")
else:                       print("\n\nUsing CUDA! :D\n")



data_file = r"C:\Users\tedjt\Desktop\data"
#data_file = "/home/ted/Desktop/data"


from string import ascii_lowercase
chars = ascii_lowercase + " 1234567890"

cap_len = 64

def char_to_hot(c):
    try:
        index = chars.index(c)
    except:
        print("\n\n{}\n\n".format(c))
        assert False
    h = [0 for _ in range(len(chars))]
    h[index] = 1
    return(h)

def text_to_hot(t):
    h = []
    for c in t:
        h.append(char_to_hot(c))
    while(len(h) < cap_len): h.append(char_to_hot(" "))
    if(len(h) > cap_len):    h = h[:cap_len]
    return(h)

def texts_to_hot(texts):
    h = []
    for t in texts:
        h.append(text_to_hot(t))
    return(torch.tensor(h).to(device).float())




def plot_losses(changes, train_losses, test_losses, title):
    for i, c in enumerate(changes):
        if(c):
            plt.axvline(x=i, color = (0,0,0,.4), linestyle='dashed')
    plt.plot(train_losses, label = "training", color = "red")
    plt.plot(test_losses,  label = "testing",  color = "pink")
    plt.legend(loc = 'upper left')
    plt.title(title)
    plt.show()
    plt.close()
    
def plot_acc(changes, d, train_fakes_acc, train_reals_acc, test_fakes_acc, test_reals_acc):
    for i, c in enumerate(changes):
        if(c):
            plt.axvline(x=i, color = (0,0,0,.4), linestyle='dashed')
    plt.plot(train_fakes_acc, label = "train fake acc", color = "red")
    plt.plot(train_reals_acc, label = "train real acc", color = "blue")
    plt.plot(test_fakes_acc,  label = "test  fake acc", color = "pink")
    plt.plot(test_reals_acc,  label = "test  real acc", color = "dodgerblue")
    plt.ylim([0,1])
    plt.legend(loc = 'upper left')
    plt.title("Discriminator {} Accuracy".format(d))
    plt.show()
    plt.close()
    
def plot_image(image):
    image = image.cpu().detach()
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    plt.close()
    
def plot_images(images, rows, columns):
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i-1])
        plt.axis("off")
    plt.show()
