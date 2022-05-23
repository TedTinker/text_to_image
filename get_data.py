import torch

import numpy as np
import string
from random import sample, choice
try:
    from keras.preprocessing.image import load_img
except:
    from PIL import Image
import re

from utils import device, plot_image, data_file

train_data = np.loadtxt(
    data_file + "/images_train.txt", 
    dtype = str, 
    comments="!", 
    delimiter = "\n")
train_data.sort()

test_data = np.loadtxt(
    data_file + "/images_test.txt", 
    dtype = str, 
    comments="!", 
    delimiter = "\n")
test_data.sort()

text_raw = np.loadtxt(
    data_file + "/text.txt", 
    dtype = str, 
    comments="!", 
    delimiter = "\n")  
new_texts = []
for i in range(0,len(text_raw),5):
    new_text = [text_raw[i].split("\t")[0][:-2]]
    for j in [i, i+1, i+2, i+3, i+4]:
        t = text_raw[j].split("\t")[1]
        t = t.translate(str.maketrans("", "", string.punctuation))
        t = re.sub(" +", " ", t)
        new_text.append(t.lower())
    new_texts.append(new_text)
new_texts.sort(key=lambda t: t[0])
text = np.vstack(new_texts)

def get_image(label, size = 32):
    try:
        image = np.array(load_img(data_file + "/images/" + label, target_size=(size, size)))/255
    except:
        image = np.array(Image.open(data_file + "/images/" + label).resize((size, size)))/255
    return(image)

def get_data(batch_size = 64, size = 32, test = False):
    if(test): data = test_data 
    else:     data = train_data
    index = [i for i in range(len(data))]
    batch_index = sample(index, batch_size)
    batch_label = data[batch_index]
    batch_label.sort()
    texts = np.vstack([t[choice([1,2,3,4,5])] for t in text if t[0] in batch_label])
    images = np.vstack([np.expand_dims(get_image(label, size),0) for label in batch_label])
    return(batch_label, np.squeeze(texts,-1), torch.tensor(images).to(device).float())

if __name__ == "__main__":
    labels, texts, images = get_data(2, 256)
    for l, t, i in zip(labels, texts, images):
        print()
        print("{}: {}".format(l, t))
        plot_image(i)