import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from PIL import Image
import numpy as np
from keras.models import Sequential
import keras
from keras import layers


randomlist =[]

for i in range(0,10):
    randomlist.append(random.randint(0,3999))

total_imgs = dict()
for item in randomlist:
    temp = []
    nm_of_imgs=0
    for filename in glob.glob("train_data/"+str(item)+"/*.jpg"):
        img = mpimg.imread(filename)
        gray = np.dot(img[...,:3],[0.2989, 0.5870, 0.1140]) # conversion to grayscale
        gray = np.reshape(gray, 4096)
        temp.append(gray)
        nm_of_imgs += 1
        if nm_of_imgs==50:
            break

    total_imgs[item] = temp
    
folders = []
for folder in total_imgs:
    images_of_folder = []
    for img in total_imgs[folder]:
        folders.append(img)
        
all_photos_100 = np.array(folders)
all_photos_50 = np.array(folders)
all_photos_25 = np.array(folders)

print("----------------------------- 100 -------------------------------------")

input_img = keras.Input(shape=(4096,))
encoded = layers.Dense(4096/4, activation='relu')(input_img)
encoded = layers.Dense(100, activation='relu')(encoded)  
decoded = layers.Dense(4096/4, activation='relu')(encoded)
decoded = layers.Dense(4096, activation='relu')(decoded)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(all_photos_100, all_photos_100,
                epochs=50,
                validation_data=(all_photos_100, all_photos_100))

print("----------------------------- 50 -------------------------------------")

input_img = keras.Input(shape=(4096,))
encoded = layers.Dense(4096/4, activation='relu')(input_img)
encoded = layers.Dense(50, activation='relu')(encoded)  
decoded = layers.Dense(4096/4, activation='relu')(encoded)
decoded = layers.Dense(4096, activation='relu')(decoded)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(all_photos_50, all_photos_50,
                epochs=50,
                validation_data=(all_photos_50, all_photos_50))

print("----------------------------- 25 -------------------------------------")

input_img = keras.Input(shape=(4096,))
encoded = layers.Dense(4096/4, activation='relu')(input_img)
encoded = layers.Dense(25, activation='relu')(encoded)  
decoded = layers.Dense(4096/4, activation='relu')(encoded)
decoded = layers.Dense(4096, activation='relu')(decoded)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(all_photos_25, all_photos_25,
                epochs=50,
                validation_data=(all_photos_25, all_photos_25))                