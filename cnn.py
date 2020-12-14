import numpy as np 
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
train_dir="/home/ankit/Desktop/code_warriors_game_of_data_ai_challenge-dataset/train/"

def menu_sorting():
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
        bakes_and_brews_mouth_water = data_gen.flow_from_directory(train_dir,color_mode='grayscale',
        classes=['Bread','Dairy Product','Dessert','Egg','Fried Food','Meat','Noodles-Pasta','Rice','Seafood','Soup','Vegetable-Fruit'],class_mode='categorical',target_size=(180,180),batch_size=32)
        return bakes_and_brews_mouth_water


class bakes_and_brews_bot:
       def train_me(self,images):
            seeker = Sequential()
            seeker.add(Conv2D(32, (3, 3), activation='relu', input_shape=(180,180,1)))
            seeker.add(BatchNormalization())
            seeker.add(MaxPooling2D(pool_size=(2, 2)))
            seeker.add(Dropout(0.25))
            seeker.add(Conv2D(64, (3, 3), activation='relu'))
            seeker.add(BatchNormalization())
            seeker.add(MaxPooling2D(pool_size=(2, 2)))
            seeker.add(Dropout(0.25))
            seeker.add(Conv2D(128, (3, 3), activation='relu'))
            seeker.add(BatchNormalization())
            seeker.add(MaxPooling2D(pool_size=(2, 2)))
            seeker.add(Dropout(0.25))
            seeker.add(Flatten())
            seeker.add(Dense(512, activation='relu'))
            seeker.add(BatchNormalization())
            seeker.add(Dropout(0.5))
            seeker.add(Dense(11, activation='softmax'))
            seeker.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
            seeker.fit(tem, epochs=15)



tem = menu_sorting()
bakery_object = bakes_and_brews_bot()
bakery_object.train_me(tem)
