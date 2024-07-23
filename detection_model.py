#importing libraries
import matplotlib.pyplot as plt
import cv2 #for image & video processing
from glob import glob # for maintaining a list of files in the directory
import os
import random

from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.layers import Dense,Input,Dropout,Flatten,Conv2D # type: ignore
from tensorflow.keras.layers import BatchNormalization,Activation,MaxPooling2D # type: ignore
from tensorflow.keras.models import Model,Sequential # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau # type: ignore
from tensorflow.keras.utils import plot_model  # type: ignore

from IPython.display import SVG,Image
import tensorflow as tf
print('TensorFlow version: ',tf.__version__)

#exploring dataset
img_size = 48
batch_size = 64
datagen_train = ImageDataGenerator()
train_generator = datagen_train.flow_from_directory(r"C:\Users\VIAN\Desktop\nullclass\facial_contexts\train",
                                                    target_size = (img_size,img_size),
                                                    color_mode = "grayscale",
                                                    batch_size = batch_size,
                                                    class_mode = "categorical",
                                                    shuffle = True)
datagen_validation = ImageDataGenerator()
validation_generator = datagen_validation.flow_from_directory(r"C:\Users\VIAN\Desktop\nullclass\facial_contexts\test",
                                                    target_size = (img_size,img_size),
                                                    color_mode = "grayscale",
                                                    batch_size = batch_size,
                                                    class_mode = "categorical",
                                                    shuffle = True)

#defining the model
def Convolution(input_tensor,filters,kernel_size):
    x = Conv2D(filters = filters, kernel_size = kernel_size, padding = 'same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.25)(x)
    return x
def Dense_f(input_tensor,nodes):
    x = Dense(nodes)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    return x
def model_fer(input_shape):
    inputs = Input(input_shape)
    conv1 = Convolution(inputs,32,(3,3))
    conv2 = Convolution(inputs,64,(5,5))
    conv3 = Convolution(inputs,128,(3,3))
    flatten = Flatten()(conv3)
    dense_1 = Dense_f(flatten,256)
    output = Dense(7,activation='softmax')(dense_1)
    model = Model(inputs=[inputs],outputs=[output])
    model.compile(loss=['categorical_crossentropy'],optimizer='adam',metrics=['accuracy'])
    return model
model = model_fer((48,48,1))
model.summary()

#initialising the model
epochs = 15
steps_per_epoch = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size
checkpoint = ModelCheckpoint("model.weights.h5",monitor="val_acccuracy",save_weights_only=True,mode='max',verbose=1)
callbacks = [checkpoint]

#training the model
history = model.fit(
    x = train_generator,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = validation_steps,
    callbacks = callbacks)
#model evaluation
model.evaluate(validation_generator)

#saving the model
model_json=model.to_json()
with open("model_a.json",'w') as json_file:
          json_file.write(model_json)