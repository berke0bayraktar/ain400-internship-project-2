import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model,save_model
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.losses import CategoricalCrossentropy

import cv2

import random as rnd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix


from model import make_ACFF_model


# ARGUMENTS

K.clear_session()

img_height=224
img_width=224
num_classes=3
num_workers=2
batch_size=32
epochs=500
lr_init=1e-3

model_name = "emergency_net"
fus = 'max'
act = 'l'

# DATASETS

DATASET = "2_FRAME_AUGMENTED"


train_data_dir = f"D:/data/covidx-us/{DATASET}/train_data"
test_data_dir = f"D:/data/covidx-us/{DATASET}/test_data"



train_datagen = ImageDataGenerator(rescale=1./255.,
                                    rotation_range=5,
                                    shear_range=0.02,
                                    zoom_range=0.02,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    horizontal_flip=True,
                                    fill_mode="nearest",
                                    validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset="training")

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset="validation")

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    shuffle=False)

# MODEL

inp, cls = make_ACFF_model(img_height,img_width,C=num_classes,fus=fus,act=act)
model = Model(inputs=[inp], outputs=[cls], name=model_name)
model.summary()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr_init, 200, 0.6, staircase=False, name=None)

model.compile(
    optimizer= tf.keras.optimizers.Adam(learning_rate=lr_schedule), 
    loss=CategoricalCrossentropy(), 
    metrics=tf.keras.metrics.CategoricalAccuracy())

# TRAINING

checkpoint = ModelCheckpoint('../results/model.h5', 
                             monitor='val_categorical_accuracy', 
                             save_best_only=True, 
                             mode='max', verbose=1, save_weights_only=False)

train_steps = train_generator.samples // batch_size
val_steps = validation_generator.samples // batch_size

assert(val_steps > 0)

history=model.fit(x=train_generator,
                  steps_per_epoch=train_steps,
                  verbose=1,
                  validation_data=validation_generator,
                  validation_steps = val_steps,
                  workers=num_workers,
                  epochs=epochs,
                  callbacks=[checkpoint])

# PLOTS

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim((0,1.1))
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('../results/acc.png')
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('../results/loss.png')


# TESTING

model = load_model('../results/model.h5')

Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')

target_names = ['covid','pneumonia','regular']
str_report = classification_report(test_generator.classes, y_pred, target_names=target_names)
dict_report = classification_report(test_generator.classes, y_pred, target_names=target_names, output_dict=True)
print(str_report)


# SAVE RUN DATA

import io

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


run_data = {
    "dataset": DATASET,
    "img_size": img_height,
    "batch_size": batch_size,
    "epochs": epochs,
    "optimizer": model.optimizer.get_config()["name"],
    "learning_rate": lr_init,
    "model": model.name,
    "activation": act,
    "fusion": fus,
    "classification_report": dict_report,
    "model_sum": get_model_summary(model),
    }

import json

with open('../results/run.json', 'w') as fp:
    json.dump(run_data, fp)
    
import shutil
import os
from datetime import datetime

file_name = str(datetime.now()).replace(" ","-").replace(":","-").replace(".","-")
file_name = DATASET + "-" + file_name
shutil.copytree("../results/", f"../runs/{file_name}")









