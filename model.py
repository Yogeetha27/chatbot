# Import required libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
import numpy as np
import pandas as pd
import shutil
import time
import cv2 as cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os
import seaborn as sns
sns.set_style('darkgrid')
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from IPython.core.display import display, HTML
print(123)
sdir=r'C:\Users\qwertyu\Desktop\Projects\major\another\skin-disease-datasaet\train_set'
filepaths=[]
labels=[]
classlist=os.listdir(sdir)
for klass in classlist:
    classpath=os.path.join(sdir,klass)
    if os.path.isdir(classpath):
        flist=os.listdir(classpath)
        for f in flist:
            fpath=os.path.join(classpath,f)
            filepaths.append(fpath)
            labels.append(klass)
Fseries= pd.Series(filepaths, name='filepaths')
Lseries=pd.Series(labels, name='labels')
df=pd.concat([Fseries, Lseries], axis=1)
print (df.head())
print (df['labels'].value_counts())
train_split=0.9
test_split=0.05
dummy_split=test_split/(1-train_split)
train_df, dummy_df=train_test_split(df, train_size=train_split, shuffle=True, random_state=123)
test_df, valid_df=train_test_split(dummy_df, train_size=dummy_split, shuffle=True, random_state=123)
print ('train_df length: ', len(train_df), '  test_df length: ', len(test_df), '  valid_df length: ', len(valid_df))
height=128
width=128
channels=3
batch_size=40
img_shape=(height, width, channels)
img_size=(height, width)
length=len(test_df)
test_batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80],reverse=True)[0]
test_steps=int(length/test_batch_size)
print ( 'test batch size: ' ,test_batch_size, '  test steps: ', test_steps)
def scalar(img):
    return img/127.5-1  # scale pixel between -1 and +1
gen=ImageDataGenerator(preprocessing_function=scalar)
train_gen=gen.flow_from_dataframe( train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                    color_mode='rgb', shuffle=True, batch_size=batch_size)
test_gen=gen.flow_from_dataframe( test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                    color_mode='rgb', shuffle=False, batch_size=test_batch_size)
valid_gen=gen.flow_from_dataframe( valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                    color_mode='rgb', shuffle=True, batch_size=batch_size)
classes=list(train_gen.class_indices.keys())
print (classes)
class_count=len(classes)
train_steps=int(len(train_gen.labels)/batch_size)
# Define hyperparameters
input_shape = (128, 128, 3)
batch_size = 40
epochs = 12
num_classes = 8
from efficientnet.tfkeras import EfficientNetB0
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
# Freeze the base model
base_model.trainable = False
# Define the model architecture
# Add custom classification head
inputs = tf.keras.Input(shape=input_shape)
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
# Define the model
model = tf.keras.Model(inputs, outputs)
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#  Train the model
model.fit(train_gen,epochs=epochs,validation_data=valid_gen)
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_gen)
print('Test accuracy:', (test_acc)*100)
# Import required libraries
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load the test set
test_images, test_labels = test_gen.next()

# Predict the classes of the test set
y_pred = np.argmax(model.predict(test_images), axis=-1)
print(y_pred)

# Get the true classes of the test set
y_true = np.argmax(test_labels, axis=-1)

# Print the confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_true, y_pred))

# Print the classification report
target_names = ['BA- cellulitis', 'BA-impetigo', 'FU-athlete-foot', 'FU-nail-fungus', 'FU-ringworm', 'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles']
print('\nClassification Report:')
print(classification_report(y_true, y_pred, target_names=target_names))
report = classification_report(y_true, y_pred, output_dict=True)
df = pd.DataFrame(report).transpose()
df
from PIL import Image
import numpy as np

# Function to preprocess an image for prediction
def preprocess_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict skin disease
def predict_skin_disease(image_path, model, classes):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = classes[predicted_class_index]
    return predicted_class

# # Example usage
# image_path_to_predict = r'C:\Users\qwertyu\Desktop\Projects\major\another\skin-disease-datasaet\test_set\VI-shingles\139_VI-shingles (10).jpg'
# predicted_class = predict_skin_disease(
#     image_path_to_predict, model, classes)
# print('Predicted Skin Disease:', predicted_class)
