import numpy as np 
import keras
from loguru import logger 
import pandas 
import os
imageSize=150
import skimage
import seaborn as sns
import numpy as np
import tensorflow as tf
import os 
import pandas as pd
import zipfile
import tarfile
import shutil
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
import pandas as pd

# ['DME', 'CNV', 'NORMAL', '.DS_Store', 'DRUSEN']
from tqdm import tqdm
import cv2 
def get_batch_data(folder,input_shape,perc_init=0,perc_final=1):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['NORMAL']:
                label = 0
            elif folderName in ['CNV']:
                label = 1
            elif folderName in ['DME']:
                label = 2
            elif folderName in ['DRUSEN']:
                label = 3
            else:
                label = 4
            number_img=len(os.listdir(folder + "/"+ folderName))
            init_index=int(number_img*perc_init)
            final_index=int(number_img*perc_final)
            for image_filename in tqdm(os.listdir(folder + "/"+ folderName)[init_index:final_index]):
                img_file = cv2.imread(folder + "/"+folderName + '/' + image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, input_shape)
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)

    
    X = np.asarray(X)
    y = np.asarray(y)
    return X,to_categorical(y)


def get_all_data(folder,input_shape,step=0.2):
    X=[]
    Y=[]
    i=0
    while(i<1):
        x,y=get_batch_data(folder,input_shape,i,i+step)
        X.append(x)
        Y.append(y)
        i+=step


def schedule(epoch,lr):
    if epoch<10 :
        return lr 
    else :
        return lr * tf.math.exp(-0.1)
def callback(name_path,id):
    
    return [keras.callbacks.EarlyStopping(monitor='val_loss',patience=10),
                  keras.callbacks.ModelCheckpoint(filepath=f"/Users/mac/Desktop/search/deeplearnigComputerVision/Stage/OCT-Glaucome/code/MLpipeline/backup_training/models/{name_path}_{id}.h5",monitor='val_acc',save_best_only=True),
                  keras.callbacks.LearningRateScheduler(schedule,verbose=0),
                  ]
    
def optimizer(name,lr):
    if name=="sgd":
        return tf.keras.optimizers.SGD(
                         learning_rate=lr,momentum=0.5, nesterov=True, name="SGD"
                )
    if name=="adam":
        return tf.keras.optimizers.Adam(
            learning_rate=lr
        )
    if name=="rmsprop":
        return tf.keras.optimizers.RMSprop(
            learning_rate=lr
        )
    if name=="adagrad":
        return tf.keras.optimizers.Adagrad(
            learning_rate=lr
        )
        
def generer_metadata_file(path,history):
    history_dict = history.history 
    loss_values = history_dict['loss']
    epochs=range(1,len(loss_values)+1)
    history_dict["epoch"]=epochs

    pd.DataFrame.from_dict(history_dict).to_csv(f"/Users/mac/Desktop/search/deeplearnigComputerVision/Stage/OCT-Glaucome/code/MLpipeline/backup_training/metadata/{path}.csv")
    logger.success("meta data of the Model successfully save")

def number_parameters_file(path,models):
    def get_number_params(model):
        return model.count_params()
    pd.DataFrame.from_dict({"model":models.keys(),"parameters":map(get_number_params,models.values())}).to_csv(f"/Users/mac/Desktop/search/deeplearnigComputerVision/Stage/OCT-Glaucome/code/MLpipeline/backup_training/metadata/{path}.csv")
    logger.success("number-parameters saved")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
imageDelegate=ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    
    )
def data_generator(input_size,data_path,batch_size):
    return imageDelegate.flow_from_directory(
        data_path,
        batch_size=batch_size,
        target_size=(input_size,input_size)  
    )
   

