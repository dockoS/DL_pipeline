from email.policy import strict
import numpy as np 
from loguru import logger 
import tensorflow as tf 
import click
from tqdm import tqdm
import cv2 
import os
import skimage
import keras
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization,ReLU
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import class_weight
from keras.models import Sequential, model_from_json,Model
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,MaxPooling2D,AveragePooling2D, BatchNormalization,GlobalAveragePooling2D,Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from sklearn.utils import class_weight
import keras.applications as pretrained_model
from keras.layers import Input
import json
from batchgenerator import get_batch_data,get_all_data,callback,optimizer,generer_metadata_file,number_parameters_file,data_generator
data_path="/Users/mac/Desktop/search/deeplearnigComputerVision/Stage/OCT-Glaucome/OCT2017"
train_data_path=os.path.join(data_path,'train')
test_data_path=os.path.join(data_path,'test')
val_data_path=os.path.join(data_path,'val')

class MLpipeline():
    def __init__(self,input_shape,n_classes,activation_output,path_data,path_hyperameters):
        self.input_shape=input_shape
        self.n_classes=n_classes
        self.output_act=activation_output
        self.models={}
        self.history={}
        self.path_data=path_data
        self.path_hyperparameters=path_hyperameters
        # self.train_data_path=os.path.join(data_path,'train')
        # test_data_path=os.path.join(data_path,'test')
        # val_data_path=os.path.join(data_path,'val')
    def DenseNet(self,growth_rate=32,list_multiplicateurs=[6,12,24,16]):
        def bn_lr_cnv(x,f,kernel=1,stride=1):
            x=BatchNormalization()(x)
            x=ReLU()(x)
            x=Conv2D(f,kernel,stride,padding='same')(x)
            return x
        def dense_block(x,r):
            for _ in range(r):
                _x=bn_lr_cnv(x,4*growth_rate)
                _x=bn_lr_cnv(_x,growth_rate,3)
                x=Concatenate()([x,_x])
            return x
        def transition_layer(x):
            x=bn_lr_cnv(x,x.shape[-1]//2)
            x=AveragePooling2D(2,strides=2)(x)
            return x

        input=Input(self.input_shape)
        x=Conv2D(64,7,strides=2)(input)
        x=MaxPool2D(3,strides=2)(x)

        for multipli in list_multiplicateurs:
            d=dense_block(x,multipli)
            x=transition_layer(d)
        ## Classificateur
        x=GlobalAveragePooling2D()(d)
        output=Dense(self.n_classes,activation=self.output_act)(x)
        model=Model(inputs=input,outputs=output)

        return model
    def GenereDenseNetModels(self):
        list_growth_rate=[16,32]
        list_models={"densenet_121":[6,12,24,16],"densnet_169":[6,12,32,32],"densenet_201":[6,12,48,32],"densenet_26":[6,12,64,48]}

        for model_name in list_models.keys():
            for k in list_growth_rate:
                model=self.DenseNet(k,list_models[model_name])
                self.models[f"{model_name}_{k}"]=model
    def GenererPretrainedModel(self):
        input=Input(self.input_shape)
        models={"vgg19":pretrained_model.vgg19.VGG19(
            include_top=False,
            input_shape=self.input_shape,
            input_tensor=input
        )
        # ,
        # "vgg16":pretrained_model.vgg16.VGG16(
        #     include_top=False,
        #     input_shape=self.input_shape 
        # ),
        # "inception_v3":pretrained_model.inception_v3.InceptionV3(
        #     include_top=False,
        #     input_shape=self.input_shape 
        # ),
        # "resnet50":pretrained_model.resnet50.ResNet50(
        #     include_top=False,
        #     input_shape=self.input_shape 
        # )
             }
        for model in models.keys():
            x=models[model].output
            output=Dense(self.n_classes,activation=self.output_act)(x)
            self.models[model]=Model(inputs=input,outputs=output)
            self.models[model].trainable=False
            logger.success(self.models[model].count_params())
            
            break
    
    def train(self,deb,fin,epochs,batch_size,lr):
        logger.success(self.input_shape)
        x_val,y_val=get_batch_data(val_data_path,self.input_shape)
        x_train,y_train=get_batch_data(train_data_path,self.input_shape,deb,fin)
        with open(self.path_hyperparameters) as f:
            data = json.load(f)
        hyperparameters=data["hyperparameters"]
        for hyper in hyperparameters:
            for model_key in self.models.keys():
                logger.success(f"debut modele")
                id=hyper["id"]
                logger.error(f"{model_key}_{id}")
                self.models[model_key].compile(optimizer =optimizer(hyper["optimizer"],lr), loss = 'categorical_crossentropy',
                           metrics = ['acc',tf.keras.metrics.Recall(name="recall"),   keras.metrics.Precision(name='precision'),])
                self.history[f"{model_key}_{id}"]= self.models[model_key].fit(x_train,y_train,epochs=epochs,
                                batch_size=batch_size,validation_data=(x_val,y_val),callbacks=callback(model_key,id))
                generer_metadata_file(f"{model_key}_{id}",self.history[f"{model_key}_{id}"])

                logger.success(f"fin modele")
        number_parameters_file("number_parameters",self.models)
        
        
        
        
    def train_with_generator(self,epochs=1,batch_size=32,lr=0.001):
        logger.success(self.input_shape)
        train_generator=data_generator(self.input_shape[0],self.path_data+"/_train",batch_size)
        validation_generator=data_generator(self.input_shape[0],self.path_data+"/_val",batch_size)
        classes_weight=class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(train_generator.classes),y=train_generator.classes)
        classes_weight=dict(enumerate(classes_weight))
        with open(self.path_hyperparameters) as f:
            data = json.load(f)
        hyperparameters=data["hyperparameters"]
        for hyper in hyperparameters:
            for model_key in self.models.keys():
                logger.success(f"debut modele")
                id=hyper["id"]
                logger.error(f"{model_key}_{id}")
                self.models[model_key].compile(optimizer =optimizer(hyper["optimizer"],lr), loss = 'categorical_crossentropy',
                           metrics = ['acc',tf.keras.metrics.Recall(name="recall"),keras.metrics.Precision(name='precision')])
                self.history[f"{model_key}_{id}"]= self.models[model_key].fit(train_generator,epochs=epochs,
                                batch_size=batch_size,validation_data=validation_generator,callbacks=callback(model_key,id),class_weight=classes_weight)
                generer_metadata_file(f"{model_key}_{id}",self.history[f"{model_key}_{id}"])

                logger.success(f"fin modele")
        number_parameters_file("number_parameters",self.models)
@click.command()
@click.option('-he', '--height', help='enter the height ',type=int)
@click.option('-w', '--width', help='enter the width ',type=int)
@click.option('-c', '--nbr_channels', help='enter the number of channels',type=int,default=3)
@click.option('-o', '--output', help='enter the number of classes',default=1000,type=int)
@click.option('-a', '--activation', help='enter the height',default="softmax",type=str)
@click.option('-p', '--path_data', help='enter the path',default="/Users/mac/Desktop/search/deeplearnigComputerVision/Stage/OCT-Glaucome/OCT2017",type=str)
@click.option('-h', '--path_hyperparameters', help='enter the path of hyperparameters',default="/Users/mac/Desktop/search/deeplearnigComputerVision/Stage/OCT-Glaucome/code/MLpipeline/modelizations/train_params.json",type=str)

def predict(height,width,nbr_channels,output,activation,path_data,path_hyperparameters):
    print(height)
    print(width)
    print(nbr_channels)
    pipeline=MLpipeline((height,width,nbr_channels),output,activation,path_data,path_hyperparameters)
    dense_nets=pipeline.GenereDenseNetModels()
    #logger.success(dense_nets)
    pipeline.train_with_generator()

if __name__ == '__main__':
    predict()
    
        