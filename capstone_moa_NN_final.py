
import pandas as pd
import matplotlib.pyplot as plt
import random
from matplotlib import image
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cv2
from sklearn.model_selection import train_test_split
from keras import layers, models, optimizers
import tensorflow_addons as tfa
# be sure to check tfa compatability chart: 
#    https://github.com/tensorflow/addons#python-op-compatibility-matrix

# this was part of my capstone project for my MSDS degree
# this uses the Kaggle competition data from MoA prediction
# https://www.kaggle.com/c/lish-moa/data


pretrain = pd.read_csv("train_features.csv")
pretrain.shape # (23814, 876)
pretrain.columns # 876 columns


# remove ID column
sig_id = pretrain.sig_id
pretrain = pretrain.drop(columns = 'sig_id')
# get dummies on cp_type, cp_time, cp_dose
s1 = pretrain[pretrain.columns[0]]
s2 = pretrain[pretrain.columns[1]]
s3 = pretrain[pretrain.columns[2]]
s1 = pd.get_dummies(s1)
s2 = pd.get_dummies(s2)
s3 = pd.get_dummies(s3)
pretrain = pretrain.drop(columns = ['cp_type', 'cp_time', 'cp_dose'])
pretrain = pd.concat([pretrain, s1, s2, s3], axis = 1)
# standardscaler
scaler = StandardScaler()
pt_scaled = scaler.fit_transform(pretrain)
pt_scaled_df = pd.DataFrame(pt_scaled, index = pretrain.index, columns = pretrain.columns)
# PCA selection
pca = PCA(n_components = 0.90)
pt_reduced = pca.fit_transform(pt_scaled_df)
pt_reduced_df = pd.DataFrame(pt_reduced)
# make list of results
targets = pd.read_csv("train_targets_scored.csv")
targets = targets.drop(columns = 'sig_id')
targets = pd.DataFrame(targets)

evr = pca.explained_variance_ratio_
len(evr)
evr_df = pd.DataFrame(evr)
evr_df.to_csv('evr.csv', index = False)

# figure out best NN plan
    # 3x three layers: norm1d, dropout, dense, then ReLU
    # as per https://www.kaggle.com/liuhdme/moa-competition
# implement k-folds testing

# hyperparameters
epochs = 100
n_features = 388 # how many features your dataset has, pt_reduced has 388
hidden_size =512
n_targets = 872
dropratio = 0.2
seed = 123

x = pt_scaled
y = targets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)


# making a model
#model = Model(n_features, n_targets)
model = tf.keras.models.Sequential()

#model.add(tf.keras.layers.Input(x_train.shape))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(rate = dropratio))
#model.add(tf.keras.layers.Flatten())
model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(879, activation = 'relu')))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(rate = dropratio))
#model.add(tf.keras.layers.Flatten())
model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(543, activation = 'relu')))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(rate = dropratio))
#model.add(tf.keras.layers.Flatten())
model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation = 'relu')))
model.compile(loss ='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # change learning rate here
          


# class Model(nn.Module):
#     def __init__(self, n_features, n_targets, hidden_size=512, dropratio=0.2):
#         super(Model, self).__init__()
#         self.batch_norm1 = nn.BatchNorm1d(n_features)
#         self.dropout1 = nn.Dropout(dropratio)
#         self.dense1 = nn.utils.weight_norm(nn.Linear(n_features, hidden_size))
        
#         self.batch_norm2 = nn.BatchNorm1d(hidden_size)
#         self.dropout2 = nn.Dropout(dropratio)
#         self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        
#         self.batch_norm3 = nn.BatchNorm1d(hidden_size)
#         self.dropout3 = nn.Dropout(dropratio)
#         self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, n_targets))
        
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         x = self.batch_norm1(x)
#         x = self.dropout1(x)
#         x = self.relu(self.dense1(x))
        
#         x = self.batch_norm2(x)
#         x = self.dropout2(x)
#         x = self.relu(self.dense2(x))
        
#         x = self.batch_norm3(x)
#         x = self.dropout3(x)
#         x = self.dense3(x)
        
#         return x
    
# guide: https://towardsdatascience.com/writing-tensorflow-2-custom-loops-438b1ab6eb6c
# def train(x, y):
#     with tf.GradientTape() as tape:
#         y_hat = model(x_train)
#         loss_value = loss(y, y_hat)
#     grads = tape.gradient(loss_value, model.trainable_weights)
#     optimizer.apply_gradients(zip(grads, model.trainable_weights))

# def validate(x, y):
#     y_hat = model(x_train)
#     loss_value = loss(y_train, y_hat)
#     return loss_value



# MAIN
##############################################

x_train = np.array(x_train)
y_train = np.array(y_train)
x = np.array(x)
y = np.array(y)

# model.fit(x_train, y_train, epochs = 100) # only took about 20min
history = model.fit(x, y, epochs = 100) # 99.72 pct accuracy

model.save('moa_model_100epochs_11-12-20.h5')

# load model, replace XXX with model number (if necessary)
# model = keras.models.load_model('modelXXX.h5')

# val_loss = []

# for epoch in range(epochs):
#     train(x_train, y_train)
#     val_loss.append(validate(x_train, y_train))
#     print('Epoch :{0}, Loss: {1}'.format(epoch, val_loss[-1]))
#     if epochs % 3 == 0:
#         # save every 3 epochs
#         model.save('model{}.h5'.format(epoch+1))
#         print("Save {} working".format(epoch+1))

# make predictions
test_feat = pd.read_csv('test_features.csv')
test_feat_id = test_feat.sig_id
test_feat = test_feat.drop(columns = 'sig_id')
t1 = test_feat.cp_type
t2 = test_feat.cp_time
t3 = test_feat.cp_dose
t1 = pd.get_dummies(t1)
t2 = pd.get_dummies(t2)
t3 = pd.get_dummies(t3)
test_feat = test_feat.drop(columns = ['cp_type', 'cp_time', 'cp_dose'])
test_feat = pd.concat([test_feat, t1, t2, t3], axis = 1)


# test_feat = np.asarray(test_feat).astype('float32')

kag_test_pred = model.predict(test_feat)
kag_test_pred_df = pd.DataFrame(kag_test_pred)
kag_final = pd.concat([test_feat_id, kag_test_pred_df], axis = 1)
kag_col_list = ['sig_id']
kag_col_list.extend(targets.columns)
kag_final.columns = kag_col_list
kag_final.to_csv('kag_bear_pred.csv', index = False)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()