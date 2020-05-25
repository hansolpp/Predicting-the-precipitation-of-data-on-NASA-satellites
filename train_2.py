import glob
import pandas as pd
import numpy as np
from keras.optimizers import adam
from tensorflow_core.python.keras import Input
from tensorflow_core.python.keras.models import load_model
from tqdm import tqdm
import tensorflow as tf
import os
import random
from sklearn.metrics import f1_score
from tensorflow.keras import Model
import warnings

from Models import resnet_model, unet_model, train2_unet_model, train2_unet2_model
from trainfile_generator import trainGenerator, testGenerator, train2_Generator
from util_functions import mae, fscore, maeOverFscore, fscore_keras, maeOverFscore_keras
from keras import backend as K
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import KFold


# 교차검증을 위한 train code
def train_model(x_data, y_data, rain_data, k):
    k_fold = KFold(n_splits=k, shuffle=True, random_state=0)
    stratified_k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

    model_number = 0
    # for train_idx, val_idx in tqdm(stratified_k_fold.split(x_data, rain_data)):
    for train_idx, val_idx in tqdm(k_fold.split(x_data)):
        x_train, y_train = x_data[train_idx], y_data[train_idx]
        x_val, y_val = x_data[val_idx], y_data[val_idx]

        input_layer = Input(x_train.shape[1:])
        output_layer = train2_unet2_model(input_layer, 32)
        model = Model(input_layer, output_layer)

        callbacks_list = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=3,
                # new_lr = lr * factor
                factor=0.8
            ),

            tf.keras.callbacks.ModelCheckpoint(
                filepath='./models/model' + str(model_number) + '.h5',
                monitor='val_score',
                save_weights_only=True
            )
        ]

        adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True,
                                        name='Adam')
        model.compile(loss='mae', optimizer=adam, metrics=[maeOverFscore_keras, fscore_keras])
        # stratified_k_fold 사용시 batch_size는 최소 128에서 256이 되야한다.
        model.fit(x_train, y_train, epochs=10, batch_size=32, shuffle=True, validation_data=(x_val, y_val),
                  callbacks=callbacks_list)

    model_number += 1

if __name__ == "__main__":

    my_col = 10
    k = 3
    models = []

    # 학습데이터 생성
    train, rain = train2_Generator()
    train = np.array(train)
    rain = np.array(rain)
    x_train = train[:, :, :, :10]
    y_train = train[:, :, :, 14].reshape(-1, 40, 40, 1)

    # 교차검증 학습
    train_model(x_train, y_train, rain, k)

    for n in range(k):
        model = load_model('models/model' + str(n) + '.h5',
                           custom_objects={'score': maeOverFscore_keras, 'fscore_keras': fscore_keras})
        models.append(model)

    # 제출용
    preds = []
    for model in models:
        preds.append(model.predict(testGenerator()))

    pred = sum(preds) / len(preds)

    submission = pd.read_csv('Submission_form.csv')
    submission.iloc[:, 1:] = pred.reshape(-1, 1600)
    submission.to_csv('Answer.csv', index=False)