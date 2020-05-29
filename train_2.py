import glob
import pandas as pd
import numpy as np
from keras.optimizers import adam
from tensorflow_core.python.keras import Input
from tensorflow_core.python.keras.models import load_model
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import Model
from sklearn.metrics import f1_score

from Models import resnet_model, unet_model, train2_unet_model, train2_unet2_model
from trainfile_generator import trainGenerator, testGenerator, train2_Generator
from util_functions import mae, fscore, maeOverFscore, fscore_keras, maeOverFscore_keras, score, mae_over_fscore
from keras import backend as K
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import KFold
from keras_radam import RAdam
from keras_radam.training import RAdamOptimizer
import os
import pickle
os.environ['TF_KERAS'] = '1'


# 교차검증을 위한 train code
def train_model(x_data, y_data, rain_data, k):
    k_fold = KFold(n_splits=k, shuffle=True, random_state=0)
    #stratified_k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

    model_number = 0
    #for train_idx, val_idx in tqdm(stratified_k_fold.split(x_data, rain_data)):
    for train_idx, val_idx in tqdm(k_fold.split(x_data)):
        x_train, y_train = x_data[train_idx], y_data[train_idx]
        x_val, y_val = x_data[val_idx], y_data[val_idx]

        input_layer = Input(x_train.shape[1:])
        output_layer = train2_unet2_model(input_layer, 32)
        model = Model(input_layer, output_layer)

        callbacks_list = [
            # 스케쥴러?
            #tf.keras.callbacks.ReduceLROnPlateau(
            #    monitor='val_loss',
            #    patience=3,
            #    factor=0.8
            #),

            tf.keras.callbacks.ModelCheckpoint(
                filepath='./models/model' + str(model_number) + '.h5',
                monitor='score',
                save_best_only=True,
                #save_weights_only=True,
                verbose=1
            )
        ]

        model.compile(loss='mae', optimizer=RAdamOptimizer(learning_rate=1e-3)
                        , metrics=[score, maeOverFscore_keras, fscore_keras])
        # stratified_k_fold 사용시 batch_size는 최소 128에서 256이 되어야한다.
        model.fit(x_train, y_train, epochs=50, batch_size=128, shuffle=True, validation_data=(x_val, y_val),
                  callbacks=callbacks_list)

        model_number += 1


if __name__ == "__main__":

    my_col = 10
    k = 5
    models = []

    # 학습데이터 생성
    train, rain = train2_Generator()
    train = np.array(train)
    rain = np.array(rain)
    x_train = train[:, :, :, :10]
    y_train = train[:, :, :, 14].reshape(-1, 40, 40, 1)

    # train set, test set 분리
    # 학습데이터를 더 많이 사용하기 위해 7 : 3 비율을 유지하지 않습니다
    # 10%
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.02, random_state=0)

    # 교차검증 학습
    train_model(x_train, y_train, rain, k)

    # 학습된 모델 저장
    for n in range(k):
        model = load_model('models/model' + str(n) + '.h5', custom_objects={'score':score,'fscore_keras':fscore_keras})
        models.append(model)

    # 모델 성능 일반화
    preds = []
    for model in models:
        preds.append(model.predict(x_test))
        print(mae_over_fscore(y_test, preds[-1]))

    pred = sum(preds) / len(preds)
    print(mae_over_fscore(y_test, pred))
