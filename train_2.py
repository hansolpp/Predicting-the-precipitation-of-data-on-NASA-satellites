import glob
import pandas as pd
import numpy as np
from keras.optimizers import adam
#from tensorflow_core.python.keras import Input
#from tensorflow_core.python.keras.models import load_model
from tensorflow.python.keras import Input
from tensorflow.python.keras.saving.save import load_model
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import Model

from Models import resnet_model, unet_model, train2_unet_model, train2_unet2_model
from trainfile_generator import trainGenerator, testGenerator, train2_Generator
from util_functions import mae, fscore, maeOverFscore, fscore_keras, maeOverFscore_keras, score, mae_over_fscore
from keras import backend as K
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import KFold
from keras_radam import RAdam
from keras_radam.training import RAdamOptimizer
import tensorflow_addons as tfa
import os
from scipy import stats
os.environ['TF_KERAS'] = '1'
os.environ['KERAS_BACKEND'] = 'theano'


# 교차검증을 위한 train code
def train_model(x_data, y_data, k):
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
            #tf.keras.callbacks.TensorBoard(
            #    log_dir='./log/plugins/profile/20',
            #    histogram_freq=0,  # How often to log histogram visualizations
            #    embeddings_freq=0,  # How often to log embedding visualizations
            #    update_freq='epoch'),  # How often to write logs (default: once per epoch)

            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                patience=3,
                # new_lr = lr * factor
                factor=0.8
            ),

            tf.keras.callbacks.ModelCheckpoint(
                filepath='./models/model' + str(model_number) + '.h5',
                monitor='score',
                save_best_only=True,
                #save_weights_only=True,
                verbose=1
            )
        ]
        RADAM = tfa.optimizers.RectifiedAdam()
        ranger = tfa.optimizers.Lookahead(RADAM, sync_period=10, slow_step_size=0.5)
        #RADAM = RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-5)
        #ADAM = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08,amsgrad=True, name='Adam')

        # 저장된 가중치 불러오기
        model.load_weights(filepath='./models/model' + str(model_number) + '.h5')

        model.compile(loss='mae', optimizer=ranger, metrics=[score, fscore_keras])

        # stratified_k_fold 사용시 batch_size는 최소 128에서 256이 되어야한다.
        model.fit(x_train, y_train, epochs=1, batch_size=128, validation_data=(x_val, y_val),
                  callbacks=callbacks_list)

        model_number += 1


if __name__ == "__main__":

    my_col = 10
    k = 5
    models = []

    # 학습데이터 생성
    train = train2_Generator()
    train = np.array(train)
    x_train = train[:, :, :, :10]
    y_train = train[:, :, :, 14].reshape(-1, 40, 40, 1)

    # train set, test set 분리
    # 학습데이터를 더 많이 사용하기 위해 7 : 3 비율을 유지하지 않습니다
    # 10%
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

    del train

    # 교차검증 학습
    train_model(x_train, y_train, k)

    del x_train, y_train

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


    print("here is rain comparision")
    # 강수량 비교( pred와 real의 강수량 차의 기본 통계량)
    rain_diff =[]
    preds.append(models[-1].predict(x_test))
    for i in range(len(y_test)):
        result = abs(preds[-1][i].sum() - y_test[i].sum())
        rain_diff.append(result)
        #print(result)

    print(stats.describe(np.array(rain_diff)))


    hit_rate_count_5_0 = 0
    hit_rate_count_1_0 = 0
    hit_rate_count_0_5 = 0
    hit_rate_count_0_1 = 0

    y_true, y_pred = np.array(y_test), np.array(preds[-1])

    y_true = y_true.reshape(1, -1)[0]

    y_pred = y_pred.reshape(1, -1)[0]

    diff = abs(y_pred - y_true)

    for i in tqdm(diff):
        if i <= 5:
            hit_rate_count_5_0 += 1

        if i <= 1:
            hit_rate_count_1_0 += 1

        if i <= 0.5:
            hit_rate_count_0_5 += 1

        if i <= 0.1:
            hit_rate_count_0_1 += 1

    print(hit_rate_count_5_0/len(diff))
    print(hit_rate_count_1_0/len(diff))
    print(hit_rate_count_0_5/len(diff))
    print(hit_rate_count_0_1/len(diff))



