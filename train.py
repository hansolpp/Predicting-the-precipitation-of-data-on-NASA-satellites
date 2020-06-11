import glob
import pandas as pd
import numpy as np
from keras.optimizers import adam
from tensorflow.python.keras import Input
from tqdm import tqdm
import tensorflow as tf
import os
import random
from sklearn.metrics import f1_score
from tensorflow.keras import Model
import warnings

from Models import resnet_model, unet_model
from trainfile_generator import trainGenerator, testGenerator
from util_functions import mae, fscore, maeOverFscore, fscore_keras, maeOverFscore_keras
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


warnings.filterwarnings("ignore")

# 재 생산성을 위해 시드 고정
np.random.seed(7)
random.seed(7)

# 사용할 데이터 col
my_col = 10

# tf.random.set_seed(42)

# 가중치의 체크 포인트 이름 저장
checkpoint_path = ".saved_weight/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 체크포인트 콜백 만들기
callbacks_list = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                patience=3,
                # new_lr = lr * factor
                factor=0.8
            ),

            tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
        ]
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

# 데이터 파이프라인 만들기
train_dataset = tf.data.Dataset.from_generator(trainGenerator, (tf.float32, tf.float32),
                                               (tf.TensorShape([40, 40, my_col]), tf.TensorShape([40, 40, 1])))

# batch크기가 512로 너무 큰거 같다는 생각이 듬 논문을 참고하면 32에서 64가 이미지 분석에는 적당하다고 봄
train_dataset = train_dataset.batch(128).prefetch(1)

input_layer = Input((40, 40, my_col))
# 처음 시작 뉴론 32개 말고 다른것으로 변경해 볼 것

#output_layer = resnet_model(input_layer)
output_layer = unet_model(input_layer, 32)
#output_layer = improved_unet_model(input_layer, 32)

model = Model(input_layer, output_layer)
# 저장된 가중치 불러오기
#model.load_weights(checkpoint_path)

# ################## compile 메서드로 모형 완성 ###########################
adam = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07,amsgrad=True, name='Adam')

model.compile(loss="mae", optimizer=adam, metrics=[maeOverFscore_keras, fscore_keras, "accuracy"])

# ################## fit 메서드로 트레이닝 #################################
#model_history = model.fit_sample()
model_history = model.fit(train_dataset, epochs=10, verbose=1, shuffle=True, callbacks=callbacks_list)

pred = model.predict(testGenerator())
submission = pd.read_csv('Submission_form.csv')
submission.iloc[:, 1:] = pred.reshape(-1, 1600)
submission.to_csv('Answer.csv', index=False)