import glob
import pandas as pd
import numpy as np
from keras.optimizers import adam
from tqdm import tqdm
import tensorflow as tf
import os
import random
from sklearn.metrics import f1_score
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Conv2DTranspose, MaxPooling2D, BatchNormalization, \
    Activation, concatenate, Input, GlobalAveragePooling2D
from tensorflow.keras import Model
import warnings

warnings.filterwarnings("ignore")

# 재 생산성을 위해 시드 고정
# 42 시도해볼 것
np.random.seed(7)
random.seed(7)
tf.set_random_seed(7)

# 사용할 데이터 col
# 10 이랑 14 중에 최적이 어느것인지 확인 중 -> 10이 나은거 같음
# 별 차이 없으면 10 씀
my_col = 14

# tf.random.set_seed(42)

# 가중치의 체크 포인트 이름 저장
checkpoint_path = ".saved_weight/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 체크포인트 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

# 강우량 리스트
rain_list = []


#################################################################
def trainGenerator():
    train_path = './train/train'
    train_files = sorted(glob.glob(train_path + '/*'))
    # 10000개의 데이터만 우선 학습할거야
    # 50850이 최적
    # 늘려서 실험해볼것 38350
    train_files = train_files[50850::]

    for npy_file in train_files:

        one_npy_data = np.load(npy_file)

        # 평균 뺀게 성능이 좋음
        for npy_colum in range(my_col):
            x = one_npy_data[:, :, npy_colum]
            y = (x - x.mean())
            #y = (x - x.mean()) / (x.std() + 1e-10)
            one_npy_data[:, :, npy_colum] = y

        # 강수량 값
        target = one_npy_data[:, :, -1].reshape(40, 40, 1)

        cutoff_labels = np.where(target < 0, -1, target)
        feature = one_npy_data[:, :, :my_col]

        # 강수량이 0보다 작은 것 제거
        if cutoff_labels.sum() < 0:
            continue

        # 강수량이 0 < x < 1 인것 제거
        if (cutoff_labels.sum() > 0) and (cutoff_labels.sum() < 1):
            continue

        rain_list.append(cutoff_labels.sum())

        # 강수량이 1000이상인것 제거
        # 다음할것은 dropout 수정
        # 500이 최적
        if cutoff_labels.sum() > 500:
            continue

        # 강수량 데이터 분포를 알기위해
        #rain_list.append(cutoff_labels.sum())

        yield (feature, cutoff_labels)


# 데이터 파이프라인 만들기(데이터셋 API이용) - 메모리나 파일에 있는 데이터를 데이터 소스로 만듬
# from_generator()는 생성자에서 데이터셋을 만든다.
train_dataset = tf.data.Dataset.from_generator(trainGenerator, (tf.float32, tf.float32),
                                               (tf.TensorShape([40, 40, my_col]), tf.TensorShape([40, 40, 1])))

# batch크기가 512로 너무 큰거 같다는 생각이 듬 논문을 참고하면 32에서 64가 이미지 분석에는 적당하다고 봄
# 현재는 32가 최적 - 16, 64 로 실험하였지만 나은 결과를 보지 못함
train_dataset = train_dataset.batch(32).prefetch(1)
######################################################################
test_path = './test/test'
test_files = sorted(glob.glob(test_path + '/*'))

X_test = []

for file in tqdm(test_files, desc='test'):
    data = np.load(file)

    # 평균 뺀게 성능이 좋음
    for npy_colum in range(my_col):
        x = data[:, :, npy_colum]
        y = (x - x.mean())
        #y = (x - x.mean()) / (x.std() + 1e-10)
        data[:, :, npy_colum] = y

    X_test.append(data[:, :, :my_col])

X_test = np.array(X_test)


#######################################################################

# train data와 test data의 점수 분포가 있는것은 과 학습이 되어있는것
# dropout을 늘려볼까?

def build_model(input_layer, start_neurons):
    # 40 x 40 -> 20 x 20
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv1)
    pool1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(pool1)
    pool1 = Dropout(0.25)(pool1)

    # 20 x 20 -> 10 x 10
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv2)
    pool2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(pool2)
    pool2 = Dropout(0.25)(pool2)

    # 10 x 10
    convm = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(pool2)
    convm = BatchNormalization()(convm)

    # 10 x 10 -> 20 x 20
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same", kernel_initializer='he_normal')(convm)
    deconv2 = BatchNormalization()(deconv2)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = BatchNormalization()(uconv2) #
    uconv2 = Dropout(0.25)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(uconv2)
    uconv2 = BatchNormalization()(uconv2)

    # 20 x 20 -> 40 x 40
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same", kernel_initializer='he_normal')(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = BatchNormalization()(uconv1) #
    uconv1 = Dropout(0.25)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    #uconv1 = Dropout(0.25)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", activation='relu', kernel_initializer='he_normal')(uconv1)

    return output_layer


input_layer = Input((40, 40, my_col))
# 처음 시작 뉴론 32개 말고 다른것으로 변경해 볼 것
# 현재 32개가 최적
output_layer = build_model(input_layer, 32)

model = Model(input_layer, output_layer)
# 저장된 가중치 불러오기
#model.load_weights(checkpoint_path)


#######################################################################


# 평균 절대 오차
def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    y_true = y_true.reshape(1, -1)[0]

    y_pred = y_pred.reshape(1, -1)[0]

    over_threshold = y_true >= 0.1

    return np.mean(np.abs(y_true[over_threshold] - y_pred[over_threshold]))


def fscore(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    y_true = y_true.reshape(1, -1)[0]

    y_pred = y_pred.reshape(1, -1)[0]

    remove_NAs = y_true >= 0

    y_true = np.where(y_true[remove_NAs] >= 0.1, 1, 0)

    y_pred = np.where(y_pred[remove_NAs] >= 0.1, 1, 0)

    return (f1_score(y_true, y_pred))


def maeOverFscore(y_true, y_pred):
    return mae(y_true, y_pred) / (fscore(y_true, y_pred) + 1e-07)


def fscore_keras(y_true, y_pred):
    score = tf.py_function(func=fscore, inp=[y_true, y_pred], Tout=tf.float32, name='fscore_keras')
    return score


def maeOverFscore_keras(y_true, y_pred):
    score = tf.py_function(func=maeOverFscore, inp=[y_true, y_pred], Tout=tf.float32, name='custom_mse')
    return score


# ################## compile 메서드로 모형 완성 ###########################
# loss -> "mae" "categorical_crossentropy" - 성능 지표가 mae이기 대문에 수정 불가
# optimizer -> 다른걸로 바꾸어서 해보기 -> Nadam이 더 정확하게 나타남
# metrics -> 인수로 트레이닝 단계에서 기록할 성능 기준 설정 - 우리가 필요한건 mae/Fscore
my_adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                   amsgrad=False, name='Adam')
tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam')
model.compile(loss="mae", optimizer=my_adam, metrics=[maeOverFscore_keras, fscore_keras, "accuracy"])

# ################## fit 메서드로 트레이닝 #################################
# epochs는 실험적으로 적용해 보기 반복을 얼마나 할 것인지
# 500 이 최고 성능
# 오버 피팅이 일어날 수 있으므로 학습 횟수의 최적화를 찾아야함
model_history = model.fit(train_dataset, epochs=10, verbose=1, callbacks=[cp_callback])


pred = model.predict(X_test)
submission = pd.read_csv('Submission_form.csv')
submission.iloc[:, 1:] = pred.reshape(-1, 1600)
submission.to_csv('Answer.csv', index=False)

s = pd.Series(rain_list)
print(s.describe())