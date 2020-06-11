import glob
import string
from os.path import isfile
import os
import random
import pickle
import matplotlib.pylab as plt

import numpy as np
import pandas as pd
from pandas import DataFrame
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")


# 범위 데이터 제한 개수
limite_data_count = 500


# 전체 데이터에서 결측치 및 0 < sum < 1 사이의 데이터를 제거합니다.
def delete_unimportant_npy():
    # from_path는 반드시 있는 경로와 데이터여야 합니다.
    from_path = './train/train'
    from_path_files = sorted(glob.glob(from_path + '/*'))
    from_path_files = from_path_files[::]
    # to_path는 해당하는 경로가 없을 경우 생성하여 사용합니다.
    to_path = './delete_missing_value_data/'

    # NASA 디렉터토리(현재 make_train_set.py가 실행되는 경로) 안에 새롭게 저장할 디렉토리 생성
    # 존재할 경우 덮어쓰고, 존재하지 않을 경우 새롭게 생성
    try:
        if not (os.path.isdir(to_path)):
            os.makedirs(os.path.join(to_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory")
            raise

    # 데이터 처리
    for npy_file in tqdm(from_path_files):

        # 한개의 npy 데이터 불러오기
        one_npy_data = np.load(npy_file)
        # 파일 데이터 이름
        real_file_name = npy_file.replace('train', '').replace('/', '')

        # 강수량 값(데이터를 판별할 기준 값)
        target = one_npy_data[:, :, -1].reshape(40, 40, 1)

        # 강수량이 0보다 작은 것 제거
        if target.sum() < 0:
            continue

        # 강수량이 0 < x < 1 인것 제거
        if (target.sum() > 0) and (target.sum() < 1):
            continue
        
        # 데이터 저장
        np.save(to_path + real_file_name, one_npy_data)


# delete_unimportant_npy()에서 정제된 데이터를 가지고 balance한 데이터를 만듭니다.
def make_balance_npy():
    # from_path는 반드시 있는 경로와 데이터여야 합니다.
    from_path = './delete_missing_value_data/'
    from_path_files = sorted(glob.glob(from_path + '/*'))
    from_path_files = from_path_files[::]
    # to_path는 해당하는 경로가 없을 경우 생성하여 사용합니다.
    to_path = './balance_data/'

    # NASA 디렉터토리(현재 make_train_set.py가 실행되는 경로) 안에 새롭게 저장할 디렉토리
    # 존재할 경우 덮어쓰고, 존재하지 않을 경우 새롭게 생성
    try:
        if not (os.path.isdir(to_path)):
            os.makedirs(os.path.join(to_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory")
            raise

    # 강수량 합 데이터 리스트
    rain_data_list = []
    # npy 이름 데이터 리스트
    data_name_list = []
    '''
    # 데이터 처리
    for npy_file in tqdm(from_path_files):

        # 한개의 npy 데이터 불러오기
        one_npy_data = np.load(npy_file)
        # 파일 데이터 이름
        real_file_name = npy_file.replace('delete_missing_value_data', '').replace('/', '').replace('.', '').replace('\\', '')

        # 강수량 값(데이터를 판별할 기준 값)
        target = one_npy_data[:, :, -1].reshape(40, 40, 1).sum()

            # 'rain_data_list' col 만들기
        rain_data_list.append(target)
        # '*.npy 이름도 저장
        data_name_list.append(real_file_name)
        #5347개의 구간이 필요함( 최대 26731.1666을 커버하기 위해)
    '''
    # 리스트 데이터 저장
    #with open('./rain_data_list.pickle', 'wb') as f:
    #    pickle.dump(rain_data_list, f, pickle.HIGHEST_PROTOCOL)
    #with open('./data_name_list.pickle', 'wb') as f:
    #    pickle.dump(data_name_list, f, pickle.HIGHEST_PROTOCOL)

    # 리스트 데이터 사용
    with open('./rain_data_list.pickle', 'rb') as c:
        rain_data_list = pickle.load(c)
    with open('./data_name_list.pickle', 'rb') as c:
        data_name_list = pickle.load(c)

    # dataframe에 담기
    rain_data_df = DataFrame({'rain_data_list': rain_data_list, 'data_name_list': data_name_list})
    range_data_df = pd.cut(rain_data_df.rain_data_list, 5347)

    # dataframe 합치기
    total_data_df = pd.merge(rain_data_df, range_data_df)
    print(total_data_df)

    # make tatble
    grouped_by_range_df = rain_data_df.rain_data_list.groupby(range_data_df)
    grouped_by_range_dict = dict(list(grouped_by_range_df))
    #print(grouped_by_range_dict[()])
    #print(grouped_by_range_df.agg(['count', 'mean', 'std', 'min', 'max']))
    #print(grouped_by_range_df)
    #grouped_by_range_df.count().to_csv("test.csv")
    rain_data_df.to_csv("original.csv")
    #plt.show()

    # make plot
    #plt.figure()
    #rain_data_df.plot(grouped_by_rage_df)


# 전체 데이터에서 결측치 및 0 < sum < 1 사이의 데이터를 제거합니다.
#delete_unimportant_npy()

# delete_unimportant_npy()에서 정제된 데이터를 가지고 balance한 데이터를 만듭니다.
make_balance_npy()