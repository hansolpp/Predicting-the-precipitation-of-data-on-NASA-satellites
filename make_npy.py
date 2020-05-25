import glob
import string
from os.path import isfile
import random
import seaborn

import numpy as np
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

warnings.filterwarnings("ignore")

# 사용할 데이터 col
my_col = 10

# 강우량 리스트
rain_list = []

# 구간 당 최대 데이터 갯수
common_count = 800


#################################################################
def make_npy(is_check):
    train_path = './train/train'
    train_files = sorted(glob.glob(train_path + '/*'))
    train_files = train_files[::]

    common_save_path = './my_train/'

    # count_list 초기화
    count_list = [0 for _ in range(165)]
    # save_list 초기화
    save_list = [' ' for _ in range(165)]

    for npy_file in tqdm(train_files):

        # 저장 해야하는 값
        # 다 완성후에는 target 저장을 one_npy_data로 바꾸어야 함, 반드시
        one_npy_data = np.load(npy_file)
        real_file_name = npy_file.replace('train', '').replace('/', '')  # .replace('.', '').replace('\\', '')

        # 강수량 값
        target = one_npy_data[:, :, -1].reshape(40, 40, 1)

        # 강수량이 0보다 작은 것 제거
        if target.sum() < 0:
            continue

        # 강수량이 0 < x < 1 인것 제거
        if (target.sum() > 0) and (target.sum() < 1):
            continue

        # sum = 0
        if target.sum() == 0 and count_list[0] < common_count:
            if is_check:
                save_path = 'sum_zero/'
                if count_list[0] == 0:
                    save_list[0] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[0] += 1

        # 1 <= sum < 5
        elif 1 <= target.sum() < 5 and count_list[1] < common_count:
            if is_check:
                save_path = 'sum_zero_to_five/'
                if count_list[1] == 0:
                    save_list[1] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[1] += 1

        # 5 <= sum < 10
        elif 5 <= target.sum() < 10 and count_list[2] < common_count:
            if is_check:
                save_path = 'sum_five_to_ten/'
                if count_list[2] == 0:
                    save_list[2] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[2] += 1

        # 10 <= sum < 15
        elif 10 <= target.sum() < 15 and count_list[3] < common_count:
            if is_check:
                save_path = 'sum_ten_to_fifteen/'
                if count_list[3] == 0:
                    save_list[3] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[3] += 1

        # 15 <= sum < 20
        elif 15 <= target.sum() < 20 and count_list[4] < common_count:
            if is_check:
                save_path = 'sum_fifteen_to_twenty/'
                if count_list[4] == 0:
                    save_list[4] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[4] += 1

        # 20 <= sum < 25
        elif 20 <= target.sum() < 25 and count_list[5] < common_count:
            if is_check:
                save_path = 'sum_twenty_to_twenty_five/'
                if count_list[5] == 0:
                    save_list[5] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[5] += 1

        # 25 <= sum < 30
        elif 25 <= target.sum() < 30 and count_list[6] < common_count:
            if is_check:
                save_path = 'sum_twenty_five_to_thirty/'
                if count_list[6] == 0:
                    save_list[6] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[6] += 1

        # 30 <= sum < 35
        elif 30 <= target.sum() < 35 and count_list[7] < common_count:
            if is_check:
                save_path = 'sum_thirty_to_thirty_five/'
                if count_list[7] == 0:
                    save_list[7] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[7] += 1

        # 35 <= sum < 40
        elif 35 <= target.sum() < 40 and count_list[8] < common_count:
            if is_check:
                save_path = 'sum_thirty_five_to_forty/'
                if count_list[8] == 0:
                    save_list[8] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[8] += 1

        # 40 <= sum < 45
        elif 40 <= target.sum() < 45 and count_list[9] < common_count:
            if is_check:
                save_path = 'sum_forty_to_forty_five/'
                if count_list[9] == 0:
                    save_list[9] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[9] += 1

        # 45 <= sum < 50
        elif 45 <= target.sum() < 50 and count_list[10] < common_count:
            if is_check:
                save_path = 'sum_forty_five_to_fifty/'
                if count_list[10] == 0:
                    save_list[10] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[10] += 1

        # 50 <= sum < 55
        elif 50 <= target.sum() < 55 and count_list[11] < common_count:
            if is_check:
                save_path = 'sum_fifty_to_fifty_five/'
                if count_list[11] == 0:
                    save_list[11] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[11] += 1

        # 55 <= sum < 60
        elif 55 <= target.sum() < 60 and count_list[12] < common_count:
            if is_check:
                save_path = 'sum_fifty_five_to_sixty/'
                if count_list[12] == 0:
                    save_list[12] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[12] += 1

        # 60 <= sum < 65
        elif 60 <= target.sum() < 65 and count_list[13] < common_count:
            if is_check:
                save_path = 'sum_sixty_to_sixty_five/'
                if count_list[13] == 0:
                    save_list[13] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[13] += 1

        # 65 <= sum < 70
        elif 65 <= target.sum() < 70 and count_list[14] < common_count:
            if is_check:
                save_path = 'sum_sixty_five_to_seventy/'
                if count_list[14] == 0:
                    save_list[14] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[14] += 1

        # 70 <= sum < 75
        elif 70 <= target.sum() < 75 and count_list[15] < common_count:
            if is_check:
                save_path = 'sum_seventy_to_seventy_five/'
                if count_list[15] == 0:
                    save_list[15] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[15] += 1

        # 75 <= sum < 80
        elif 75 <= target.sum() < 80 and count_list[16] < common_count:
            if is_check:
                save_path = 'sum_seventy_five_to_eighty/'
                if count_list[16] == 0:
                    save_list[16] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[16] += 1

        # 80 <= sum < 85
        elif 80 <= target.sum() < 85 and count_list[17] < common_count:
            if is_check:
                save_path = 'sum_eighty_to_eighty_five/'
                if count_list[17] == 0:
                    save_list[17] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[17] += 1

        # 85 <= sum < 90
        elif 85 <= target.sum() < 90 and count_list[18] < common_count:
            if is_check:
                save_path = 'sum_eighty_five_to_ninety/'
                if count_list[18] == 0:
                    save_list[18] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[18] += 1

        # 90 <= sum < 95
        elif 90 <= target.sum() < 95 and count_list[19] < common_count:
            if is_check:
                save_path = 'sum_ninety_to_ninety_five/'
                if count_list[19] == 0:
                    save_list[19] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[19] += 1

        # 95 <= sum < 100
        elif 95 <= target.sum() < 100 and count_list[20] < common_count:
            if is_check:
                save_path = 'sum_ninety_five_to_hundred/'
                if count_list[20] == 0:
                    save_list[20] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[20] += 1

        # 100 <= sum < 105
        elif 100 <= target.sum() < 105 and count_list[21] < common_count:
            if is_check:
                save_path = 'sum_hundred_to_hundred_five/'
                if count_list[21] == 0:
                    save_list[21] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[21] += 1

        # 105 <= sum < 110
        elif 105 <= target.sum() < 110 and count_list[22] < common_count:
            if is_check:
                save_path = 'sum_hundred_five_to_hundred_ten/'
                if count_list[22] == 0:
                    save_list[22] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[22] += 1

        # 110 <= sum < 115
        elif 110 <= target.sum() < 115 and count_list[23] < common_count:
            if is_check:
                save_path = 'sum_hundred_ten_to_hundred_fifteen/'
                if count_list[23] == 0:
                    save_list[23] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[23] += 1

        # 115 <= sum < 120
        elif 115 <= target.sum() < 120 and count_list[24] < common_count:
            if is_check:
                save_path = 'sum_hundred_fifteen_to_hundred_twenty/'
                if count_list[24] == 0:
                    save_list[24] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[24] += 1

        # 120 <= sum < 125
        elif 120 <= target.sum() < 125 and count_list[25] < common_count:
            if is_check:
                save_path = 'sum_hundred_twenty_to_hundred_twenty_five/'
                if count_list[25] == 0:
                    save_list[25] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[25] += 1

        # 125 <= sum < 130
        elif 125 <= target.sum() < 130 and count_list[26] < common_count:
            if is_check:
                save_path = 'sum_hundred_twenty_five_to_hundred_thirty/'
                if count_list[26] == 0:
                    save_list[26] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[26] += 1

        # 130 <= sum < 135
        elif 130 <= target.sum() < 135 and count_list[27] < common_count:
            if is_check:
                save_path = 'sum_hundred_thirty_to_hundred_thirty_five/'
                if count_list[27] == 0:
                    save_list[27] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[27] += 1

        # 135 <= sum < 140
        elif 135 <= target.sum() < 140 and count_list[28] < common_count:
            if is_check:
                save_path = 'sum_hundred_thirty_five_to_hundred_forty/'
                if count_list[28] == 0:
                    save_list[28] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[28] += 1

        # 140 <= sum < 145
        elif 140 <= target.sum() < 145 and count_list[29] < common_count:
            if is_check:
                save_path = 'sum_hundred_forty_to_hundred_forty_five/'
                if count_list[29] == 0:
                    save_list[29] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[29] += 1

        # 145 <= sum < 150
        elif 145 <= target.sum() < 150 and count_list[30] < common_count:
            if is_check:
                save_path = 'sum_hundred_forty_five_to_hundred_fifty/'
                if count_list[30] == 0:
                    save_list[30] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[30] += 1

        # 150 <= sum < 155
        elif 150 <= target.sum() < 155 and count_list[31] < common_count:
            if is_check:
                save_path = 'sum_hundred_fifty_to_hundred_fifty_five/'
                if count_list[31] == 0:
                    save_list[31] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[31] += 1

        # 155 <= sum < 160
        elif 155 <= target.sum() < 160 and count_list[32] < common_count:
            if is_check:
                save_path = 'sum_hundred_fifty_five_to_hundred_sixty/'
                if count_list[32] == 0:
                    save_list[32] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[32] += 1

        # 160 <= sum < 165
        elif 160 <= target.sum() < 165 and count_list[33] < common_count:
            if is_check:
                save_path = 'sum_hundred_sixty_to_hundred_sixty_five/'
                if count_list[33] == 0:
                    save_list[33] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[33] += 1

        # 165 <= sum < 170
        elif 165 <= target.sum() < 170 and count_list[34] < common_count:
            if is_check:
                save_path = 'sum_hundred_sixty_five_to_hundred_seventy/'
                if count_list[34] == 0:
                    save_list[34] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[34] += 1

        # 170 <= sum < 175
        elif 170 <= target.sum() < 175 and count_list[35] < common_count:
            if is_check:
                save_path = 'sum_hundred_seventy_to_hundred_seventy_five/'
                if count_list[35] == 0:
                    save_list[35] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[35] += 1

        # 175 <= sum < 180
        elif 175 <= target.sum() < 180 and count_list[36] < common_count:
            if is_check:
                save_path = 'sum_hundred_seventy_five_to_hundred_eighty/'
                if count_list[36] == 0:
                    save_list[36] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[36] += 1

        # 180 <= sum < 185
        elif 180 <= target.sum() < 185 and count_list[37] < common_count:
            if is_check:
                save_path = 'sum_hundred_eighty_to_hundred_eighty_five/'
                if count_list[37] == 0:
                    save_list[37] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[37] += 1

        # 185 <= sum < 190
        elif 185 <= target.sum() < 190 and count_list[38] < common_count:
            if is_check:
                save_path = 'sum_hundred_eighty_five_to_hundred_ninety/'
                if count_list[38] == 0:
                    save_list[38] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[38] += 1

        # 190 <= sum < 195
        elif 190 <= target.sum() < 195 and count_list[39] < common_count:
            if is_check:
                save_path = 'sum_hundred_ninety_to_hundred_ninety_five/'
                if count_list[39] == 0:
                    save_list[39] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[39] += 1

        # 195 <= sum < 200
        elif 195 <= target.sum() < 200 and count_list[40] < common_count:
            if is_check:
                save_path = 'sum_hundred_ninety_five_to_two_hundred/'
                if count_list[40] == 0:
                    save_list[40] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[40] += 1

        # 200 <= sum < 205
        elif 200 <= target.sum() < 205 and count_list[41] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_to_two_hundred_five/'
                if count_list[41] == 0:
                    save_list[41] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[41] += 1

        # 205 <= sum < 210
        elif 205 <= target.sum() < 210 and count_list[42] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_five_to_two_hundred_ten/'
                if count_list[42] == 0:
                    save_list[42] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[42] += 1

        # 210 <= sum < 215
        elif 210 <= target.sum() < 215 and count_list[43] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_ten_to_two_hundred_fifteen/'
                if count_list[43] == 0:
                    save_list[43] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[43] += 1

        # 215 <= sum < 220
        elif 215 <= target.sum() < 220 and count_list[44] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_fifteen_to_two_hundred_twenty/'
                if count_list[44] == 0:
                    save_list[44] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[44] += 1

        # 220 <= sum < 225
        elif 220 <= target.sum() < 225 and count_list[45] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_twenty_to_two_hundred_twenty_five/'
                if count_list[45] == 0:
                    save_list[45] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[45] += 1

        # 225 <= sum < 230
        elif 225 <= target.sum() < 230 and count_list[46] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_twenty_five_to_two_hundred_thirty/'
                if count_list[46] == 0:
                    save_list[46] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[46] += 1

        # 230 <= sum < 235
        elif 230 <= target.sum() < 235 and count_list[47] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_thirty_to_two_hundred_thirty_five/'
                if count_list[47] == 0:
                    save_list[47] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[47] += 1

        # 235 <= sum < 240
        elif 235 <= target.sum() < 240 and count_list[48] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_thirty_five_to_two_hundred_forty/'
                if count_list[48] == 0:
                    save_list[48] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[48] += 1

        # 240 <= sum < 245
        elif 240 <= target.sum() < 245 and count_list[49] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_forty_to_two_hundred_forty_five/'
                if count_list[49] == 0:
                    save_list[49] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[49] += 1

        # 245 <= sum < 250
        elif 245 <= target.sum() < 250 and count_list[50] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_forty_five_to_two_hundred_fifty/'
                if count_list[50] == 0:
                    save_list[50] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[50] += 1

        # 250 <= sum < 255
        elif 250 <= target.sum() < 255 and count_list[51] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_fifty_to_two_hundred_fifty_five/'
                if count_list[51] == 0:
                    save_list[51] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[51] += 1

        # 255 <= sum < 260
        elif 255 <= target.sum() < 260 and count_list[52] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_fifty_five_to_two_hundred_sixty/'
                if count_list[52] == 0:
                    save_list[52] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[52] += 1

        # 260 <= sum < 265
        elif 260 <= target.sum() < 265 and count_list[53] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_sixty_to_two_hundred_sixty_five/'
                if count_list[53] == 0:
                    save_list[53] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[53] += 1

        # 265 <= sum < 270
        elif 265 <= target.sum() < 270 and count_list[54] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_sixty_five_to_two_hundred_seventy/'
                if count_list[54] == 0:
                    save_list[54] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[54] += 1

        # 270 <= sum < 275
        elif 270 <= target.sum() < 275 and count_list[55] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_seventy_to_two_hundred_seventy_five/'
                if count_list[55] == 0:
                    save_list[55] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[55] += 1

        # 275 <= sum < 280
        elif 275 <= target.sum() < 280 and count_list[56] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_seventy_five_to_two_hundred_eighty/'
                if count_list[56] == 0:
                    save_list[56] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[56] += 1

        # 280 <= sum < 285
        elif 280 <= target.sum() < 285 and count_list[57] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_eighty_to_two_hundred_eighty_five/'
                if count_list[57] == 0:
                    save_list[57] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[57] += 1

        # 285 <= sum < 290
        elif 285 <= target.sum() < 290 and count_list[58] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_eighty_five_to_two_hundred_ninety/'
                if count_list[58] == 0:
                    save_list[58] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[58] += 1

        # 290 <= sum < 295
        elif 290 <= target.sum() < 295 and count_list[59] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_ninety_to_two_hundred_ninety_five/'
                if count_list[59] == 0:
                    save_list[59] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[59] += 1

        # 295 <= sum < 300
        elif 295 <= target.sum() < 300 and count_list[60] < common_count:
            if is_check:
                save_path = 'sum_two_hundred_ninety_five_to_three_hundred/'
                if count_list[60] == 0:
                    save_list[60] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[60] += 1

        # 300 <= sum < 305
        elif 300 <= target.sum() < 305 and count_list[61] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_to_three_hundred_five/'
                if count_list[61] == 0:
                    save_list[61] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[61] += 1

        # 305 <= sum < 310
        elif 305 <= target.sum() < 310 and count_list[62] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_five_to_three_hundred_ten/'
                if count_list[62] == 0:
                    save_list[62] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[62] += 1

        # 310 <= sum < 315
        elif 310 <= target.sum() < 315 and count_list[63] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_ten_to_three_hundred_fifteen/'
                if count_list[63] == 0:
                    save_list[63] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[63] += 1

        # 315 <= sum < 320
        elif 315 <= target.sum() < 320 and count_list[64] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_fifteen_to_three_hundred_twenty/'
                if count_list[64] == 0:
                    save_list[64] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[64] += 1

        # 320 <= sum < 325
        elif 320 <= target.sum() < 325 and count_list[65] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_twenty_to_three_hundred_twenty_five/'
                if count_list[65] == 0:
                    save_list[65] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[65] += 1

        # 325 <= sum < 330
        elif 325 <= target.sum() < 330 and count_list[66] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_twenty_five_to_three_hundred_thirty/'
                if count_list[66] == 0:
                    save_list[66] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[66] += 1

        # 330 <= sum < 335
        elif 330 <= target.sum() < 335 and count_list[67] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_thirty_to_three_hundred_thirty_five/'
                if count_list[67] == 0:
                    save_list[67] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[67] += 1

        # 335 <= sum < 340
        elif 335 <= target.sum() < 340 and count_list[68] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_thirty_five_to_three_hundred_forty/'
                if count_list[68] == 0:
                    save_list[68] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[68] += 1

        # 340 <= sum < 345
        elif 340 <= target.sum() < 345 and count_list[69] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_forty_to_three_hundred_forty_five/'
                if count_list[69] == 0:
                    save_list[69] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[69] += 1

        # 345 <= sum < 350
        elif 345 <= target.sum() < 350 and count_list[70] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_forty_five_to_three_hundred_fifty/'
                if count_list[70] == 0:
                    save_list[70] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[70] += 1

        # 350 <= sum < 355
        elif 350 <= target.sum() < 355 and count_list[71] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_fifty_to_three_hundred_fifty_five/'
                if count_list[71] == 0:
                    save_list[71] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[71] += 1

        # 355 <= sum < 360
        elif 355 <= target.sum() < 360 and count_list[72] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_fifty_five_to_three_hundred_sixty/'
                if count_list[72] == 0:
                    save_list[72] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[72] += 1

        # 360 <= sum < 365
        elif 360 <= target.sum() < 365 and count_list[73] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_sixty_to_three_hundred_sixty_five/'
                if count_list[73] == 0:
                    save_list[73] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[73] += 1

        # 365 <= sum < 370
        elif 365 <= target.sum() < 370 and count_list[74] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_sixty_five_to_three_hundred_seventy/'
                if count_list[74] == 0:
                    save_list[74] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[74] += 1

        # 370 <= sum < 375
        elif 370 <= target.sum() < 375 and count_list[75] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_seventy_to_three_hundred_seventy_five/'
                if count_list[75] == 0:
                    save_list[75] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[75] += 1

        # 375 <= sum < 380
        elif 375 <= target.sum() < 380 and count_list[76] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_seventy_five_to_three_hundred_eighty/'
                if count_list[76] == 0:
                    save_list[76] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[76] += 1

        # 380 <= sum < 385
        elif 380 <= target.sum() < 385 and count_list[77] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_eighty_to_three_hundred_eighty_five/'
                if count_list[77] == 0:
                    save_list[77] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[77] += 1

        # 385 <= sum < 390
        elif 385 <= target.sum() < 390 and count_list[78] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_eighty_five_to_three_hundred_ninety/'
                if count_list[78] == 0:
                    save_list[78] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[78] += 1

        # 390 <= sum < 395
        elif 390 <= target.sum() < 395 and count_list[79] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_ninety_to_three_hundred_ninety_five/'
                if count_list[79] == 0:
                    save_list[79] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[79] += 1

        # 395 <= sum < 400
        elif 395 <= target.sum() < 400 and count_list[80] < common_count:
            if is_check:
                save_path = 'sum_three_hundred_ninety_five_to_four_hundred/'
                if count_list[80] == 0:
                    save_list[80] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[80] += 1

        # 400 <= sum < 405
        elif 400 <= target.sum() < 405 and count_list[81] < common_count:
            if is_check:
                save_path = 'sum_four_hundred_to_four_hundred_five/'
                if count_list[81] == 0:
                    save_list[81] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[81] += 1

        # 405 <= sum < 410
        elif 405 <= target.sum() < 410 and count_list[82] < common_count:
            if is_check:
                save_path = 'sum_four_hundred_five_to_four_hundred_ten/'
                if count_list[82] == 0:
                    save_list[82] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[82] += 1

        # 410 <= sum < 415
        elif 410 <= target.sum() < 415 and count_list[83] < common_count:
            if is_check:
                save_path = 'sum_four_hundred_ten_to_four_hundred_fifteen/'
                if count_list[83] == 0:
                    save_list[83] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[83] += 1

        # 415 <= sum < 420
        elif 415 <= target.sum() < 420 and count_list[84] < common_count:
            if is_check:
                save_path = 'sum_four_hundred_fifteen_to_four_hundred_twenty/'
                if count_list[84] == 0:
                    save_list[84] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[84] += 1

        # 420 <= sum < 425
        elif 420 <= target.sum() < 425 and count_list[85] < common_count:
            if is_check:
                save_path = 'sum_four_hundred_twenty_to_four_hundred_twenty_five/'
                if count_list[85] == 0:
                    save_list[85] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[85] += 1

        # 425 <= sum < 430
        elif 425 <= target.sum() < 430 and count_list[86] < common_count:
            if is_check:
                save_path = 'sum_four_hundred_twenty_five_to_four_hundred_thirty/'
                if count_list[86] == 0:
                    save_list[86] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[86] += 1

        # 430 <= sum < 435
        elif 430 <= target.sum() < 435 and count_list[87] < common_count:
            if is_check:
                save_path = 'sum_four_hundred_thirty_to_four_hundred_thirty_five/'
                if count_list[87] == 0:
                    save_list[87] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[87] += 1

        # 435 <= sum < 440
        elif 435 <= target.sum() < 440 and count_list[88] < common_count:
            if is_check:
                save_path = 'sum_four_hundred_thirty_five_to_four_hundred_forty/'
                if count_list[88] == 0:
                    save_list[88] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[88] += 1

        # 440 <= sum < 445
        elif 440 <= target.sum() < 445 and count_list[89] < common_count:
            if is_check:
                save_path = 'sum_four_hundred_forty_to_four_hundred_forty_five/'
                if count_list[89] == 0:
                    save_list[89] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[89] += 1

        # 445 <= sum < 450
        elif 445 <= target.sum() < 450 and count_list[90] < common_count:
            if is_check:
                save_path = 'sum_four_hundred_forty_five_to_four_hundred_fifty/'
                if count_list[90] == 0:
                    save_list[90] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[90] += 1

        # 450 <= sum < 455
        elif 450 <= target.sum() < 455 and count_list[91] < common_count:
            if is_check:
                save_path = 'sum_four_hundred_fifty_to_four_hundred_fifty_five/'
                if count_list[91] == 0:
                    save_list[91] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[91] += 1

        # 455 <= sum < 465
        elif 455 <= target.sum() < 465 and count_list[92] < common_count:
            if is_check:
                save_path = 'sum_four_hundred_fifty_five_to_four_hundred_sixty_five/'
                if count_list[92] == 0:
                    save_list[92] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[92] += 1

        # 465 <= sum < 475
        elif 465 <= target.sum() < 475 and count_list[93] < common_count:
            if is_check:
                save_path = 'sum_four_hundred_sixty_five_to_four_hundred_seventy_five/'
                if count_list[93] == 0:
                    save_list[93] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[93] += 1

        # 475 <= sum < 485
        elif 475 <= target.sum() < 485 and count_list[94] < common_count:
            if is_check:
                save_path = 'sum_four_hundred_seventy_five_to_four_hundred_eighty_five/'
                if count_list[94] == 0:
                    save_list[94] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[94] += 1

        # 485 <= sum < 495
        elif 485 <= target.sum() < 495 and count_list[95] < common_count:
            if is_check:
                save_path = 'sum_four_hundred_eighty_five_to_four_hundred_ninety_five/'
                if count_list[95] == 0:
                    save_list[95] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[95] += 1

        # 495 <= sum < 505
        elif 495 <= target.sum() < 505 and count_list[96] < common_count:
            if is_check:
                save_path = 'sum_four_hundred_ninety_five_to_five_hundred_five/'
                if count_list[96] == 0:
                    save_list[96] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[96] += 1

        # 505 <= sum < 515
        elif 505 <= target.sum() < 515 and count_list[97] < common_count:
            if is_check:
                save_path = 'sum_five_hundred_five_to_five_hundred_fifteen/'
                if count_list[97] == 0:
                    save_list[97] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[97] += 1

        # 515 <= sum < 525
        elif 515 <= target.sum() < 525 and count_list[98] < common_count:
            if is_check:
                save_path = 'sum_five_hundred_fifteen_to_five_hundred_twenty_five/'
                if count_list[98] == 0:
                    save_list[98] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[98] += 1

        # 525 <= sum < 535
        elif 525 <= target.sum() < 535 and count_list[99] < common_count:
            if is_check:
                save_path = 'sum_five_hundred_twenty_five_to_five_hundred_thirty_five/'
                if count_list[99] == 0:
                    save_list[99] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[99] += 1

        # 535 <= sum < 545
        elif 535 <= target.sum() < 545 and count_list[100] < common_count:
            if is_check:
                save_path = 'sum_five_hundred_thirty_five_to_five_hundred_forty_five/'
                if count_list[100] == 0:
                    save_list[100] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[100] += 1

        # 545 <= sum < 555
        elif 545 <= target.sum() < 555 and count_list[101] < common_count:
            if is_check:
                save_path = 'sum_five_hundred_forty_five_to_five_hundred_fifty_five/'
                if count_list[101] == 0:
                    save_list[101] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[101] += 1

        # 555 <= sum < 565
        elif 555 <= target.sum() < 565 and count_list[102] < common_count:
            if is_check:
                save_path = 'sum_five_hundred_fifty_five_to_five_hundred_sixty_five/'
                if count_list[102] == 0:
                    save_list[102] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[102] += 1

        # 565 <= sum < 575
        elif 565 <= target.sum() < 575 and count_list[103] < common_count:
            if is_check:
                save_path = 'sum_five_hundred_sixty_five_to_five_hundred_seventy_five/'
                if count_list[103] == 0:
                    save_list[103] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[103] += 1

        # 575 <= sum < 585
        elif 575 <= target.sum() < 585 and count_list[104] < common_count:
            if is_check:
                save_path = 'sum_five_hundred_seventy_five_to_five_hundred_eighty_five/'
                if count_list[104] == 0:
                    save_list[104] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[104] += 1

        # 585 <= sum < 595
        elif 585 <= target.sum() < 595 and count_list[105] < common_count:
            if is_check:
                save_path = 'sum_five_hundred_eighty_five_to_five_hundred_ninety_five/'
                if count_list[105] == 0:
                    save_list[105] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[105] += 1

        # 595 <= sum < 605
        elif 595 <= target.sum() < 605 and count_list[106] < common_count:
            if is_check:
                save_path = 'sum_five_hundred_ninety_five_to_six_hundred_five/'
                if count_list[106] == 0:
                    save_list[106] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[106] += 1

        # 605 <= sum < 615
        elif 605 <= target.sum() < 615 and count_list[107] < common_count:
            if is_check:
                save_path = 'sum_six_hundred_five_to_six_hundred_fifteen/'
                if count_list[107] == 0:
                    save_list[107] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[107] += 1

        # 615 <= sum < 625
        elif 615 <= target.sum() < 625 and count_list[108] < common_count:
            if is_check:
                save_path = 'sum_six_hundred_fifteen_to_six_hundred_twenty_five/'
                if count_list[108] == 0:
                    save_list[108] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[108] += 1

        # 625 <= sum < 635
        elif 625 <= target.sum() < 635 and count_list[109] < common_count:
            if is_check:
                save_path = 'sum_six_hundred_twenty_five_to_six_hundred_thirty_five/'
                if count_list[109] == 0:
                    save_list[109] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[109] += 1

        # 635 <= sum < 645
        elif 635 <= target.sum() < 645 and count_list[110] < common_count:
            if is_check:
                save_path = 'sum_six_hundred_thirty_five_to_six_hundred_forty_five/'
                if count_list[110] == 0:
                    save_list[110] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[110] += 1

        # 645 <= sum < 655
        elif 645 <= target.sum() < 655 and count_list[111] < common_count:
            if is_check:
                save_path = 'sum_six_hundred_forty_five_to_six_hundred_fifty_five/'
                if count_list[111] == 0:
                    save_list[111] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[111] += 1

        # 655 <= sum < 665
        elif 655 <= target.sum() < 665 and count_list[112] < common_count:
            if is_check:
                save_path = 'sum_six_hundred_fifty_five_to_six_hundred_sixty_five/'
                if count_list[112] == 0:
                    save_list[112] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[112] += 1

        # 665 <= sum < 675
        elif 665 <= target.sum() < 675 and count_list[113] < common_count:
            if is_check:
                save_path = 'sum_six_hundred_sixty_five_to_six_hundred_seventy_five/'
                if count_list[113] == 0:
                    save_list[113] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[113] += 1

        # 675 <= sum < 685
        elif 675 <= target.sum() < 685 and count_list[114] < common_count:
            if is_check:
                save_path = 'sum_six_hundred_seventy_five_to_six_hundred_eighty_five/'
                if count_list[114] == 0:
                    save_list[114] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[114] += 1

        # 685 <= sum < 695
        elif 685 <= target.sum() < 695 and count_list[115] < common_count:
            if is_check:
                save_path = 'sum_six_hundred_eighty_five_to_six_hundred_ninety_five/'
                if count_list[115] == 0:
                    save_list[115] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[115] += 1

        # 695 <= sum < 705
        elif 695 <= target.sum() < 705 and count_list[116] < common_count:
            if is_check:
                save_path = 'sum_six_hundred_ninety_five_to_seven_hundred_five/'
                if count_list[116] == 0:
                    save_list[116] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[116] += 1

        # 705 <= sum < 725
        elif 705 <= target.sum() < 725 and count_list[117] < common_count:
            if is_check:
                save_path = 'sum_seven_hundred_five_to_seven_hundred_twenty_five/'
                if count_list[117] == 0:
                    save_list[117] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[117] += 1

        # 725 <= sum < 745
        elif 725 <= target.sum() < 745 and count_list[118] < common_count:
            if is_check:
                save_path = 'sum_seven_hundred_twenty_five_to_seven_hundred_forty_five/'
                if count_list[118] == 0:
                    save_list[118] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[118] += 1

        # 745 <= sum < 765
        elif 745 <= target.sum() < 765 and count_list[119] < common_count:
            if is_check:
                save_path = 'sum_seven_hundred_forty_five_to_seven_hundred_sixty_five/'
                if count_list[119] == 0:
                    save_list[119] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[119] += 1

        # 765 <= sum < 785
        elif 765 <= target.sum() < 785 and count_list[120] < common_count:
            if is_check:
                save_path = 'sum_seven_hundred_sixty_five_to_seven_hundred_eighty_five/'
                if count_list[120] == 0:
                    save_list[120] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[120] += 1

        # 785 <= sum < 805
        elif 785 <= target.sum() < 805 and count_list[121] < common_count:
            if is_check:
                save_path = 'sum_seven_hundred_eighty_five_to_eigh_hundred_five/'
                if count_list[121] == 0:
                    save_list[121] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[121] += 1

        # 805 <= sum < 825
        elif 805 <= target.sum() < 825 and count_list[122] < common_count:
            if is_check:
                save_path = 'sum_eight_hundred_five_to_eight_hundred_twenty_five/'
                if count_list[122] == 0:
                    save_list[122] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[122] += 1

        # 825 <= sum < 845
        elif 825 <= target.sum() < 845 and count_list[123] < common_count:
            if is_check:
                save_path = 'sum_eight_hundred_twenty_five_to_eight_hundred_forty_five/'
                if count_list[123] == 0:
                    save_list[123] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[123] += 1

        # 845 <= sum < 865
        elif 845 <= target.sum() < 865 and count_list[124] < common_count:
            if is_check:
                save_path = 'sum_eight_hundred_forty_five_to_eight_hundred_sixty_five/'
                if count_list[124] == 0:
                    save_list[124] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[124] += 1

        # 865 <= sum < 885
        elif 865 <= target.sum() < 885 and count_list[125] < common_count:
            if is_check:
                save_path = 'sum_eight_hundred_sixty_five_to_eight_hundred_eighty_five/'
                if count_list[125] == 0:
                    save_list[125] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[125] += 1

        # 885 <= sum < 905
        elif 885 <= target.sum() < 905 and count_list[126] < common_count:
            if is_check:
                save_path = 'sum_eight_hundred_eighty_five_to_nine_hundred_five/'
                if count_list[126] == 0:
                    save_list[126] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[126] += 1

        # 905 <= sum < 925
        elif 905 <= target.sum() < 925 and count_list[127] < common_count:
            if is_check:
                save_path = 'sum_nine_hundred_five_to_nine_hundred_twenty_five/'
                if count_list[127] == 0:
                    save_list[127] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[127] += 1

        # 925 <= sum < 945
        elif 925 <= target.sum() < 945 and count_list[128] < common_count:
            if is_check:
                save_path = 'sum_nine_hundred_twenty_five_to_nine_hundred_forty_five/'
                if count_list[128] == 0:
                    save_list[128] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[128] += 1

        # 945 <= sum < 965
        elif 945 <= target.sum() < 965 and count_list[129] < common_count:
            if is_check:
                save_path = 'sum_nine_hundred_forty_five_to_nine_hundred_sixty_five/'
                if count_list[129] == 0:
                    save_list[129] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[129] += 1

        # 965 <= sum < 985
        elif 965 <= target.sum() < 985 and count_list[130] < common_count:
            if is_check:
                save_path = 'sum_nine_hundred_sixty_five_to_nine_hundred_eighty_five/'
                if count_list[130] == 0:
                    save_list[130] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[130] += 1

        # 985 <= sum < 1005
        elif 985 <= target.sum() < 1005 and count_list[131] < common_count:
            if is_check:
                save_path = 'sum_nine_hundred_eighty_five_to_one_thousand_five/'
                if count_list[131] == 0:
                    save_list[131] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[131] += 1

        # 1005 <= sum < 1025
        elif 1005 <= target.sum() < 1025 and count_list[132] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_five_to_one_thousand_twenty_five/'
                if count_list[132] == 0:
                    save_list[132] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[132] += 1

        # 1025 <= sum < 1045
        elif 1025 <= target.sum() < 1045 and count_list[133] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_twenty_five_to_one_thousand_forty_five/'
                if count_list[133] == 0:
                    save_list[133] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[133] += 1

        # 1045 <= sum < 1065
        elif 1045 <= target.sum() < 1065 and count_list[134] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_forty_five_to_one_thousand_sixty_five/'
                if count_list[134] == 0:
                    save_list[134] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[134] += 1

        # 1065 <= sum < 1085
        elif 1065 <= target.sum() < 1085 and count_list[135] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_sixty_five_to_one_thousand_eighty_five/'
                if count_list[135] == 0:
                    save_list[135] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[135] += 1

        # 1085 <= sum < 1105
        elif 1085 <= target.sum() < 1105 and count_list[136] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_eighty_five_to_one_thousand_one_hundred_five/'
                if count_list[136] == 0:
                    save_list[136] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[136] += 1

        # 1105 <= sum < 1125
        elif 1105 <= target.sum() < 1125 and count_list[137] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_one_hundred_five_to_one_thousand_one_hundred_twenty_five/'
                if count_list[137] == 0:
                    save_list[137] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[137] += 1

        # 1125 <= sum < 1145
        elif 1125 <= target.sum() < 1145 and count_list[138] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_one_hundred_twenty_five_to_one_thousand_one_hundred_forty_five/'
                if count_list[138] == 0:
                    save_list[138] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[138] += 1

        # 1145 <= sum < 1165
        elif 1145 <= target.sum() < 1165 and count_list[139] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_one_hundred_forty_five_to_one_thousand_one_hundred_sixty_five/'
                if count_list[139] == 0:
                    save_list[139] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[139] += 1

        # 1165 <= sum < 1185
        elif 1165 <= target.sum() < 1185 and count_list[140] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_one_hundred_sixty_five_to_one_thousand_one_hundred_eighty_five/'
                if count_list[140] == 0:
                    save_list[140] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[140] += 1

        # 1185 <= sum < 1205
        elif 1185 <= target.sum() < 1205 and count_list[141] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_one_hundred_eighty_five_to_one_thousand_tow_hundred_five/'
                if count_list[141] == 0:
                    save_list[141] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[141] += 1

        # 1205 <= sum < 1255
        elif 1205 <= target.sum() < 1255 and count_list[142] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_two_hundred_five_to_one_thousand_tow_hundred_fifty_five/'
                if count_list[142] == 0:
                    save_list[142] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[142] += 1

        # 1255 <= sum < 1305
        elif 1255 <= target.sum() < 1305 and count_list[143] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_two_hundred_fifty_five_to_one_thousand_three_hundred_five/'
                if count_list[143] == 0:
                    save_list[143] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[143] += 1

        # 1305 <= sum < 1355
        elif 1305 <= target.sum() < 1355 and count_list[144] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_three_hundred_five_to_one_thousand_three_hundred_fifty_five/'
                if count_list[144] == 0:
                    save_list[144] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[144] += 1

        # 1355 <= sum < 1405
        elif 1355 <= target.sum() < 1405 and count_list[145] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_three_hundred_fifty_five_to_one_thousand_four_hundred_five/'
                if count_list[145] == 0:
                    save_list[145] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[145] += 1

        # 1405 <= sum < 1455
        elif 1405 <= target.sum() < 1455 and count_list[146] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_four_hundred_five_to_one_thousand_four_hundred_fifty_five/'
                if count_list[146] == 0:
                    save_list[146] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[146] += 1

        # 1455 <= sum < 1505
        elif 1455 <= target.sum() < 1505 and count_list[147] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_four_hundred_fifty_five_to_one_thousand_five_hundred_five/'
                if count_list[147] == 0:
                    save_list[147] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[147] += 1

        # 1505 <= sum < 1555
        elif 1505 <= target.sum() < 1555 and count_list[148] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_five_hundred_five_to_one_thousand_five_hundred_fifty_five/'
                if count_list[148] == 0:
                    save_list[148] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[148] += 1

        # 1555 <= sum < 1605
        elif 1555 <= target.sum() < 1605 and count_list[149] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_five_hundred_fifty_five_to_one_thousand_six_hundred_five/'
                if count_list[149] == 0:
                    save_list[149] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[149] += 1

        # 1605 <= sum < 1705
        elif 1605 <= target.sum() < 1705 and count_list[150] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_six_hundred_five_to_one_thousand_seven_hundred_five/'
                if count_list[150] == 0:
                    save_list[150] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[150] += 1

        # 1705 <= sum < 1805
        elif 1705 <= target.sum() < 1805 and count_list[151] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_seven_hundred_five_to_one_thousand_eight_hundred_five/'
                if count_list[151] == 0:
                    save_list[151] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[151] += 1

        # 1805 <= sum < 1905
        elif 1805 <= target.sum() < 1905 and count_list[152] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_eight_hundred_five_to_one_thousand_nine_hundred_five/'
                if count_list[152] == 0:
                    save_list[152] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[152] += 1

        # 1905 <= sum < 2005
        elif 1905 <= target.sum() < 2005 and count_list[153] < common_count:
            if is_check:
                save_path = 'sum_one_thousand_nine_hundred_five_to_two_thousand_five/'
                if count_list[153] == 0:
                    save_list[153] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[153] += 1

        # 2005 <= sum < 2105
        elif 2005 <= target.sum() < 2105 and count_list[154] < common_count:
            if is_check:
                save_path = 'sum_tow_thousand_five_to_two_thousand_one_hundred_five/'
                if count_list[154] == 0:
                    save_list[154] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[154] += 1

        # 2105 <= sum < 2205
        elif 2105 <= target.sum() < 2205 and count_list[155] < common_count:
            if is_check:
                save_path = 'sum_tow_thousand_one_hundred_five_to_two_thousand_two_hundred_five/'
                if count_list[155] == 0:
                    save_list[155] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[155] += 1

        # 2205 <= sum < 2305
        elif 2205 <= target.sum() < 2305 and count_list[156] < common_count:
            if is_check:
                save_path = 'sum_tow_thousand_two_hundred_five_to_two_thousand_three_hundred_five/'
                if count_list[156] == 0:
                    save_list[156] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[156] += 1

        # 2305 <= sum < 2505
        elif 2305 <= target.sum() < 2505 and count_list[157] < common_count:
            if is_check:
                save_path = 'sum_two_thousand_three_hundred_five_to_two_thousand_five_hundred_five/'
                if count_list[157] == 0:
                    save_list[157] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[157] += 1

        # 2505 <= sum < 2705
        elif 2505 <= target.sum() < 2705 and count_list[158] < common_count:
            if is_check:
                save_path = 'sum_two_thousand_five_hundred_five_to_two_thousand_seven_hundred_five/'
                if count_list[158] == 0:
                    save_list[158] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[158] += 1

        # 2705 <= sum < 3005
        elif 2705 <= target.sum() < 3005 and count_list[159] < common_count:
            if is_check:
                save_path = 'sum_two_thousand_seven_hundred_five_to_three_thousand_five/'
                if count_list[159] == 0:
                    save_list[159] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[159] += 1

        # 3005 <= sum < 3405
        elif 3005 <= target.sum() < 3405 and count_list[160] < common_count:
            if is_check:
                save_path = 'sum_three_thousand_five_to_three_thousand_four_hundred_five/'
                if count_list[160] == 0:
                    save_list[160] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[160] += 1

        # 3405 <= sum < 3905
        elif 3405 <= target.sum() < 3905 and count_list[161] < common_count:
            if is_check:
                save_path = 'sum_three_thousand_four_hundred_to_three_thousand_nine_hundred_five/'
                if count_list[161] == 0:
                    save_list[161] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[161] += 1

        # 3905 <= sum < 4705
        elif 3905 <= target.sum() < 4705 and count_list[162] < common_count:
            if is_check:
                save_path = 'sum_three_thousand_nine_hundred_to_four_thousand_seven_hundred_five/'
                if count_list[162] == 0:
                    save_list[162] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[162] += 1

        # 4705 <= sum < 7005
        elif 4705 <= target.sum() < 7005 and count_list[163] < common_count:
            if is_check:
                save_path = 'sum_four_thousand_seven_hundred_five_to_seven_thousand_five/'
                if count_list[163] == 0:
                    save_list[163] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[163] += 1

        # 7005 <= sum
        elif 7005 <= target.sum() and count_list[164] < common_count:
            if is_check:
                save_path = 'sum_seven_thousand_five_to_else/'
                if count_list[164] == 0:
                    save_list[164] = save_path
                if isfile(common_save_path + save_path + real_file_name):
                    pass
                else:
                    np.save(common_save_path + save_path + real_file_name, one_npy_data)
            count_list[164] += 1


    x = range(0, 165)
    plt.plot(x, count_list)
    plt.show()

    with open('./count_list.pickle', 'wb') as f:
        pickle.dump(count_list, f, pickle.HIGHEST_PROTOCOL)

    with open('./save_list.pickle', 'wb') as f:
        pickle.dump(save_list, f, pickle.HIGHEST_PROTOCOL)


    #with open('./count_list.pickle', 'rb') as c:
    #    count_list = pickle.load(c)

    #with open('./save_list.pickle', 'rb') as s:
    #    save_list = pickle.load(s)

    # 데이터 부풀리기
    ###################################################
    idx = 0
    new_name = 0

    for idx_count in tqdm(count_list):
        # 지정한 데이터 갯수보다 작으면 데이터를 부풀립니다.
        if idx_count < common_count:
            save_path = save_list[idx]
            save_path_replace = save_path.replace('/', '')
            save_files = sorted(glob.glob('./my_train/' + save_path + '/*'))
            save_files = save_files[::]

            # 반전
            for file in tqdm(save_files):

                if count_list[idx] == common_count:
                    break

                present_file = np.load(file)
                x_T = np.zeros_like(present_file)


                # col의 갯수만큼
                for i in range(15):
                    x_T[:, :, i] = present_file[:, :, i].T

                np.save(common_save_path + save_path + "new_reversal_" + str(new_name), x_T)
                count_list[idx] += 1
                new_name += 1

            # 90회전
            for file in tqdm(save_files):

                if count_list[idx] == common_count:
                    break

                present_file = np.load(file)
                rotate_90 = np.zeros_like(present_file)


                # col의 개수만큼
                for i in range(15):
                    rotate_90[:, :, i] = np.rot90(present_file[:, :, i])

                np.save(common_save_path + save_path + "new_rotate90_" + str(new_name), rotate_90)
                count_list[idx] += 1
                new_name += 1

            # 180 최전
            for file in tqdm(save_files):

                if count_list[idx] == common_count:
                    break

                present_file = np.load(file)
                rotate_180 = np.zeros_like(present_file)


                # col의 개수만큼
                for i in range(15):
                    rotate_180[:, :, i] = np.rot90(np.rot90(present_file[:, :, i]))

                np.save(common_save_path + save_path + "new_rotate180_" + str(new_name), rotate_180)
                count_list[idx] += 1
                new_name += 1
                
            # 270 회전
            for file in tqdm(save_files):

                if count_list[idx] == common_count:
                    break

                present_file = np.load(file)
                rotate_270 = np.zeros_like(present_file)


                # col의 개수만큼
                for i in range(15):
                    rotate_270[:, :, i] = np.rot90(np.rot90(np.rot90(present_file[:, :, i])))

                np.save(common_save_path + save_path + "new_rotate270_" + str(new_name), rotate_270)
                count_list[idx] += 1
                new_name += 1

            # 90회전 - 반전
            for file in tqdm(save_files):

                if count_list[idx] == common_count:
                    break

                present_file = np.load(file)
                rotate_90_T = np.zeros_like(present_file)

                # col의 개수만큼
                for i in range(15):
                    rotate_90_T[:, :, i] = np.rot90(present_file[:, :, i]).T

                np.save(common_save_path + save_path + "new_rotate90_T_" + str(new_name), rotate_90_T)
                count_list[idx] += 1
                new_name += 1

            # 180 최전 - 반전
            for file in tqdm(save_files):

                if count_list[idx] == common_count:
                    break

                present_file = np.load(file)
                rotate_180_T = np.zeros_like(present_file)


                # col의 개수만큼
                for i in range(15):
                    rotate_180_T[:, :, i] = np.rot90(np.rot90(present_file[:, :, i])).T

                np.save(common_save_path + save_path + "new_rotate180_T_" + str(new_name), rotate_180_T)
                count_list[idx] += 1
                new_name += 1

            # 270 회전
            for file in tqdm(save_files):

                if count_list[idx] == common_count:
                    break

                present_file = np.load(file)
                rotate_270_T = np.zeros_like(present_file)


                # col의 개수만큼
                for i in range(15):
                    rotate_270_T[:, :, i] = np.rot90(np.rot90(np.rot90(present_file[:, :, i]))).T

                np.save(common_save_path + save_path + "new_rotate270_T_" + str(new_name), rotate_270_T)
                count_list[idx] += 1
                new_name += 1


        idx += 1

    x = range(0, 165)
    plt.plot(x, count_list)
    plt.show()



make_npy(True)
