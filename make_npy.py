import glob
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# 사용할 데이터 col
# 10 이랑 14 중에 최적이 어느것인지 확인 중 -> 10이 나은거 같음
# 별 차이 없으면 10 씀
my_col = 10

# 강우량 리스트
rain_list = []


#################################################################
def make_npy():
    train_path = './train/train'
    train_files = sorted(glob.glob(train_path + '/*'))
    train_files = train_files[::]

    common_save_path = './my_train/'
    common_count = 500
    zero_count = 0
    one_to_five_count = 0
    five_to_ten_count = 0
    ten_to_fifteen_count = 0
    fifteen_to_twenty_count = 0
    twenty_to_twenty_five_count = 0
    twenty_five_to_thirty_count = 0
    thirty_to_thirty_five_count = 0
    thirty_five_to_forty_count = 0
    forty_to_forty_five_count = 0
    forty_five_to_fifty_count = 0
    fifty_to_fifty_five_count = 0
    fifty_five_to_sixty_count = 0
    sixty_to_sixty_five_count = 0
    sixty_five_to_seventy_count = 0
    seventy_to_seventy_five_count = 0
    seventy_five_to_eighty_count = 0
    eighty_to_eighty_five_count = 0
    eighty_five_to_ninety_count = 0
    ninety_to_ninety_five_count = 0
    ninety_five_to_hundred_count = 0
    hundred_to_hundred_five_count = 0
    hundred_five_to_hundred_ten_count = 0
    hundred_ten_to_hundred_fifteen_count = 0
    hundred_fifteen_to_hundred_twenty_count = 0
    hundred_twenty_to_hundred_twenty_five_count = 0
    hundred_twenty_five_to_hundred_thirty_count = 0
    hundred_thirty_to_hundred_thirty_five_count = 0
    hundred_thirty_five_to_hundred_forty_count = 0
    hundred_forty_to_hundred_forty_five_count = 0
    hundred_forty_five_to_hundred_fifty_count = 0
    hundred_fifty_to_hundred_fifty_five_count = 0
    hundred_fifty_five_to_hundred_sixty_count = 0
    hundred_sixty_to_hundred_sixty_five_count = 0
    hundred_sixty_five_to_hundred_seventy_count = 0
    hundred_seventy_to_hundred_seventy_five_count = 0
    hundred_seventy_five_to_hundred_eighty_count = 0
    hundred_eighty_to_hundred_eighty_five_count = 0
    hundred_eighty_five_to_hundred_ninety_count = 0
    hundred_ninety_to_hundred_ninety_five_count = 0
    hundred_ninety_five_to_hundred_two_hundred_count = 0


    for npy_file in train_files:

        one_npy_data = np.load(npy_file)
        real_file_name = npy_file.replace('train', '').replace('/', '') #.replace('.', '').replace('\\', '')

        # 강수량 값
        target = one_npy_data[:, :, -1].reshape(40, 40, 1)

        # 강수량이 0보다 작은 것 제거
        if target.sum() < 0:
            continue

        # 강수량이 0 < x < 1 인것 제거
        if (target.sum() > 0) and (target.sum() < 1):
            continue

        # sum = 0
        if target.sum() == 0 and zero_count < common_count:
            save_path = 'sum_zero/'
            np.save(common_save_path + save_path + real_file_name, target)
            zero_count += 1

        # 1 <= sum < 5
        elif 1 <= target.sum() < 5 and one_to_five_count < common_count:
            save_path = 'sum_zero_to_five/'
            np.save(common_save_path + save_path + real_file_name, target)
            one_to_five_count += 1

        # 5 <= sum < 10
        elif 5 <= target.sum() < 10 and five_to_ten_count < common_count:
            save_path = 'sum_five_to_ten/'
            np.save(common_save_path + save_path + real_file_name, target)
            five_to_ten_count += 1

        # 10 <= sum < 15
        elif 10 <= target.sum() < 15 and ten_to_fifteen_count < common_count:
            save_path = 'sum_ten_to_fifteen/'
            np.save(common_save_path + save_path + real_file_name, target)
            ten_to_fifteen_count += 1

        # 15 <= sum < 20
        elif 15 <= target.sum() < 20 and fifteen_to_twenty_count < common_count:
            save_path = 'sum_fifteen_to_twenty/'
            np.save(common_save_path + save_path + real_file_name, target)
            fifteen_to_twenty_count += 1

        # 20 <= sum < 25
        elif 20 <= target.sum() < 25 and twenty_to_twenty_five_count < common_count:
            save_path = 'sum_twenty_to_twenty_five/'
            np.save(common_save_path + save_path + real_file_name, target)
            twenty_to_twenty_five_count += 1

        # 25 <= sum < 30
        elif 25 <= target.sum() < 30 and twenty_five_to_thirty_count < common_count:
            save_path = 'sum_twenty_five_to_thirty/'
            np.save(common_save_path + save_path + real_file_name, target)
            twenty_five_to_thirty_count += 1

        # 30 <= sum < 35
        elif 30 <= target.sum() < 35 and thirty_to_thirty_five_count < common_count:
            save_path = 'sum_thirty_to_thirty_five/'
            np.save(common_save_path + save_path + real_file_name, target)
            thirty_to_thirty_five_count += 1

        # 35 <= sum < 40
        elif 35 <= target.sum() < 40 and thirty_five_to_forty_count < common_count:
            save_path = 'sum_thirty_five_to_forty/'
            np.save(common_save_path + save_path + real_file_name, target)
            thirty_five_to_forty_count += 1

        # 40 <= sum < 45
        elif 40 <= target.sum() < 45 and forty_to_forty_five_count < common_count:
            save_path = 'sum_forty_to_forty_five/'
            np.save(common_save_path + save_path + real_file_name, target)
            forty_to_forty_five_count += 1

        # 45 <= sum < 50
        elif 45 <= target.sum() < 50 and forty_five_to_fifty_count < common_count:
            save_path = 'sum_forty_five_to_fifty/'
            np.save(common_save_path + save_path + real_file_name, target)
            forty_five_to_fifty_count += 1

        # 50 <= sum < 55
        elif 50 <= target.sum() < 55 and fifty_to_fifty_five_count < common_count:
            save_path = 'sum_fifty_to_fifty_five/'
            np.save(common_save_path + save_path + real_file_name, target)
            fifty_to_fifty_five_count += 1

        # 55 <= sum < 60
        elif 55 <= target.sum() < 60 and fifty_five_to_sixty_count < common_count:
            save_path = 'sum_fifty_five_to_sixty/'
            np.save(common_save_path + save_path + real_file_name, target)
            fifty_five_to_sixty_count += 1

        # 60 <= sum < 65
        elif 60 <= target.sum() < 65 and sixty_to_sixty_five_count < common_count:
            save_path = 'sum_sixty_to_sixty_five/'
            np.save(common_save_path + save_path + real_file_name, target)
            sixty_to_sixty_five_count += 1

        # 65 <= sum < 70
        elif 65 <= target.sum() < 70 and sixty_five_to_seventy_count < common_count:
            save_path = 'sum_sixty_five_to_seventy/'
            np.save(common_save_path + save_path + real_file_name, target)
            sixty_five_to_seventy_count += 1

        # 70 <= sum < 75
        elif 70 <= target.sum() < 75 and seventy_to_seventy_five_count < common_count:
            save_path = 'sum_seventy_to_seventy_five/'
            np.save(common_save_path + save_path + real_file_name, target)
            seventy_to_seventy_five_count += 1

        # 75 <= sum < 80
        elif 75 <= target.sum() < 80 and seventy_five_to_eighty_count < common_count:
            save_path = 'sum_seventy_five_to_eighty/'
            np.save(common_save_path + save_path + real_file_name, target)
            seventy_five_to_eighty_count += 1

        # 80 <= sum < 85
        elif 80 <= target.sum() < 85 and eighty_to_eighty_five_count < common_count:
            save_path = 'sum_eighty_to_eighty_five/'
            np.save(common_save_path + save_path + real_file_name, target)
            eighty_to_eighty_five_count += 1

        # 85 <= sum < 90
        elif 85 <= target.sum() < 90 and eighty_five_to_ninety_count < common_count:
            save_path = 'sum_eighty_five_to_ninety/'
            np.save(common_save_path + save_path + real_file_name, target)
            eighty_five_to_ninety_count += 1

        # 90 <= sum < 95
        elif 90 <= target.sum() < 95 and ninety_to_ninety_five_count < common_count:
            save_path = 'sum_ninety_to_ninety_five/'
            np.save(common_save_path + save_path + real_file_name, target)
            ninety_to_ninety_five_count += 1

        # 95 <= sum < 100
        elif 95 <= target.sum() < 100 and ninety_five_to_hundred_count < common_count:
            save_path = 'sum_ninety_five_to_hundred/'
            np.save(common_save_path + save_path + real_file_name, target)
            ninety_five_to_hundred_count += 1

        # 100 <= sum < 105
        elif 100 <= target.sum() < 105 and hundred_to_hundred_five_count < common_count:
            save_path = 'sum_hundred_to_hundred_five/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_to_hundred_five_count += 1

        # 105 <= sum < 110
        elif 105 <= target.sum() < 110 and hundred_five_to_hundred_ten_count < common_count:
            save_path = 'sum_hundred_five_to_hundred_ten/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_five_to_hundred_ten_count += 1

        # 110 <= sum < 115
        elif 110 <= target.sum() < 115 and hundred_ten_to_hundred_fifteen_count < common_count:
            save_path = 'sum_hundred_ten_to_hundred_fifteen/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_ten_to_hundred_fifteen_count += 1

        # 115 <= sum < 120
        elif 115 <= target.sum() < 120 and hundred_fifteen_to_hundred_twenty_count < common_count:
            save_path = 'sum_hundred_fifteen_to_hundred_twenty/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_fifteen_to_hundred_twenty_count += 1

        # 120 <= sum < 125
        elif 120 <= target.sum() < 125 and hundred_twenty_to_hundred_twenty_five_count < common_count:
            save_path = 'sum_hundred_twenty_to_hundred_twenty_five/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_twenty_to_hundred_twenty_five_count += 1

        # 125 <= sum < 130
        elif 125 <= target.sum() < 130 and hundred_twenty_five_to_hundred_thirty_count < common_count:
            save_path = 'sum_hundred_twenty_five_to_hundred_thirty/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_twenty_five_to_hundred_thirty_count += 1

        # 130 <= sum < 135
        elif 130 <= target.sum() < 135 and hundred_thirty_to_hundred_thirty_five_count < common_count:
            save_path = 'sum_hundred_thirty_to_hundred_thirty_five/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_thirty_to_hundred_thirty_five_count += 1

        # 135 <= sum < 140
        elif 135 <= target.sum() < 140 and hundred_thirty_five_to_hundred_forty_count < common_count:
            save_path = 'sum_hundred_thirty_five_to_hundred_forty/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_thirty_five_to_hundred_forty_count += 1

        # 140 <= sum < 145
        elif 140 <= target.sum() < 145 and hundred_forty_to_hundred_forty_five_count < common_count:
            save_path = 'sum_hundred_forty_to_hundred_forty_five/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_forty_to_hundred_forty_five_count += 1

        # 145 <= sum < 150
        elif 145 <= target.sum() < 150 and hundred_forty_five_to_hundred_fifty_count < common_count:
            save_path = 'sum_hundred_forty_five_to_hundred_fifty/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_forty_five_to_hundred_fifty_count += 1

        # 150 <= sum < 155
        elif 150 <= target.sum() < 155 and hundred_fifty_to_hundred_fifty_five_count < common_count:
            save_path = 'sum_hundred_fifty_to_hundred_fifty_five/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_fifty_to_hundred_fifty_five_count += 1

        # 155 <= sum < 160
        elif 155 <= target.sum() < 160 and hundred_fifty_five_to_hundred_sixty_count < common_count:
            save_path = 'sum_hundred_fifty_five_to_hundred_sixty/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_fifty_five_to_hundred_sixty_count += 1

        # 160 <= sum < 165
        elif 160 <= target.sum() < 165 and hundred_sixty_to_hundred_sixty_five_count < common_count:
            save_path = 'sum_hundred_sixty_to_hundred_sixty_five/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_sixty_to_hundred_sixty_five_count += 1

        # 165 <= sum < 170
        elif 165 <= target.sum() < 170 and hundred_sixty_five_to_hundred_seventy_count < common_count:
            save_path = 'sum_hundred_sixty_five_to_hundred_seventy/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_sixty_five_to_hundred_seventy_count += 1

        # 170 <= sum < 175
        elif 170 <= target.sum() < 175 and hundred_seventy_to_hundred_seventy_five_count < common_count:
            save_path = 'sum_hundred_seventy_to_hundred_seventy_five/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_seventy_to_hundred_seventy_five_count += 1

        # 175 <= sum < 180
        elif 175 <= target.sum() < 180 and hundred_seventy_five_to_hundred_eighty_count < common_count:
            save_path = 'sum_hundred_seventy_five_to_hundred_eighty/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_seventy_five_to_hundred_eighty_count += 1

        # 180 <= sum < 185
        elif 180 <= target.sum() < 185 and hundred_eighty_to_hundred_eighty_five_count < common_count:
            save_path = 'sum_hundred_eighty_to_hundred_eighty_five/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_eighty_to_hundred_eighty_five_count += 1

        # 185 <= sum < 190
        elif 185 <= target.sum() < 190 and hundred_eighty_five_to_hundred_ninety_count < common_count:
            save_path = 'sum_hundred_eighty_five_to_hundred_ninety/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_eighty_five_to_hundred_ninety_count += 1

        # 190 <= sum < 195
        elif 190 <= target.sum() < 195 and hundred_ninety_to_hundred_ninety_five_count < common_count:
            save_path = 'sum_hundred_ninety_to_hundred_ninety_five/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_ninety_to_hundred_ninety_five_count += 1

        # 195 <= sum < 200
        elif 195 <= target.sum() < 200 and hundred_ninety_five_to_hundred_two_hundred_count < common_count:
            save_path = 'sum_hundred_ninety_five_to_hundred_two_hundred/'
            np.save(common_save_path + save_path + real_file_name, target)
            hundred_ninety_five_to_hundred_two_hundred_count += 1

        # 강수량 데이터 분포를 알기위해
        rain_list.append(target.sum())


make_npy()
s = pd.Series(rain_list)
print(s.describe())
