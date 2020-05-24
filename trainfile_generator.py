import glob
import numpy as np
from tqdm import tqdm

my_col = 10

def trainGenerator():
    train_path = './train/final_train'
    train_files = sorted(glob.glob(train_path + '/*'))
    train_files = train_files[::]

    for npy_file in train_files:

        one_npy_data = np.load(npy_file)

        # 평균 뺀게 성능이 좋음
        for npy_colum in range(my_col):
            x = one_npy_data[:, :, npy_colum]
            y = (x - x.mean())
            #y = (x - x.mean()) / (x.std() + 1e-8)
            one_npy_data[:, :, npy_colum] = y

        # 강수량 값
        feature = one_npy_data[:, :, :my_col]
        target = one_npy_data[:, :, -1].reshape(40, 40, 1)

        yield (feature, target)


def testGenerator():
    test_path = './test/test'
    test_files = sorted(glob.glob(test_path + '/*'))

    X_test = []

    for file in tqdm(test_files, desc='test'):
        data = np.load(file)

        # 평균 뺀게 성능이 좋음
        for npy_colum in range(my_col):
            x = data[:, :, npy_colum]
            y = (x - x.mean())
            #y = (x - x.mean()) / (x.std() + 1e-8)
            data[:, :, npy_colum] = y

        X_test.append(data[:, :, :my_col])

    return np.array(X_test)