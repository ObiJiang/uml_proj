import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

class Generator_minst(object):

    def __init__(self, fea=200):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train_f = x_train.reshape(-1, 28*28).astype(float)/255
        x_test_f = x_test.reshape(-1, 28*28).astype(float)/255

        pca = PCA(n_components=fea)
        x_train_pca = pca.fit_transform(x_train_f)
        print(np.sum(pca.explained_variance_ratio_))

        idx_train = [None]*10
        idx_test = [None]*10
        for i in range(10):
            idx_train[i] = np.where(y_train == i)
            idx_test[i] = np.where(y_test == i)

        self.x_train10 = [None]*10
        self.x_test10 = [None]*10
        for i in range(10):
            self.x_train10[i] = x_train_pca[idx_train[i][0], :]
            self.x_test10[i] = x_test_f[idx_test[i][0], :]

    def generate(self, size=100):
        first = np.random.randint(10)
        second = np.random.randint(10)
        while first == second:
            second = np.random.randint(10)
        
        x_train_first = self.x_train10[first]
        x_train_second = self.x_train10[second]
        x_test_first = self.x_test10[first]
        x_test_second = self.x_test10[second]

        idx_first = np.arange(len(x_train_first))
        idx_second = np.arange(len(x_train_second))
        np.random.shuffle(idx_first)
        np.random.shuffle(idx_second)

        x_train2 = np.concatenate((x_train_first[idx_first[:size]], x_train_second[idx_second[:size]]), axis=0)
        y_train2 = np.array([0]*size+[1]*size, dtype=np.float32)
        idx = np.arange(size*2)
        np.random.shuffle(idx)

        x_test2 = np.concatenate((x_test_first, x_test_second), axis=0)
        y_test2 = np.array([0]*len(x_test_first)+[1]*len(x_test_second), dtype=np.float32)

        return (x_train2[idx], y_train2[idx]), (x_test2, y_test2)

if __name__ == '__main__':
    generator = Generator_minst()
