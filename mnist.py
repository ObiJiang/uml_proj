import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Generator_minst(object):

    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train_f = x_train.reshape(-1, 28*28).astype(float)/255
        x_test_f = x_test.reshape(-1, 28*28).astype(float)/255

        idx_train = [None]*10
        idx_test = [None]*10
        for i in range(10):
            idx_train[i] = np.where(y_train == i)
            idx_test[i] = np.where(y_test == i)

        self.x_train10 = [None]*10
        self.x_test10 = [None]*10
        for i in range(10):
            self.x_train10[i] = x_train_f[idx_train[i][0], :]
            self.x_test10[i] = x_test_f[idx_test[i][0], :]

    def generate(self, size=100, fea=200, k=2):
        pool1 = [4, 7, 9]
        pool2 = [6, 0]
        if k>2:
            pool2.append(3)
        first = pool1[np.random.randint(len(pool1))]
        second = pool2[np.random.randint(len(pool2))]
        
        x_train_1 = self.x_train10[first]
        x_train_2 = self.x_train10[second]
        x_train_3 = self.x_train10[3]

        id1 = np.arange(len(x_train_1))
        id2 = np.arange(len(x_train_2))
        id3 = np.arange(len(x_train_3))
        np.random.shuffle(id1)
        np.random.shuffle(id2)
        np.random.shuffle(id3)

        if k == 2:
            x_train2 = np.concatenate((x_train_1[id1[:size]], x_train_2[id2[:size]]), axis=0)
            y_train2 = np.array([0]*size+[1]*size, dtype=np.float32)
        else:
            x_train2 = np.concatenate((x_train_1[id1[:size]], x_train_2[id2[:size]], x_train_3[id3[:size]]), axis=0)
            y_train2 = np.array([0]*size+[1]*size+[2]*size, dtype=np.float32)
        idx = np.arange(size*k)
        np.random.shuffle(idx)

        x_norm = StandardScaler().fit_transform(x_train2)
        pca = PCA(n_components=fea, whiten=True)
        x_train_pca = pca.fit_transform(x_norm)

        return StandardScaler().fit_transform(x_train_pca[idx]), y_train2[idx]

    def generate_test(self, size=100, fea=200):
        x_train_1 = self.x_train10[1]
        x_train_2 = self.x_train10[8]

        id1 = np.arange(len(x_train_1))
        id2 = np.arange(len(x_train_2))
        np.random.shuffle(id1)
        np.random.shuffle(id2)

        x_train2 = np.concatenate((x_train_1[id1[:size]], x_train_2[id2[:size]]), axis=0)
        y_train2 = np.array([0]*size+[1]*size, dtype=np.float32)
        idx = np.arange(size*2)
        np.random.shuffle(idx)

        x_norm = StandardScaler().fit_transform(x_train2)
        pca = PCA(n_components=fea, whiten=True)
        x_train_pca = pca.fit_transform(x_norm)

        return StandardScaler().fit_transform(x_train_pca[idx]), y_train2[idx]

if __name__ == '__main__':
    k = 2
    generator = Generator_minst()
    data, labels = generator.generate_test(50, 2)
    id0 = np.where(labels == 0)[0]
    id1 = np.where(labels == 1)[0]
    if k>2:
        id2 = np.where(labels == 2)[0]
    
    plt.figure()
    plt.scatter(data[id0, 0], data[id0, 1], c='r')
    plt.scatter(data[id1, 0], data[id1, 1], c='b')
    if k>2:
        plt.scatter(data[id2, 0], data[id2, 1], c='g')
    plt.show()
