import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def eduGenerate(fea=5, size=50):
    eduH = parse('edu-high.txt')
    eduL = parse('edu-low.txt')
    idH = np.arange(eduH.shape[0])
    idL = np.arange(eduL.shape[0])
    np.random.shuffle(idH)
    np.random.shuffle(idL)

    x_train = np.concatenate((eduH[idH[:size]], eduL[idL[:size]]), axis=0)
    y_train = np.array([0]*size+[1]*size, dtype=np.float32)
    idx = np.arange(size*2)
    np.random.shuffle(idx)

    pca = PCA(n_components=fea, whiten=True)
    x_train = pca.fit_transform(x_train)
    print(np.sum(pca.explained_variance_ratio_))

    return x_train[idx], y_train[idx]

def parse(fileName):
    edu = []
    with open(fileName) as fi:
        for line in fi.readlines():
            strs = line.split()
            edu.append([float(degree) for degree in strs])
    return np.array(edu)

if __name__ == '__main__':
    X, y = eduGenerate(2)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    print(np.sum(np.abs(y-kmeans.labels_)))

    id0 = np.where(y == 0)[0]
    id1 = np.where(y == 1)[0]

    plt.figure()
    plt.scatter(X[id0, 0], X[id0, 1], c='r')
    plt.scatter(X[id1, 0], X[id1, 1], c='b')
    plt.show()
