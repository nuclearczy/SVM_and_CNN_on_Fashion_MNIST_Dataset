import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plot
import numpy as np
from utils.mnist_reader import *
from include.pca_reduction import *
from include.lda_reduction import *


def dimension_visualize_2d():
    train_image_dim, train_label_dim = load_mnist('data/fashion', kind='train')
    test_image_dim, test_label_dim = load_mnist('data/fashion', kind='t10k')
    pca_train_2d, pca_test_2d = pca_reduction(train_image_dim, test_image_dim, pca_target_dim=2)
    lda_train_2d, lda_test_2d = lda_reduction(train_image_dim, train_label_dim, test_image_dim, components_number=2)

    pca_train_2d_arr = np.array(pca_train_2d)
    pca_test_2d_arr = np.array(pca_test_2d)
    lda_train_2d_arr = np.array(lda_train_2d)
    lda_test_2d_arr = np.array(lda_test_2d)

    pca_train_x = pca_train_2d_arr[:, 0]
    pca_train_y = pca_train_2d_arr[:, 1]
    pca_test_x = pca_test_2d_arr[:, 0]
    pca_test_y = pca_test_2d_arr[:, 1]
    lda_train_x = lda_train_2d_arr[:, 0]
    lda_train_y = lda_train_2d_arr[:, 1]
    lda_test_x = lda_test_2d_arr[:, 0]
    lda_test_y = lda_test_2d_arr[:, 1]

    plt.title('PCA training set 2D Visualization')
    plt.scatter(pca_train_x, pca_train_y, c=train_label_dim, alpha=0.7)
    plt.savefig('./visualization/PCA_train_2D.png')
    plt.show()
    plt.clf()
    plt.close()

    plt.title('PCA testing set 2D Visualization')
    plt.scatter(pca_test_x, pca_test_y, c=test_label_dim, alpha=0.7)
    plt.savefig('./visualization/PCA_test_2D.png')
    plt.show()
    plt.clf()
    plt.close()

    plt.title('LDA training set 2D Visualization')
    plt.scatter(lda_train_x, lda_train_y, c=train_label_dim, alpha=0.7)
    plt.savefig('./visualization/LDA_train_2D.png')
    plt.show()
    plt.clf()
    plt.close()

    plt.title('LDA testing set 2D Visualization')
    plt.scatter(lda_test_x, lda_test_y, c=test_label_dim, alpha=0.7)
    plt.savefig('./visualization/LDA_test_2D.png')
    plt.show()
    plt.clf()
    plt.close()


def dimension_visualize_3d():
    train_image_3dim, train_label_3dim = load_mnist('data/fashion', kind='train')
    test_image_3dim, test_label_3dim = load_mnist('data/fashion', kind='t10k')
    pca_train_3d, pca_test_3d = pca_reduction(train_image_3dim, test_image_3dim, pca_target_dim=3)
    lda_train_3d, lda_test_3d = lda_reduction(train_image_3dim, train_label_3dim, test_image_3dim, components_number=3)

    pca_train_3d_arr = np.array(pca_train_3d)
    pca_test_3d_arr = np.array(pca_test_3d)
    lda_train_3d_arr = np.array(lda_train_3d)
    lda_test_3d_arr = np.array(lda_test_3d)

    pca_train_x = pca_train_3d_arr[:, 0]
    pca_train_y = pca_train_3d_arr[:, 1]
    pca_train_z = pca_train_3d_arr[:, 2]
    pca_test_x = pca_test_3d_arr[:, 0]
    pca_test_y = pca_test_3d_arr[:, 1]
    pca_test_z = pca_test_3d_arr[:, 2]
    lda_train_x = lda_train_3d_arr[:, 0]
    lda_train_y = lda_train_3d_arr[:, 1]
    lda_train_z = lda_train_3d_arr[:, 2]
    lda_test_x = lda_test_3d_arr[:, 0]
    lda_test_y = lda_test_3d_arr[:, 1]
    lda_test_z = lda_test_3d_arr[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('PCA training set 3D Visualization')
    ax.scatter(pca_train_x, pca_train_y, pca_train_z, c=train_label_3dim)
    plt.savefig('./visualization/PCA_train_3D.png')
    plt.show()
    plt.clf()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('PCA testing set 3D Visualization')
    ax.scatter(pca_test_x, pca_test_y, pca_test_z, c=test_label_3dim)
    plt.savefig('./visualization/PCA_test_3D.png')
    plt.show()
    plt.clf()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('LDA training set 3D Visualization')
    ax.scatter(lda_train_x, lda_train_y, lda_train_z, c=train_label_3dim)
    plt.savefig('./visualization/LDA_train_3D.png')
    plt.show()
    plt.clf()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('LDA testing set 3D Visualization')
    ax.scatter(lda_test_x, lda_test_y, lda_test_z, c=test_label_3dim)
    plt.savefig('./visualization/LDA_test_3D.png')
    plt.show()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    dimension_visualize_2d()
    dimension_visualize_3d()
