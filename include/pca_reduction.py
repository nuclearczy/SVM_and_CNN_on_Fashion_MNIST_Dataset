from sklearn.decomposition import PCA
import time


def pca_reduction(input_train, input_test, pca_target_dim=30):
    start_time = time.time()
    print("\nPCA in progress >>> ")
    if pca_target_dim:
        pca = PCA(n_components=pca_target_dim)
        print("PCA target dimension chosen as: ", pca.n_components)
    else:
        pca = PCA()
        print("PCA target dimension selected as auto")
    pca.fit(input_train)
    pca_train = pca.transform(input_train)
    pca_test = pca.transform(input_test)
    end_time = time.time()
    print("PCA time : ", end_time - start_time, " seconds. ")
    print(">>> Done PCA\n")
    return pca_train, pca_test
