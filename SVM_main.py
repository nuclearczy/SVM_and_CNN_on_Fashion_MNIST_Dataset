from sklearn import svm
import time
import utils.mnist_reader
import include.pca_reduction
import include.lda_reduction
import include.test_accuracy


def main():
    train_image, train_label = utils.mnist_reader.load_mnist('data/fashion', kind='train')
    test_image, test_label = utils.mnist_reader.load_mnist('data/fashion', kind='t10k')
    pca_train, pca_test = include.pca_reduction.pca_reduction(train_image, test_image)
    lda_train, lda_test = include.lda_reduction.lda_reduction(train_image, train_label, test_image)

    start_time = time.time()
    print("\nSVM poly in progress >>> ")
    svm_clf = svm.SVC(kernel='poly', gamma='scale')
    svm_clf.fit(pca_train, train_label)
    poly_predict = svm_clf.predict(pca_test)
    poly_accuracy_pca = include.test_accuracy.test_accuracy(poly_predict, test_label)
    print("SVM poly accuracy on pca:", poly_accuracy_pca)

    svm_clf.fit(lda_train, train_label)
    poly_predict = svm_clf.predict(lda_test)
    poly_accuracy_lda = include.test_accuracy.test_accuracy(poly_predict, test_label)
    print("SVM poly accuracy on lda:", poly_accuracy_lda)
    end_time = time.time()
    print("SVM poly time : ", end_time - start_time, " seconds. ")
    print(">>> Done SVM poly\n")

    start_time_poly = time.time()
    print("\nSVM rbf in progress >>> ")
    svm_clf = svm.SVC(kernel='rbf', gamma='scale')
    svm_clf.fit(pca_train, train_label)
    rbf_predict = svm_clf.predict(pca_test)
    rbf_accuracy_pca = include.test_accuracy.test_accuracy(rbf_predict, test_label)
    print("SVM rbf accuracy on pca:", rbf_accuracy_pca)

    svm_clf.fit(lda_train, train_label)
    rbf_predict = svm_clf.predict(lda_test)
    rbf_accuracy_lda = include.test_accuracy.test_accuracy(rbf_predict, test_label)
    print("SVM rbf accuracy on lda:", rbf_accuracy_lda)
    end_time_poly = time.time()
    print("SVM rbf time : ", end_time_poly - start_time_poly, " seconds. ")
    print(">>> Done SVM rbf\n")

    print("\n\n-------------------------\nOn training set: \n-------------------------")
    start_time = time.time()
    print("\nSVM poly in progress >>> ")
    svm_clf = svm.SVC(kernel='poly', gamma='scale')
    svm_clf.fit(pca_train, train_label)
    poly_predict = svm_clf.predict(pca_train)
    poly_accuracy_pca = include.test_accuracy.test_accuracy(poly_predict, train_label)
    print("SVM poly accuracy on pca:", poly_accuracy_pca)

    svm_clf.fit(lda_train, train_label)
    poly_predict = svm_clf.predict(lda_train)
    poly_accuracy_lda = include.test_accuracy.test_accuracy(poly_predict, train_label)
    print("SVM poly accuracy on lda:", poly_accuracy_lda)
    end_time = time.time()
    print("SVM poly time : ", end_time - start_time, " seconds. ")
    print(">>> Done SVM poly\n")

    start_time_poly = time.time()
    print("\nSVM rbf in progress >>> ")
    svm_clf = svm.SVC(kernel='rbf', gamma='scale')
    svm_clf.fit(pca_train, train_label)
    rbf_predict = svm_clf.predict(pca_train)
    rbf_accuracy_pca = include.test_accuracy.test_accuracy(rbf_predict, train_label)
    print("SVM rbf accuracy on pca:", rbf_accuracy_pca)

    svm_clf.fit(lda_train, train_label)
    rbf_predict = svm_clf.predict(lda_train)
    rbf_accuracy_lda = include.test_accuracy.test_accuracy(rbf_predict, train_label)
    print("SVM rbf accuracy on lda:", rbf_accuracy_lda)
    end_time_poly = time.time()
    print("SVM rbf time : ", end_time_poly - start_time_poly, " seconds. ")
    print(">>> Done SVM rbf\n")


if __name__ == '__main__':
    main()
