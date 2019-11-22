from sklearn import svm
import time
import utils.mnist_reader
import include.pca_reduction
import include.lda_reduction
import include.test_accuracy

train_image, train_label = utils.mnist_reader.load_mnist('data/fashion', kind='train')
test_image, test_label = utils.mnist_reader.load_mnist('data/fashion', kind='t10k')
pca_train, pca_test = include.pca_reduction.pca_reduction(train_image, test_image)
lda_train, lda_test = include.lda_reduction.lda_reduction(train_image, train_label, test_image)

start_time = time.time()
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
end_time = time.time()
print("SVM rbf time : ", end_time - start_time, " seconds. ")
print(">>> Done SVM rbf\n")
