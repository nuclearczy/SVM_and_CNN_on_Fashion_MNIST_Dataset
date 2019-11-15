from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import time


def lda_reduction(input_train, input_train_label, input_test, components_number=None):
    start_time = time.time()
    print("\nLDA in progress >>> ")
    lda = LDA(n_components=components_number)
    lda.fit(input_train, input_train_label)
    lda_train = lda.transform(input_train)
    lda_test = lda.transform(input_test)
    end_time = time.time()
    print("LDA time : ", end_time - start_time, " seconds. ")
    print(">>> Done LDA\n")
    return lda_train, lda_test
