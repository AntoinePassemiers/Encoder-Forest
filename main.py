# -*- coding: utf-8 -*-
# main.py
# author : Antoine Passemiers

from encoder import EncoderForest


if __name__ == '__main__':

    from sklearn.datasets import fetch_openml, load_digits
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    mnist = fetch_openml('mnist_784')
    X, y = mnist.data, mnist.target
    X = X.reshape(X.shape[0], -1)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.25)

    encoder = EncoderForest(500)
    encoder.fit(X_train, max_depth=10)
    print("end fit")
    encoded = encoder.encode(X_train)
    print("end encode")
    

    f, axarr = plt.subplots(2,2)
    for i in range(2):
        img = X_train[i].reshape(28, 28)
        img_prime = encoder.decode(encoded[i]).reshape(28, 28)
        print("end decode")
        axarr[i, 0].imshow(img, cmap='gray')
        axarr[i, 1].imshow(img_prime, cmap='gray')
    plt.show()
