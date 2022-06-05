import pylab
from sklearn import datasets
from sklearn.datasets import load_iris
from scipy.io import loadmat
import matplotlib.pyplot as plt
digit_dataset = datasets.load_digits()

# print(iris_dataset.keys())
# print(digit_dataset.DESCR)
# print(digit_dataset.target[2])
# pylab.imshow(digit_dataset.images[0], cmap=pylab.cm.gray_r)
# pylab.show()
iris_dataset = load_iris()

print(iris_dataset.data.shape)

mnist_raw=loadmat("mnist-original.mat")
print(mnist_raw)
mnist={
    "data":mnist_raw["data"].T,
    "target":mnist_raw["label"][0]
}
x,y=mnist["data"], mnist["target"]
number=x[15000]
number_image=number.reshape(28,28)

plt.imshow(
        number_image,
        cmap=plt.cm.binary,
        interpolation="nearest")
plt.show()
print(x.shape)