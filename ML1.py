import pylab
from sklearn import datasets
from sklearn.datasets import load_iris

digit_dataset = datasets.load_digits()

# print(iris_dataset.keys())
# print(digit_dataset.DESCR)
# print(digit_dataset.target[2])
# pylab.imshow(digit_dataset.images[0], cmap=pylab.cm.gray_r)
# pylab.show()
iris_dataset = load_iris()

print(iris_dataset.data.shape)