from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

mnist_raw=loadmat("mnist-original.mat")
mnist={
    "data":mnist_raw["data"].T,
    "target":mnist_raw["label"][0]
}
x,y=mnist["data"],mnist["target"]
# print(mnist["data"].shape)
# print(mnist["target"].shape)

x_train, x_test,y_train,y_test= x[:60000],x[60000:],y[:60000],y[60000:]

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

predict_number = 5000
y_test_0 = (y_test==0)
y_train_0 = (y_train==0)
print(y_train_0.shape,y_train_0)
print(y_test_0.shape,y_test_0)