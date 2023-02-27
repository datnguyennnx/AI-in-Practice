import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
print('Load data from MNIST')
mnist = tf.keras.datasets.mnist

# Mapping data from TF
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Get digital number ( 0 -> 9 ) follow label MNIST
dig = np.array([1,3,5,7,9,11,13,15,17,19])
x = x_train[dig,:,:]
y = np.eye(10,10)
plt.subplot(121)
plt.imshow(x[0])
plt.subplot(122)
plt.imshow(x[1])
x = np.reshape(x, (-1,784))/255

def sigmoid(x): 
    return 1./(1.+np.exp(-x))

# Random weight before training
w = np.random.uniform(-0.1, 0.1, (10, 784))
# Transpose 10x784 to 10x10 weight for matrix multiply with x 
o = sigmoid(np.matmul(x, w.transpose())) 
print('output of first neuron with 10 digits ', o[:,0])
fig = plt.figure()
plt.bar([i for i, _ in enumerate(o)], o[:, 0])
plt.show()

#training process
n = 0.05 
num_epoch = 10
for epoch in range(num_epoch):
  o = sigmoid(np.matmul(x, w.transpose()))
  loss = np.power(0-y, 2).mean()
  # "@x" mean mathmul()
  dw = np.transpose((y-o)*o*(1-o))@x
  w = w + n*dw
  print(loss)

# Active function sigmoid
o = sigmoid(np.matmul(x, w.transpose()))
# Input number to o[:, number] to check model
print('output of the first neuron with 10 input digit ', o[:, 9])
fig = plt.figure()
# Plot accuracy follow number o[:, number]
plt.bar([i for i, _ in enumerate(o)], o[:,9])
plt.show()
