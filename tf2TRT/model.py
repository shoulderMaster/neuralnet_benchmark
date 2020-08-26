import tensorflow as tf
import time
import numpy as np


EPOCHS = 5

x_train = np.random.rand(100,56,56,32)
y_train = np.random.rand(100,56,56,192)

x_train = x_train.astype(np.float32) / 255.0
y_train = y_train.astype(np.float32) / 255.0

# model_conv2d
conv2d = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(192, (3,3), padding='SAME')
  ])

conv2d.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
conv2d.fit(x_train, y_train, epochs=EPOCHS)

tf.saved_model(conv2d, "./saved_model/")

#model_pw_dw
#   pwdw = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(192, (1,1), padding='SAME'),
#     tf.keras.layers.DepthwiseConv2D((3,3), padding='SAME', depth_multiplier=1)
#     ])

#   pwdw.compile(optimizer='adam',
#                 loss='mean_squared_error',
#                 metrics=['accuracy'])
#   pwdw.fit(x_train, y_train, epochs=EPOCHS)

#   make_SavedModel(pwdw)
