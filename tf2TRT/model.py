import tensorflow as tf
import time
import numpy as np
import os

def save_model(name, model) :
    path = "./saved_model/%s" % name
    if not os.path.isdir(path) :
        os.makedirs(path)
    tf.saved_model.save(model, path)


def conv2d_model(name) :
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
    save_model(name, conv2d)

def depthpointwise_model(name) :
    EPOCHS = 5
    x_train = np.random.rand(100,56,56,32)
    y_train = np.random.rand(100,56,56,192)
    x_train = x_train.astype(np.float32) / 255.0
    y_train = y_train.astype(np.float32) / 255.0

    # model_pw_dw
    pwdw = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(192, (1,1), padding='SAME'),
      tf.keras.layers.DepthwiseConv2D((3,3), padding='SAME', depth_multiplier=1)
      ])

    pwdw.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    pwdw.fit(x_train, y_train, epochs=EPOCHS)
    save_model(name, pwdw)

if __name__ == "__main__" :
    conv2d_model("conv2d")
    depthpointwise_model("pwdw")
