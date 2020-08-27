import tensorflow as tf
import time
import numpy as np
import os

def list_saved_model() :
    path = "./saved_model/"
    pb_list = os.listdir(path)
    print(pb_list)

if __name__ == "__main__" :
    list_saved_model()
