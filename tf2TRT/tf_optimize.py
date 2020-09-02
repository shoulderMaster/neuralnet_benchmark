
import os
import tensorflow as tf

def list_saved_model() :
    path = "./saved_model/"
    pb_list = os.listdir(path)
    output_path = "./optimized_model/"
    if not os.path.isdir(output_path) :
        os.makedirs(output_path)
    return [(path+model, output_path+model) for model in pb_list]


if __name__ == "__main__" :
    saved_model_list = list_saved_model()
    # conversion option
    for input_saved_model_dir, output_saved_model_dir in saved_model_list :
        with tf.Graph().as_default() :
            if not tf.executing_eagerly() :
                tf.compat.v1.enable_eager_execution()
            from tensorflow.python.compiler.tensorrt import trt_convert as trt
            print(input_saved_model_dir, tf.executing_eagerly())
            print(input_saved_model_dir, tf.executing_eagerly())
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
           #conversion_params = conversion_params._replace(
           #max_workspace_size_bytes=(1<<32))
            conversion_params = conversion_params._replace(precision_mode="FP16")
           #conversion_params = conversion_params._replace(
           #maximum_cached_engiens=100)
            print(input_saved_model_dir, output_saved_model_dir)
            converter = trt.TrtGraphConverterV2(
                    input_saved_model_dir=input_saved_model_dir,
                    conversion_params=conversion_params)
            converter.convert()
            converter.save(output_saved_model_dir)
            tf.compat.v1.disable_eager_execution()
