# Create a test script (save as basic_gpu_test.py)
import tensorflow as tf

print('Is GPU available:', tf.config.list_physical_devices('GPU'))
print('Is built with CUDA:', tf.test.is_built_with_cuda())

# Simple matrix multiplication test
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print('Matrix multiplication result:', c)