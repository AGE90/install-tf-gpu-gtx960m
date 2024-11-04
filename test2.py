# Save as gpu_test.py
import os
# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import time

# Print system info
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("GPU Device Name:", tf.test.gpu_device_name())

# Create large matrices
size = 2000
a = tf.random.normal([size, size])
b = tf.random.normal([size, size])

# Time CPU computation
with tf.device('/CPU:0'):
    start = time.time()
    c = tf.matmul(a, b)
    cpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.2f} seconds")

# Time GPU computation
with tf.device('/GPU:0'):
    start = time.time()
    c = tf.matmul(a, b)
    gpu_time = time.time() - start
    print(f"GPU time: {gpu_time:.2f} seconds")

print(f"GPU speedup: {cpu_time/gpu_time:.2f}x")