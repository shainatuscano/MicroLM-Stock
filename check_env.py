#!/usr/bin/env python3
import sys

print("Python:", sys.version.replace("\n", " "))
try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
    devices = tf.config.list_physical_devices()
    print("Physical devices:", [f"{d.device_type}:{d.name}" for d in devices])
    gpus = tf.config.list_physical_devices("GPU")
    print("GPU available:", len(gpus) > 0)
except Exception as e:
    print("TensorFlow error:", e)

try:
    import keras
    print("Keras:", keras.__version__)
except Exception as e:
    print("Keras error:", e)

try:
    import tokenizers
    print("tokenizers OK:", tokenizers.__version__)
except Exception as e:
    print("tokenizers error:", e)

try:
    import datasets
    print("datasets OK:", datasets.__version__)
except Exception as e:
    print("datasets error:", e)

try:
    import ftfy
    print("ftfy OK:", ftfy.__version__)
except Exception as e:
    print("ftfy error:", e)

try:
    import flask
    import importlib.metadata
    print("flask OK:", importlib.metadata.version("flask"))
except Exception as e:
    print("flask error:", e)
