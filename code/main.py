import argparse
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

"""

We define the input photograph, the radiance attenuation
ratio map and the volumetric scattering radiance map as the
Ref ection Map, the Transmittance Map and the Volumetric Map respectively

input photograph = reflection map 
radiance attenuation ratio map = transmittance map 
volumetric scattering radiance map = volumetric map 

"""


def illuminationEstimation():
    print("not implemented yet")


def geometryEstimation():
    interpreter = tf.lite.Interpreter(
        model_path="/Users/Elizabeth/Desktop/cs1290/MiDaS/midas/model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = cv2.imread(
        "/Users/Elizabeth/Desktop/cs1290/synthetic-fog/data/3.png")

    # Preprocess the image
    input_data = np.expand_dims(image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output depth map
    depth_map = interpreter.get_tensor(output_details[0]['index'])[0, :, :, 0]

    return depth_map


def transmittanceMap():
    print("not implemented yet")


def volumetricMap():
    print("not implemented yet")


def main():
    # file = "1.png"
    # img = cv2.imread(f"../data/{file}", )
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # reflectionMap = img

    # plt.imshow(img)
    # plt.show()

    depth_map = geometryEstimation()

    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar()
    plt.show()


main()
