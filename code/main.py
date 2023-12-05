


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
        "../data/3.png")

    # Preprocess the image
    input_data = np.expand_dims(image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output depth map
    depth_map = interpreter.get_tensor(output_details[0]['index'])[0, :, :, 0]

    return depth_map


def transmittanceMap(img):
    min_channel = np.min(img, axis=2)
    window_size = 15
    dark_channel = cv2.erode(min_channel, np.ones((window_size, window_size)))

    num_pixels = dark_channel.size
    num_brightest = int(0.001 * num_pixels)
    indices = np.argpartition(dark_channel.flatten(), -num_brightest)[-num_brightest:]

    atmospheric_light = np.max(img.reshape(-1, 3)[indices], axis=0)
    omega = 0.95
    t0 = 0.1
    min_channel2 = np.min(img / atmospheric_light, axis=2)
    dark_channel2 = cv2.erode(min_channel2, np.ones((window_size, window_size)))
    transmission = 1 - omega * dark_channel2

    # Clamp the values to be between t0 and 1
    transmission = np.maximum(transmission, t0)
    transmissionMap =  (transmission * 255).astype(np.uint8)
    return transmissionMap


def volumetricMap():
    print("not implemented yet")


def main():
    file = "1.png"
    img = cv2.imread(f"../data/{file}", )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    reflectionMap = img
    tMap = transmittanceMap(img)

    # Display the results
    # cv2.imshow('Original Image', cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    # cv2.imshow('Transmission Map', tMap)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.imshow(tMap, cmap='gray')
    plt.show()

    # depth_map = geometryEstimation()

    # plt.imshow(depth_map, cmap='plasma')
    # # plt.colorbar()
    # plt.show()


main()