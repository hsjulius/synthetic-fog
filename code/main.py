

import argparse
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms as T

"""
We define the input photograph, the radiance attenuation
ratio map and the volumetric scattering radiance map as the
Ref ection Map, the Transmittance Map and the Volumetric Map respectively

input photograph = reflection map 
radiance attenuation ratio map = transmittance map 
volumetric scattering radiance map = volumetric map 
"""

def illuminationEstimation(image):
    # Your illumination estimation logic here
    # Example: Gray World Assumption
    avg_color = np.mean(image, axis=(0, 1))
    illumination = avg_color / np.mean(avg_color)
    return illumination

def compute_surface_normals(image):
    # Your surface normals computation logic here
    # Example: Sobel operator for gradient estimation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    normals_x = gradient_x / magnitude
    normals_y = gradient_y / magnitude
    normals_z = np.ones_like(magnitude)
    return normals_x, normals_y, normals_z
    # # Convert the image to floating point representation
    # img_float = image.astype(np.float32)

    # # Compute the average color in each channel
    # avg_r = np.mean(img_float[:, :, 2])
    # avg_g = np.mean(img_float[:, :, 1])
    # avg_b = np.mean(img_float[:, :, 0])

    # # Compute the overall average color
    # avg_color = (avg_r + avg_g + avg_b) / 3.0

    # # Scale each channel to make the overall average color gray
    # img_gray_world = img_float * (avg_color / np.array([avg_r, avg_g, avg_b]))

    # # Clip the values to be in the valid range [0, 255]
    # img_gray_world = np.clip(img_gray_world, 0, 255).astype(np.uint8)
    # # plt.imshow(img_gray_world)
    # # plt.show()

    # return img_gray_world


def geometryEstimation():
    midas = torch.hub.load("isl-org/MiDaS", "MiDaS")

    # Load and preprocess the input image using OpenCV
    image_path = "../data/3.png"
    input_image = cv2.imread(image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    transform = T.Compose([T.Resize(384), T.ToTensor(), T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_batch = transform(input_image).unsqueeze(0)

    # Perform depth estimation
    with torch.no_grad():
        prediction = midas(input_batch)

    # Retrieve the depth map
    depth_map = prediction.squeeze().cpu().numpy()
    return depth_map


def transmittanceMap(img):
    min_channel = np.min(img, axis=2)
    window_size = 15
    dark_channel = cv2.erode(min_channel, np.ones((window_size, window_size)))

    num_pixels = dark_channel.size
    num_brightest = int(0.001 * num_pixels)
    indices = np.argpartition(
        dark_channel.flatten(), -num_brightest)[-num_brightest:]

    atmospheric_light = np.max(img.reshape(-1, 3)[indices], axis=0)
    omega = 0.1
    t0 = 0.9
    min_channel2 = np.min(img / atmospheric_light, axis=2)
    dark_channel2 = cv2.erode(
        min_channel2, np.ones((window_size, window_size)))
    transmission = 1 - omega * dark_channel2

    # Clamp the values to be between t0 and 1
    transmission = np.maximum(transmission, t0)
    transmissionMap = (transmission * 255).astype(np.uint8)
    return transmissionMap

def volumetricMap():
    print("not implemented yet")


def main():
    file = "3.png"
    img = cv2.imread(f"../data/{file}", )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    reflectionMap = img
    tMap = transmittanceMap(img)
    # illumination = illuminationEstimation(img)
    illumination = illuminationEstimation(img)

    # Step 2: Compute Surface Normals
    # normals_x, normals_y, normals_z = compute_surface_normals(img)
    # print(f"shape {illumination.shape} {img.shape}")

    # # Step 3: Compute Albedo Map
    # albedo_map = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / illumination


    # Step 4: Compute Transmittance Map
    # tMap = np.sqrt(normals_x**2 + normals_y**2 + normals_z**2) * albedo_map

    fig, axs = plt.subplots(1, 4, figsize=(10, 5))
    axs[0].imshow(reflectionMap)
    axs[0].set_title('Reflection')

    axs[1].imshow(tMap,cmap='gray')
    axs[1].set_title('Transmittance')
    axs[2].imshow(tMap)
    axs[2].set_title('Illumination')

    axs[3].imshow(img)
    axs[3].set_title('Img')
    plt.show()

    # Display the results
    # cv2.imshow('Original Image', cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    # cv2.imshow('Transmission Map', tMap)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # plt.imshow(tMap, cmap='gray')
    # plt.imshow(illumination, cmap='gray')
    # plt.show()

    # depth_map = geometryEstimation()
    # plt.imshow(depth_map, cmap="plasma")
    # plt.colorbar()
    # plt.show()
    # print(torch.hub.list("isl-org/MiDaS"))

    # depth_map = geometryEstimation()

    # plt.imshow(depth_map, cmap='plasma')
    # plt.colorbar()
    # plt.show()


main()
