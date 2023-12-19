
import argparse
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms as T
from PIL import Image


"""
We define the input photograph, the radiance attenuation
ratio map and the volumetric scattering radiance map as the
Ref ection Map, the Transmittance Map and the Volumetric Map respectively

input photograph = reflection map 
radiance attenuation ratio map = transmittance map 
volumetric scattering radiance map = volumetric map 
"""


def depthMapMidas(image_name):
    midas = torch.hub.load("isl-org/MiDaS", "MiDaS")

    midas.eval()

    image = f"../data/{image_name}"
    input_image = Image.open(image).convert("RGB")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(input_image).unsqueeze(0)

    if input_tensor.shape[2] != 384 or input_tensor.shape[3] != 384:
        input_tensor = torch.nn.functional.interpolate(
            input_tensor, size=(384, 384), mode='bilinear', align_corners=False)

    with torch.no_grad():
        prediction = midas(input_tensor)

    depth_map_resized = torch.nn.functional.interpolate(
        prediction.unsqueeze(1), size=(input_image.size[1], input_image.size[0]), mode='bilinear', align_corners=False
    ).squeeze().cpu().numpy()
    return depth_map_resized


def coordToIdx(x, y, width):
    return y * width + x


def transmittanceMap(img, depth_map):
    light_idx = -.6
    alpha = 0.1
    delta = 0.7
    tmap = np.zeros((img.shape[0], img.shape[1]))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    width, height = img.shape

    for j in range(height):
        for i in range(width):
            surface_pt = depth_map[i][j]
            const = np.exp(-((alpha + delta) * light_idx **
                           2 - (alpha + delta) * surface_pt ** 2))

            img_val = img[i][j] / 255

            val2 = np.exp(-((alpha+delta) * img_val ** 2 -
                          (alpha + delta) * surface_pt ** 2))

            val = const * (val2)

            tmap[i][j] = const / 2.5

    tmap = np.clip(tmap, 0, 1)
    return tmap


def volumetricMapWrong(img, depth_map):
    '''
        Steals transmittance map solution to create a volume map
    '''
    light_idx = -.6
    alpha = 0.1
    delta = 0.7
    vmap = np.zeros((img.shape[0], img.shape[1]))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    width, height = img.shape

    for j in range(height):
        for i in range(width):
            surface_pt = depth_map[i][j]
            const = np.exp(-((alpha + delta) * light_idx **
                           2 - (alpha + delta) * surface_pt ** 2))

            img_val = coordToIdx(i / width, j / height,
                                 width) / width

            val2 = np.exp(-((alpha+delta) * img_val ** 2 -
                          (alpha + delta) * surface_pt ** 2))

            val = const * (val2)

            vmap[i][j] = val2

    vmap = np.clip(vmap, 0, 1)
    return vmap


def estimate_illumination(h, w, sun_intensity=0.8, sky_intensity=0.3):
    '''
        Implements illumination map with light coming top down
    '''
    sun_component = np.zeros((h, w))
    sky_component = np.zeros((h, w))

    # brdf
    rho = 1 / np.pi

    cos_theta_sun = np.cos(np.radians(70))

    # light to dark gradient from top to bottom
    gradient = np.linspace(sun_intensity * cos_theta_sun, 0, h)

    # apply the gradient to each column of the sun component
    for j in range(w):
        sun_component[:, j] = gradient * rho

    # assuming uniform sky illumination
    sky_component[:, :] = sky_intensity * rho

    total_illumination = sun_component + sky_component

    # normalize the illumination values to be between 0 and 1
    min_illum = total_illumination.min()
    max_illum = total_illumination.max()
    normalized_illumination = (
        total_illumination - min_illum) / (max_illum - min_illum)
    return normalized_illumination

    '''
        Implementation of illumination map with light coming from top left
        corner
    '''
    # illumination_map = np.full((h, w), sky_intensity)

    # sun_direction = np.array([1, 1])  # Sun starts top left
    # sun_direction = sun_direction / np.linalg.norm(sun_direction)

    # for i in range(h):
    #     for j in range(w):
    #         distance_from_sun = np.dot(
    #             np.array([i, j]) / np.array([h, w]), sun_direction)
    #         illumination_map[i, j] += sun_intensity * (1 - distance_from_sun)

    # illumination_map = cv2.GaussianBlur(
    #     illumination_map, (0, 0), sigmaX=15, sigmaY=15)
    # illumination_map = cv2.normalize(
    #     illumination_map, None, 0, 1, cv2.NORM_MINMAX)
    # return illumination_map


def depthMap(img_classes, darkness_factor=0.75, gradient_exp=2):
    '''
        Manual depth map based on pixel classifications
    '''
    h, w, _ = img_classes.shape
    depth = np.ones((h, w)) * np.nan

    depth[img_classes[:, :, 0] == 255] = 0.3  # buildings
    depth[img_classes[:, :, 2] == 255] = 0  # sky

    ground_pixels = np.all(img_classes == [0, 255, 0], axis=-1)
    for i in range(h):
        # ground — gets brighter as it gets nearer
        depth[i, ground_pixels[i, :]] = (
            (i / h)**gradient_exp) * darkness_factor

    # deal with edges
    mask = np.isnan(depth)
    depth[mask] = 0.3
    depth_blurred = cv2.GaussianBlur(depth, (5, 5), 0)
    depth[mask] = depth_blurred[mask]

    depth = (depth - depth.min()) / (depth.max() - depth.min())

    return depth


def volumetricMap(illumination_map, depth_map):
    illumination_map = illumination_map.astype(np.float64)
    depth_map = depth_map.astype(np.float64) / np.max(depth_map)

    volume_map = np.zeros_like(illumination_map)

    scattering = 0.7
    absorption = 0.1

    for i in range(illumination_map.shape[0]):
        for j in range(illumination_map.shape[1]):
            # accumulate radiance along path
            acc_radiance = 0.0
            for step in range(i, illumination_map.shape[0]):
                # distance btwn points based on depth map
                # ensure no neg distances
                distance = max(depth_map[step, j] - depth_map[i, j], 0.01)
                optical_len = (absorption + scattering) * distance
                transmittance = np.exp(-optical_len)
                acc_radiance += scattering * \
                    transmittance * illumination_map[step, j]
            volume_map[i, j] = acc_radiance

    volume_map = (volume_map - np.min(volume_map)) / \
        (np.max(volume_map) - np.min(volume_map))

    return volume_map


def main():
    file = "bruh.png"
    img = cv2.imread(f"../data/{file}", )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_class = cv2.imread(f"../data/bruh_class.png", )
    img_class = cv2.cvtColor(img_class, cv2.COLOR_BGR2RGB)

    depth_map = depthMap(img_class)
    # depth_map = depthMapMidas(file)
    # depth_map /= np.max(depth_map)

    reflectionMap = img

    height, width = depth_map.shape
    illumination_map = np.zeros((height, width))
    illumination_map = estimate_illumination(height, width)

    v = volumetricMap(illumination_map, depth_map)
    # vMap = volumetricMapWrong(img, depth_map)

    tMap = transmittanceMap(img, depth_map)

    tmap3d = tMap[:, :, np.newaxis]
    vmap3d = v[:, :, np.newaxis]
    reflectionMap3d = reflectionMap / 255

    out = (reflectionMap3d * tmap3d) + vmap3d
    out = np.clip(out, 0, 1)
    out = out.astype(np.float32)
    output = cv2.cvtColor(out, cv2.COLOR_RGB2BGR) * 255

    cv2.imwrite("../results/out.png", output)

    fig, axs = plt.subplots(1, 5, figsize=(10, 5))

    axs[0].imshow(reflectionMap)
    axs[0].set_title('Reflection')

    axs[1].imshow(tMap, cmap='gray')
    axs[1].set_title('Transmittance')

    axs[2].imshow(depth_map, cmap='gray')
    axs[2].set_title('depth')

    axs[3].imshow(v, cmap='gray')
    axs[3].set_title('vmap??')

    axs[4].imshow(out, cmap='gray')
    axs[4].set_title('combo??')
    plt.show()


main()
