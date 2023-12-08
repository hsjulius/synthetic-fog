
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

def generate_illumination_map(image):
    print(f"shape of img {image.shape}")
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
    # print(f"shape of gradient_y {gradient_y.shape}")
    gradient_x = cv2.Sobel(grey_img, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(grey_img, cv2.CV_64F, 0, 1, ksize=3)
    # print(f"shape of gradient_y {gradient_y.shape}")
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    enhanced_image = cv2.equalizeHist(np.uint8(grey_img * 255)) / 255.0
    # print(f"shape of enhanced_image {enhanced_image.shape}")
    illumination_map = gradient_magnitude * enhanced_image
    # print(f"shape of illumination_map {illumination_map.shape}")
    # illumination_map = np.vstack([illumination_map,illumination_map, illumination_map])#cv2.cvtColor(illumination_map, cv2.COLOR_GRAY2BGR)# / 255.0
    # stacked_matrix = np.clip(np.stack([illumination_map] * 3, axis=-1) * 255, 0, 255)
    
    # stacked_matrix = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 
    # print(f"shape of illumination_map {stacked_matrix}")
    return illumination_map

def illuminationEstimation(image, outdoor=1):
    if outdoor: 
        s = 2
        p_p = 0.5
        theta_sun = 1.2
        # L_sky = s * (illumination_P / np.pi * p_p)
        # illumination_sun = L_sun_p * V_p_w_sun * np.cos(theta_sun)
        # illumination_sky = np.sum(L_sky_p * V_p_w_i * np.cos(theta_dwi))
        # illumination_P = illumination_sun + illumination_sky

        # illumination_P = np.pi * L_sky_p 
        illumination_P = L_sun * p_p * np.cos(theta_sun) + np.pi * L_sky * p_p

        avg_color = np.mean(image, axis=(0, 1))
        illumination = avg_color / np.mean(avg_color)
        return illumination
    else: 
        return 0

def compute_surface_normals(image):
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


def depthMap():
    # midas = torch.hub.load("isl-org/MiDaS", "MiDaS")

    # midas.eval()

    # image_path = "../data/3.png"
    # input_image = Image.open(image_path).convert("RGB")

    # transform = T.Compose([
    #     T.Resize(384),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # # print(f"size of img  {input_image.shape}")

    # input_tensor = transform(input_image).unsqueeze(0)
    # print(f"size of tensor  {input_tensor.shape}")

    # if input_tensor.shape[2] != 384 or input_tensor.shape[3] != 384:
    #     input_tensor = torch.nn.functional.interpolate(
    #         input_tensor, size=(384, 384), mode='bilinear', align_corners=False)

    # with torch.no_grad():
    #     prediction = midas(input_tensor)

    # depth_map = prediction.squeeze().cpu().numpy()
    # return depth_map
    midas = torch.hub.load("isl-org/MiDaS", "MiDaS")

    midas.eval()

    image_path = "../data/3.png"
    input_image = Image.open(image_path).convert("RGB")

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
    print(f"shape {depth_map_resized.shape}")
    return depth_map_resized


def transmittanceMap(depth, illumination, alpha=0.1):
    # invert = 1.0 / img
    # alpha = 0.1
    # beta = 1.0
    # tMap = np.exp(-alpha * invert - beta)
    # return tMap
    # def compute_transmittance(depth_map, illumination_map, alpha=0.1):
    # Normalize depth map
    normalized_depth = depth
    illumination = np.broadcast_to(illumination, normalized_depth.shape)


    # Compute transmittance
    transmittance = np.exp(-alpha * normalized_depth * illumination)

    return transmittance

    # min_channel = np.min(img, axis=2)
    # window_size = 15
    # dark_channel = cv2.erode(min_channel, np.ones((window_size, window_size)))

    # num_pixels = dark_channel.size
    # num_brightest = int(0.001 * num_pixels)
    # indices = np.argpartition(
    #     dark_channel.flatten(), -num_brightest)[-num_brightest:]

    # atmospheric_light = np.max(img.reshape(-1, 3)[indices], axis=0)
    # omega = 0.1
    # t0 = 0.9
    # min_channel2 = np.min(img / atmospheric_light, axis=2)
    # dark_channel2 = cv2.erode(
    #     min_channel2, np.ones((window_size, window_size)))
    # transmission = 1 - omega * dark_channel2

    # # Clamp the values to be between t0 and 1
    # transmission = np.maximum(transmission, t0)
    # transmissionMap = (transmission * 255).astype(np.uint8)
    # return transmissionMap

def volumetricMap():
    print("not implemented yet")
    
def add_realistic_fog(image, distance, fog_intensity=0.2):
    # Generate a random noise image with the same size as the input image
    noise = np.random.normal(0, 1, image.shape).astype(np.uint8)

    # Calculate the depth-based fog factor
    fog_factor = np.exp(-distance * fog_intensity)

    # Blend the original image with the noise based on the fog factor
    foggy_image = cv2.addWeighted(image, 1 - fog_factor, noise, fog_factor, 0)

    return foggy_image


# def opticalLength(depth_map, k):
#     optical_length_map = np.zeros_like(depth_map)

#     for i in range(1, optical_length_map.shape[0]):
#         optical_length_map[i, :] = optical_length_map[i -
#                                                       1, :] + k(depth_map[i - 1, :])

#     return optical_length_map


# def k_fn(n):
#     return 0.1 * n


# def tMap(optical_length_map):
#     return np.exp(-optical_length_map)


def main():
    file = "3.png"
    img = cv2.imread(f"../data/{file}", )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    reflectionMap = img
    illumination_map = generate_illumination_map(img)


    # tMap = transmittanceMap(img)
    # illumination = illuminationEstimation(img)
    # illumination = illuminationEstimation(img)

    # normals_x, normals_y, normals_z = compute_surface_normals(img)
    # print(f"shape {illumination.shape} {img.shape}")

    # albedo_map = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / illumination


    # tMap = np.sqrt(normals_x**2 + normals_y**2 + normals_z**2) * albedo_map
    # tMap = transmittanceMap(img)

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
    # plt.show()

    depth_map = depthMap()
    # plt.imshow(depth_map, cmap="plasma")
    # plt.colorbar()
    # plt.show()
    distance_to_camera = 100

    print(f"shapes {depth_map.shape} {img.shape} {illumination_map.shape}")

    # Add realistic fog with a specified intensity
    foggy_image = add_realistic_fog(img, distance_to_camera, fog_intensity=0.2)


    # illumination = illuminationEstimation(depth_map)
    inv_map = (255-illumination_map)
    tMap = transmittanceMap(depth_map, illumination_map)
    fig, axs = plt.subplots(1, 5, figsize=(10, 5))
    axs[0].imshow(reflectionMap)
    axs[0].set_title('Reflection')

    axs[1].imshow(tMap,cmap='gray')
    axs[1].set_title('Transmittance')
    
    axs[2].imshow(inv_map,cmap='gray')
    axs[2].set_title('Illumination')
    # print(illumination_map)
    # print(np.max(inv_map))
    # print(np.max(reflectionMap))
    # hi = np.multiply(inv_map, reflectionMap)
    
    axs[3].imshow(foggy_image,cmap='gray')
    axs[3].set_title('Img')
    axs[4].imshow(depth_map,cmap='gray')
    axs[4].set_title('depth')
    # axs[3].imshow(hi,cmap='gray')
    # axs[3].set_title('Img')
    plt.show()

    # optical_length_map = opticalLength(depth_map, k_fn)
    # t = tMap(optical_length_map)
    # plt.imshow(t, cmap="viridis")
    # plt.colorbar()
    # plt.title("Transmittance Map")
    # plt.show()


main()
