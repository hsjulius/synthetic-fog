
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
from scipy.integrate import quad


"""
We define the input photograph, the radiance attenuation
ratio map and the volumetric scattering radiance map as the
Ref ection Map, the Transmittance Map and the Volumetric Map respectively

input photograph = reflection map 
radiance attenuation ratio map = transmittance map 
volumetric scattering radiance map = volumetric map 
"""


def compute_surface_normals(depth_map):
    depth_grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    depth_grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)

    normal_x = -depth_grad_x / np.sqrt(depth_grad_x**2 + depth_grad_y**2 + 1)
    normal_y = -depth_grad_y / np.sqrt(depth_grad_x**2 + depth_grad_y**2 + 1)
    normal_z = 1 / np.sqrt(depth_grad_x**2 + depth_grad_y**2 + 1)

    surface_normals = np.dstack((normal_x, normal_y, normal_z))
    return surface_normals


def depthMap(image_name):
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


def coord_to_idx(x, y, width):
    return y * width + x


def transmittanceMap2(img, depth_map, classified):
    light_idx = -.6
    alpha = 0.1
    delta = 0.7
    tmap = np.zeros((img.shape[0], img.shape[1]))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    width, height = img.shape

    # for i in range(width):
    #     print(f"beginning of i = {i}")
    #     for j in range(height):
    #         # print(f"beginning of j = {j}")
    #         surface_pt = depth_map[i][j]
    #         const = np.exp(-((alpha+delta) * light_idx ** 2 - (alpha + delta) * surface_pt ** 2))

    #         # img_coord = np.array([i / width, j / height])
    #         img_val = coord_to_idx(i / width, j / height, width) / width

    #         val2 = np.exp( - ((alpha+delta) * img_val ** 2 - (alpha + delta) * surface_pt ** 2))
    #         val =  const * val2

    #         tmap[i][j] = val
    #         print(f"val {val} surface pt {surface_pt} img val {img_val}")
    for j in range(height):
        # print(f"------------------------------beginning of j = {j}---------------------")
        for i in range(width):
            # print(f"beginning of j = {j}")
            surface_pt = depth_map[i][j]
            const = np.exp(-((alpha + delta) * light_idx **
                           2 - (alpha + delta) * surface_pt ** 2))

            # img_coord = np.array([i / width, j / height])
            # coord_to_idx(i / width, j / height, width) / width
            img_val = img[i][j] / 255
            # print(img_val)
            val2 = np.exp(-((alpha+delta) * img_val ** 2 -
                          (alpha + delta) * surface_pt ** 2))
            # print(f"val2 inner { - ((alpha+delta) * img_val ** 2 - (alpha + delta) * surface_pt ** 2)} const inner {-(((alpha+delta) * light_idx ** 2 - (alpha + delta) * surface_pt ** 2))}")
            # print(f"val2 {val2}")
            val = const * (val2)

            tmap[i][j] = const / 2.5
            # print(f"val {val} const {const} val2 {val2} surface pt {surface_pt} img val {img_val}")

    tmap = np.clip(tmap, 0, 1)

    # tmap[classified[:,:,2] == 255] = np.average(tmap[classified[:,:,2] == 255])
    # tmap[classified[:,:,0] == 255] = np.average(tmap[classified[:,:,0] == 255])
    # # tmap[classified[:,:,0] == 255] = np.average(tmap[classified[:,:,0] == 255])
    # # max_grad = np.max(tmap[classified[:,:,0] == 255])
    # # min_grad = np.min(tmap[classified[:,:,0] == 255])
    # ground_pixels = np.all(classified == [0,255,0], axis=-1)
    # for i in range(width):
    #     tmap[i, ground_pixels[i,:]] = i / width#(i/width) * 0.1
    # tmap[classified[:,:,0] == 255] = tmap[classified[:,:,0] == 255]/2
    # r, g, b = classified[:,:,0], classified[:,:,1], classified[:,:,2]
    # buildings_mask = (r == 255)
    # ground_mask = (g == 255)
    # sky_mask = (b == 255)

    # plt.imshow(tmap[buildings_mask])
    # plt.show()
    # plt.imshow(tmap[ground_mask])
    # plt.show()
    # plt.imshow(tmap[sky_mask])
    # plt.show()
    # tmap_at_buildings = np.where(buildings_mask, tmap, np.nan)
    # tmap_at_ground = np.where(buildings_mask, tmap, np.nan)
    # tmap_at_sky = np.where(buildings_mask, tmap, np.nan)
    # tmap_at_ground = 1
    # print(np.average(tmap[buildings_mask]))
    # tmap[ground_mask] /= 1.5
    # print(np.average(tmap[buildings_mask]))

    # plt.imshow()
    # plt.show()

    # plt.imshow(np.where(ground_mask, tmap, np.nan))
    # plt.show()

    # plt.imshow(np.where(sky_mask, tmap, np.nan))
    # plt.show()
    # print(f"shapes {tmap.shape} {classified.shape}")

    # print(classified[:,:,2])
    return tmap


def volumetricMap(img, depth_map):
    # alpha = 0.6
    # delta = 0.4
    # vmap = np.zeros((img.shape[0], img.shape[1]))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # width,height = img.shape

    # depth_max = np.max(depth_map)
    # # todo - get rid of these probably
    # dmap = depth_map.copy()
    # # dmap[dmap > depth_max-2000] /= 2
    # # dmap[dmap > depth_max-1000] /= 1.5
    # dmap /= depth_max

    # for i in range(width):
    #     for j in range(height):
    #         x0 = dmap[i][j]
    #         x = img[i][j] / 255 #oord_to_idx(i / width, j / height, width) / width
    #         val = (np.exp(-delta * x ** 2 - alpha) - np.exp(-delta * x0 ** 2 - alpha)) - (x * np.exp(-delta * x ** 2 - alpha) - x0 * np.exp(-delta * x0 ** 2 - alpha)) * 50

    #         # print(f"surface idx {surface_pt} img idx {img_val}")
    #         # val2 = np.exp( - ((alpha+delta) * img_val ** 2 - (alpha + delta) * surface_pt ** 2))
    #         # val =  const * (val2)
    #         # print(f"val {val}")
    #         vmap[i][j] = val
    # return vmap
    light_idx = -.6
    alpha = 0.1
    delta = 0.7
    tmap = np.zeros((img.shape[0], img.shape[1]))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    width, height = img.shape

    # for i in range(width):
    #     print(f"beginning of i = {i}")
    #     for j in range(height):
    #         # print(f"beginning of j = {j}")
    #         surface_pt = depth_map[i][j]
    #         const = np.exp(-((alpha+delta) * light_idx ** 2 - (alpha + delta) * surface_pt ** 2))

    #         # img_coord = np.array([i / width, j / height])
    #         img_val = coord_to_idx(i / width, j / height, width) / width

    #         val2 = np.exp( - ((alpha+delta) * img_val ** 2 - (alpha + delta) * surface_pt ** 2))
    #         val =  const * val2

    #         tmap[i][j] = val
    #         print(f"val {val} surface pt {surface_pt} img val {img_val}")
    for j in range(height):
        # print(f"------------------------------beginning of j = {j}---------------------")
        for i in range(width):
            # print(f"beginning of j = {j}")
            surface_pt = depth_map[i][j]
            const = np.exp(-((alpha + delta) * light_idx **
                           2 - (alpha + delta) * surface_pt ** 2))

            # img_coord = np.array([i / width, j / height])
            img_val = coord_to_idx(i / width, j / height,
                                   width) / width  # img[i][j] / 255
            # print(img_val)
            val2 = np.exp(-((alpha+delta) * img_val ** 2 -
                          (alpha + delta) * surface_pt ** 2))
            # print(f"val2 inner { - ((alpha+delta) * img_val ** 2 - (alpha + delta) * surface_pt ** 2)} const inner {-(((alpha+delta) * light_idx ** 2 - (alpha + delta) * surface_pt ** 2))}")
            # print(f"val2 {val2}")
            val = const * (val2)

            tmap[i][j] = val2

            # print(f"val {val} const {const} val2 {val2} surface pt {surface_pt} img val {img_val}")
    plt.imshow(tmap)
    plt.show()
    tmap = np.clip(tmap, 0, 1)
    return tmap


sun_intensity = 1.0
sky_intensity = 0.5
brdf = 0.8
visibility = 1.0


def calculate_sunlight_contribution(cos_theta_sun):
    return sun_intensity * brdf * visibility * cos_theta_sun


def calculate_skylight_contribution(cos_theta):
    integral_sky = np.sum(sky_intensity * brdf * visibility * cos_theta)
    return integral_sky


def estimate_illumination(surface_normal, sun_direction, incident_light_directions):
    # mag_norm = np.linalg.norm(surface_normal)
    # mag_sun = np.linalg.norm(sun_direction)
    # dot_sun = np.dot(surface_normal, sun_direction)
    cos_theta_sun = np.dot(surface_normal, sun_direction)
    # cos_theta_sun = dot_sun / (mag_norm * mag_sun)

    # Calculate sunlight contribution
    I_sun = calculate_sunlight_contribution(cos_theta_sun)

    # Calculate skylight contribution
    # mag_in = np.linalg.norm(incident_light_directions)
    # dot_sky = np.dot(surface_normal, incident_light_directions)
    cos_theta_sky = np.dot(surface_normal, incident_light_directions)
    # cos_theta_sky = dot_sky / (mag_norm * mag_in)
    I_sky = calculate_skylight_contribution(cos_theta_sky)

    # Combine sunlight and skylight
    I = I_sun + I_sky

    return I


def generate_incident_light_directions(num_samples=1000):
    phi = np.random.uniform(0, 2 * np.pi, num_samples)
    theta = np.arccos(np.random.uniform(0, 1, num_samples))

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    directions = np.vstack((x, y, z))
    directions /= np.linalg.norm(directions)
    return directions


def depthMap2(img_classes, darkness_factor=.75):
    h, w, _ = img_classes.shape
    depth = np.ones((h, w))

    depth[img_classes[:, :, 0] == 255] = 0.35  # buildings
    depth[img_classes[:, :, 2] == 255] = 0  # sky
    # depth[img_classes[:, :, 1] == 255] = 0.2

    ground_pixels = np.all(img_classes == [0, 255, 0], axis=-1)
    # green = depth[img_classes[:,:,1] == 255]
    # print(green)
    for i in range(h):
        # ground — gets brighter as it gets nearer
        depth[i, ground_pixels[i, :]] = (i / h) * darkness_factor
        # intensity = 0.5 #+ (i / h) #* (1.0 - 0.5) * darkness_factor
        # depth[i, ground_pixels[i, :]] = intensity
        # print(intensity)

    # deal with edges
    depth[depth == 1] = 0.5

    return depth


def main():
    file = "bruh.png"
    img = cv2.imread(f"../data/{file}", )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_class = cv2.imread(f"../data/bruh_class.png", )
    img_class = cv2.cvtColor(img_class, cv2.COLOR_BGR2RGB)

    depth_map = depthMap2(img_class)
    # plt.imshow(depth_map, cmap='gray')
    # plt.show()

    reflectionMap = img

    # sun_direction = np.array([-1.0, -0.3, 0.0])
    # surface_normals = compute_surface_normals(depth_map)
    # height, width = depth_map.shape
    # illumination_map = np.zeros((height, width))
    # incident_light_directions = generate_incident_light_directions()

    # for y in range(height):
    #     for x in range(width):
    #         surface_normal = surface_normals[y, x, :]
    #         illumination_map[y, x] = estimate_illumination(
    #             surface_normal, sun_direction, incident_light_directions)

    # tMap = transmittanceMap(depth_map, illumination_map,light_pos)
    tMap = transmittanceMap2(img, depth_map, img_class)

    # plt.imshow(tMap, cmap='gray')
    # plt.show()
    # cv2.imwrite(f"../results/out.png", tMap)

    vMap = volumetricMap(img, depth_map)
    # print(np.max(depth_map))
    tmap3d = tMap[:, :, np.newaxis]
    vmap3d = vMap[:, :, np.newaxis]
    reflectionMap3d = reflectionMap / 255
    depth_map3d = depth_map[:, :, np.newaxis]

    out = ((reflectionMap3d) * (tmap3d) + (vmap3d / 2))  # * reflectionMap3d
    out = np.clip(out, 0, 1)

    fig, axs = plt.subplots(1, 5, figsize=(10, 5))

    axs[0].imshow(reflectionMap)
    axs[0].set_title('Reflection')

    axs[1].imshow(tMap, cmap='gray')
    axs[1].set_title('Transmittance')

    # axs[2].imshow(illumination_map, cmap='gray')
    # axs[2].set_title('')
    # axs[2].imshow(illumination_map, cmap='gray')
    # axs[2].set_title('Illumination')

    # print(illumination_map)
    # print(np.max(inv_map))
    # print(f"types {reflectionMap.dtype} {tMap.dtype} {illumination_map.dtype} {depth_map.dtype}")
    # print(np.max(reflectionMap))
    # hi = np.multiply(inv_map, reflectionMap)

    axs[2].imshow(depth_map, cmap='gray')
    axs[2].set_title('depth')

    axs[3].imshow(vMap, cmap='gray')
    axs[3].set_title('vmap??')

    axs[4].imshow(out, cmap='gray')
    axs[4].set_title('combo??')
    # axs[4].imshow(foggy_image, cmap='gray')
    # axs[4].set_title('Img')
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
