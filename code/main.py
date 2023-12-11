
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



# Define your lighting parameters
# neg x = from left, neg y = from top
sun_direction = np.array([-1.0, -0.3, 0.0])
sun_direction /= np.linalg.norm(sun_direction)  # Normalize to unit vector
sun_intensity = 1.0
sky_intensity = 0.5
brdf = 0.8
visibility = 1.0

# Function to calculate sunlight contribution
def calculate_sunlight_contribution(surface_normal, cos_theta_sun):
    return sun_intensity * brdf * visibility * cos_theta_sun

# Function to calculate skylight contribution
def calculate_skylight_contribution(surface_normal, cos_theta):
    # Integration over hemisphere (simplified for illustration)
    integral_sky = np.sum(sky_intensity * brdf * visibility * cos_theta)
    return integral_sky





# Function to estimate illumination for a single point
def estimate_illumination(surface_normal, sun_direction):
    cos_theta_sun = np.dot(surface_normal, sun_direction)
    
    # Calculate sunlight contribution
    I_sun = calculate_sunlight_contribution(surface_normal, cos_theta_sun)
    
    # Calculate skylight contribution
    incident_light_directions = np.array([1.0, 0.5, 0.0])
    # print(f"shape of surface normal {surface_normal.shape} {incident_light_directions.shape}")
    cos_theta = np.dot(surface_normal, incident_light_directions)  # Replace with actual incident light directions
    I_sky = calculate_skylight_contribution(surface_normal, cos_theta)
    
    # Combine sunlight and skylight
    I = I_sun + I_sky
    
    return I


def generate_illumination_map(image):
    print(f"shape of img {image.shape}")
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
    gradient_x = cv2.Sobel(grey_img, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(grey_img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    enhanced_image = cv2.equalizeHist(np.uint8(grey_img * 255)) / 255.0
    illumination_map = gradient_magnitude * enhanced_image
    return illumination_map


def illuminationEstimation(image):
    return 0


def compute_surface_normals(depth_map):
    depth_grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    depth_grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)

    normal_x = -depth_grad_x / np.sqrt(depth_grad_x**2 + depth_grad_y**2 + 1)
    normal_y = -depth_grad_y / np.sqrt(depth_grad_x**2 + depth_grad_y**2 + 1)
    normal_z = 1 / np.sqrt(depth_grad_x**2 + depth_grad_y**2 + 1)

    surface_normals = np.dstack((normal_x, normal_y, normal_z))
    return surface_normals


def depthMap():
    midas = torch.hub.load("isl-org/MiDaS", "MiDaS")

    midas.eval()

    image_path = "../data/bruh.png"
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
    return depth_map_resized

# def calculate_optical_length(x0, xl, x):
#     # Calculate optical lengths
#     optical_length_x0_x = np.linalg.norm(x0 - x)
#     optical_length_xl_x0 = np.linalg.norm(xl - x0)
    
#     return optical_length_x0_x, optical_length_xl_x0

# def optical_thickness_integrand(t, x_prime, x):
#     # Define the integrand of the optical thickness equation
#     alpha = 0.1#absorption_coefficient(x_prime)
#     delta = 0.05#scattering_coefficient(x_prime)
#     return alpha + delta

# def calculate_optical_thickness(x_prime, x):
#     # Numerically integrate the optical thickness equation
#     result, _ = quad(optical_thickness_integrand, x_prime, x, args=(x_prime, x))
#     return result


def transmittanceMap(depth_map, illumination_map, light_position, scaling_factor=0.1):
    transmittance_map = np.zeros_like(depth_map, dtype=np.float32)

    # Normalize depth map
    normalized_depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

    # Invert depth map
    inverted_depth_map = 1 - normalized_depth_map

    # Convert illumination map to float
    illumination_map = illumination_map.astype(np.float32)

    # Iterate through each pixel in the image
    height, width = depth_map.shape
    for y in range(height):
        for x in range(width):
            x0 = np.array([x, y])
            xl = light_position

            # Calculate optical thickness using numerical integration
            optical_thickness = calculate_optical_thickness(xl[0], x0[0])

            # Adjust transmittance using a scaling factor
            transmittance = np.exp(-scaling_factor * optical_thickness)

            # Apply transmittance to the pixel based on illumination
            transmittance_map[y, x] = transmittance * illumination_map[y, x]

    # Normalize transmittance map
    transmittance_map = (transmittance_map - np.min(transmittance_map)) / (np.max(transmittance_map) - np.min(transmittance_map))

    return transmittance_map


def coord_to_idx(x, y, width):
    return y * width + x 

def transmittanceMap2(img, depth_map):
    # light_pos = np.array([.3,1.0])
    light_idx = 0.50
    # surface_pt = np.array([.4,.6])
    alpha = 0.1
    delta = 0.7
    tmap = np.zeros((img.shape[0], img.shape[1]))
    # print(((alpha + delta) * surface_pt ** 2).shape)
    
    # print(f"const {const}")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    width,height = img.shape
    # depth_map /= np.max(depth_map)
    depth_max = np.max(depth_map)
    # print(depth_map)
    # todo - get rid of these probably 
    dmap = depth_map.copy()
    # dmap[dmap > depth_max-2000] /= 2
    # dmap[dmap > depth_max-1000] /= 1.5
    dmap /= depth_max
    # print(dmap)
    # plt.imshow(dmap, cmap="gray")
    # plt.show()


    # print(f"depth map shape {depth_map.shape}")
    # plt.imshow(depth_map)
    # plt.show()
    for i in range(width):
        # print(f"beginning of i = {i}")
        for j in range(height):
            # print(f"beginning of j = {j}")
            surface_pt = dmap[i][j]
            # surface_idx = coord_to_idx(surface_pt[0], surface_pt[1], width)
            # print(surface_pt)
            # const = np.exp(-((alpha+delta) * light_pos **2 - (alpha + delta) * surface_pt ** 2) )
            const = np.exp(-((alpha+delta) * light_idx **2 - (alpha + delta) * surface_pt ** 2) )

            img_coord = np.array([i / width, j / height])
            img_val = coord_to_idx(i / width,j / height, width) / width#img[i][j] /255#coord_to_idx(img_coord[0], img_coord[1], width)
            # print(f"i : {i} j : {j} i / width {i / width} j // height {j / height} i / height {i/height} j / width {j/width} ")
            # print(f"img coord {img_coord}")
            # val2 = np.exp( - ((alpha+delta) * img_coord ** 2 - (alpha + delta) * surface_pt ** 2))
            # print(f"surface idx {surface_pt} img idx {img_val}")
            val2 = np.exp( - ((alpha+delta) * img_val ** 2 - (alpha + delta) * surface_pt ** 2))
            val =  const * (val2)
            # print(f"val {val}")
            tmap[i][j] = val

    # for j in range(height):
    #     # print(f"beginning of j = {j}")
    #     for i in range(width):
    #         # print(f"beginning of j = {j}")
    #         surface_pt = depth_map[i][j] #/ depth_max
    #         # surface_idx = coord_to_idx(surface_pt[0], surface_pt[1], width)
    #         # print(surface_pt)
    #         # const = np.exp(-((alpha+delta) * light_pos **2 - (alpha + delta) * surface_pt ** 2) )
    #         const = np.exp(-((alpha+delta) * light_idx **2 - (alpha + delta) * surface_pt ** 2) )

    #         img_coord = np.array([i / width, j / height])
    #         img_val = coord_to_idx(i / width,j / height, width) / width #img[i][j] /255#coord_to_idx(img_coord[0], img_coord[1], width)
    #         # print(f"i : {i} j : {j} i / width {i / width} j // height {j / height} i / height {i/height} j / width {j/width} ")
    #         # print(f"img coord {img_coord}")
    #         # val2 = np.exp( - ((alpha+delta) * img_coord ** 2 - (alpha + delta) * surface_pt ** 2))
    #         # print(f"surface idx {surface_pt} img idx {img_val}")
    #         val2 = np.exp( - ((alpha+delta) * img_val ** 2 - (alpha + delta) * surface_pt ** 2))
    #         val =  const * (val2)
    #         # print(f"const {const} val2 {val2} val {val}")
            
            
    #         if (val > 1.0):
    #             val = 1.0
    #         # print(f"val {val}")
    #         tmap[i][j] = val
   
    tmap = np.clip(tmap, 0,1)
    return tmap#np.ones(img.shape) * 255 


def volumetricMap(img, depth_map):
    alpha = 0.6
    delta = 0.4
    vmap = np.zeros((img.shape[0], img.shape[1]))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    width,height = img.shape
    
    depth_max = np.max(depth_map)
    # todo - get rid of these probably 
    dmap = depth_map.copy()
    # dmap[dmap > depth_max-2000] /= 2
    # dmap[dmap > depth_max-1000] /= 1.5
    dmap /= depth_max

    for i in range(width):
        for j in range(height):
            x0 = dmap[i][j]
            x = coord_to_idx(i / width, j / height, width) / width 
            val = (np.exp(-delta * x ** 2 - alpha) - np.exp(-delta * x0 ** 2 - alpha)) - (x * np.exp(-delta * x ** 2 - alpha) - x0 * np.exp(-delta * x0 ** 2 - alpha)) * 50

            # print(f"surface idx {surface_pt} img idx {img_val}")
            # val2 = np.exp( - ((alpha+delta) * img_val ** 2 - (alpha + delta) * surface_pt ** 2))
            # val =  const * (val2)
            # print(f"val {val}")
            vmap[i][j] = val
    return vmap




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
    file = "bruh.png"
    img = cv2.imread(f"../data/{file}", )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    reflectionMap = img

    illumination_map = illuminationEstimation(img)
    depth_map = depthMap()
    light_pos = np.array([1.0, 0.5])
    surface_normals = compute_surface_normals(depth_map)
    height, width = depth_map.shape
    illumination_map = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            surface_normal = surface_normals[y, x, :]
            illumination_map[y, x] = estimate_illumination(surface_normal, sun_direction)

    # tMap = transmittanceMap(depth_map, illumination_map,light_pos)
    tMap = transmittanceMap2(img,depth_map)
    vMap = volumetricMap(img, depth_map)
    # print(np.max(depth_map))
    tmap3d = tMap[:, :, np.newaxis]
    reflectionMap3d = reflectionMap / 255
    depth_map3d = depth_map[:,:,np.newaxis]
    # depth_map3d /= np.max(depth_map3d)
    depth_max = np.max(depth_map3d)
    # todo - get rid of these probably 
    # depth_map3d[depth_map3d > depth_max-2000] /= 2
    # depth_map3d[depth_map3d > depth_max-1000] /= 1.5
    depth_map3d /= depth_max
    print(f"{tmap3d} \n\n\n\n {reflectionMap3d} \n\n\n\n {depth_map3d}")
    out =   reflectionMap3d * depth_map3d + tmap3d
    out=np.clip(out,0,1)

    fig, axs = plt.subplots(1, 6, figsize=(10, 5))

    axs[0].imshow(reflectionMap)
    axs[0].set_title('Reflection')

    axs[1].imshow(tMap, cmap='gray')
    axs[1].set_title('Transmittance')

    # axs[2].imshow(illumination_map, cmap='gray')
    # axs[2].set_title('')
    axs[2].imshow(illumination_map, cmap='gray')
    axs[2].set_title('Illumination')

    # print(illumination_map)
    # print(np.max(inv_map))
    # print(f"types {reflectionMap.dtype} {tMap.dtype} {illumination_map.dtype} {depth_map.dtype}")
    # print(np.max(reflectionMap))
    # hi = np.multiply(inv_map, reflectionMap)

    axs[3].imshow(depth_map, cmap='gray')
    axs[3].set_title('depth')

    axs[4].imshow(vMap, cmap='gray')
    axs[4].set_title('vmap??')

    axs[5].imshow(out, cmap='gray')
    axs[5].set_title('combo??')
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

