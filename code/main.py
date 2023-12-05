# import argparse
# import os
# import sys
# import csv
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2


# """

# We def ne the input photograph, the radiance attenuation
# ratio map and the volumetric scattering radiance map as the
# Ref ection Map, the Transmittance Map and the Volumetric Map respectively

# input photograph = reflection map 
# radiance attenuation ratio map = transmittance map 
# volumetric scattering radiance map = volumetric map 

# """


# # Load the input image
# # input_image_path = 'your_image.jpg'
# # image = Image.open(input_image_path)

# # Perform automatic color correction to estimate illumination

# def illuminationEstimation():
#     print("not implemented yet")

# def geometryEstimation():
#     print("not implemented yet")

# def transmittanceMap():
#     print("not implemented yet")

# def volumetricMap():
#     print("not implemented yet")

# def main():
#     file = "1.png"
#     img = cv2.imread(f"../data/{file}", )
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#     reflectionMap = img
#     # corrected_image = automatic_color_correction(img)
#     # img.show(title='Original Image')
#     # corrected_image.show(title='Illumination Corrected Image')


    
#     plt.imshow(img)
#     plt.show()


# main()




import cv2
import numpy as np

def compute_dark_channel(image, window_size=15):
    # Compute the dark channel of the image
    min_channel = np.min(image, axis=2)
    dark_channel = cv2.erode(min_channel, np.ones((window_size, window_size)))

    return dark_channel

def compute_atmospheric_light(image, dark_channel):
    # Select the top 0.1% brightest pixels from the dark channel
    num_pixels = dark_channel.size
    num_brightest = int(0.001 * num_pixels)
    indices = np.argpartition(dark_channel.flatten(), -num_brightest)[-num_brightest:]

    # Find the maximum intensity value in the original image at these pixels
    atmospheric_light = np.max(image.reshape(-1, 3)[indices], axis=0)

    return atmospheric_light

def compute_transmission(image, atmospheric_light, omega=0.95, t0=0.1):
    # Compute the transmission map using the haze model
    transmission = 1 - omega * compute_dark_channel(image / atmospheric_light)

    # Clamp the values to be between t0 and 1
    transmission = np.maximum(transmission, t0)

    return transmission

# Load the input image
image_path = '../data/1.png'
image = cv2.imread(image_path) / 255.0  # Normalize to [0, 1]

# Compute the dark channel
dark_channel = compute_dark_channel(image)

# Compute the atmospheric light
atmospheric_light = compute_atmospheric_light(image, dark_channel)

# Compute the transmission map
transmission_map = compute_transmission(image, atmospheric_light)

# Display the results
cv2.imshow('Original Image', cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
cv2.imshow('Transmission Map', (transmission_map * 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
