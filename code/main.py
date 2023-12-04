import argparse
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2


"""

We def ne the input photograph, the radiance attenuation
ratio map and the volumetric scattering radiance map as the
Ref ection Map, the Transmittance Map and the Volumetric Map respectively

input photograph = reflection map 
radiance attenuation ratio map = transmittance map 
volumetric scattering radiance map = volumetric map 

"""


def illuminationEstimation():
    print("not implemented yet")

def geometryEstimation():
    print("not implemented yet")

def transmittanceMap():
    print("not implemented yet")

def volumetricMap():
    print("not implemented yet")

def main():
    file = "1.png"
    img = cv2.imread(f"../data/{file}", )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    reflectionMap = img
    
    plt.imshow(img)
    plt.show()

    


main()