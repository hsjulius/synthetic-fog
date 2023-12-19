# synthetic-fog

Our code is contained in main.py. We have a main function that compiles all of our functions together. the depthMap() function creates the depth map using a classification image, depthMapMidas() uses MiDaS. volumetricMap() generates the VMap, and transmittanceMap() generates the TMap. These are compiled to the output image in the main() function.