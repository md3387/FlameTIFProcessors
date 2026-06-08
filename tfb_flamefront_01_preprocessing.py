# -*- coding: utf-8 -*-
"""
Data reduction for flame surface calc using progress variable approach

Pick the burner bound and length scale


Author:
    Carl Hall
    20260310
"""
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.exposure import equalize_adapthist



plt.close('all')

#######################################
## User inputs

filepath       = r'D:\Downloads\AE471_CY2026_Spring'
filename       = '20260505_ruler.tif'


pick_len_in = 1   #

cutoff         = 0.5                    # threshold cutoff for each image, as a percentage of the min/max
prog_var_value = 0.90                   # location of flame front for integration
pixlen_mm      = 0.31                   # length of pixel for area conversion

vdot =     0.002881                     # m/s
Ainlet = 7.85E-05                       # m^2
gap_mm = 24                             # gap between bottom of image and exit of burner

i_img = 5                              # debug plot to show a particular image
lc    = 'r'                             # line color for contour plots
cmap  = 'grey'                       # colormap for debug plots, scroll to bottom of this for more:
                                        #  https://matplotlib.org/stable/users/explain/colors/colormaps.html
save_fig = False                        # save PNGs of figures generated

# different sizes of images to scale relative size of text/headings to image sizes
figsize=(8,6)
filename_out='./fig_8x6.png'
#figsize=(10,8)
#filename_out='./fig_10x8.png'
#figsize=(12,10)
#filename_out='./fig_12x10.png'


#######################################
## Start of processing

# load image into an array, indexed by image along first dimension
images = tifffile.imread(os.path.join(filepath,filename))
if len(images.shape)==3:
    image = images[0]
else:
    image = images

# contrast stretch
image = equalize_adapthist(image, clip_limit=0.2)

print('\n\n')

#########################################
# pick lengthscale
fig1,ax1 = plt.subplots(1,1,figsize=figsize)

im = ax1.imshow(image,cmap=cmap)
fig1.colorbar(im,ax=ax1)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title(f'Select {pick_len_in} inch spacing vertically')

coord = plt.ginput(-1)

plt.close(fig1)

spatial_cal_mm_pix = (pick_len_in*25.4)/(np.abs(coord[0][1] - coord[1][1]))

print(f'Spatial Calibration: {spatial_cal_mm_pix:0.3f} mm per pixel')



#########################################
# pick lengthscale
fig1,ax1 = plt.subplots(1,1,figsize=figsize)

im = ax1.imshow(image,cmap=cmap)
fig1.colorbar(im,ax=ax1)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Select Top Center of main jet exit')

coord = plt.ginput(-1)

plt.close(fig1)

jet_exit_plane_pix      = int(coord[-1][1])           # pick the y value of the last point since you're probably zooming in
jet_exit_centerline_pix = coord[-1][0]                # pick the x value of the last point since you're probably zooming in

print(f'Jet Exit: Exit Plane: {jet_exit_plane_pix:0.1f}')
print(f'          Centerline: {jet_exit_centerline_pix:0.1f}')


print('\n\nUse these values in the next processing script:')
print(f'    spatial_cal_mm_pix      = {spatial_cal_mm_pix:0.3f}')
print(f'    jet_exit_plane_pix      = {jet_exit_plane_pix}')
print(f'    jet_exit_centerline_pix = {jet_exit_centerline_pix:0.1f}')



#spatial_cal_mm_pix = (pick_len_in*25.4)/(np.abs(coord[0][1] - coord[1][1]))

#print(f'Spatial Calibration: {spatial_cal_mm_pix:0.3f} mm per pixel')


'''

#######################################
## Debug plots
# build debug plot of single image
fig1,ax1 = plt.subplots(1,1,figsize=figsize)

im = ax1.imshow(image,cmap=cmap)
fig1.colorbar(im,ax=ax1)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Single Image')

'''

