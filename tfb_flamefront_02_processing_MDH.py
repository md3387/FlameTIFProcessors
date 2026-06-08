# -*- coding: utf-8 -*-
"""
Data reduction for flame surface calc using progress variable approach

Script #2 - Process VIS cases for flame area


Author:
    Carl Hall
    20260312
"""

###############################################################################
###############################################################################
###############################################################################
## User inputs

###############################################################################
# 20260309 Cases
filepath  = r'D:\Downloads\AE471_CY2026_Spring'


# values from 01 script
spatial_cal_mm_pix      = 0.069
jet_exit_plane_pix      = 977
jet_exit_centerline_pix = 576.9


filename        = '20260505_3_vis.tif'
pix_offset_base = 0
cutoff    = 0.2


###############################################################################
# universal variables
i_img = 5
lc    = 'r'
lw    = 1
cmap  = 'grey'
save_fig = True
debug = False
figsize=(18,8)




###############################################################################
###############################################################################
###############################################################################
## Start of processing
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.filters import gaussian, sobel
from tfb_flamefront_functions import integrateOgive

plt.close('all')


###############################################################################
# load image into an array, indexed by image along first dimension
images = tifffile.imread(os.path.join(filepath,filename))
if len(images.shape)==2:
    print('Single image found')
    i_img = 0
    N_images = 1
    images = [images]
else: 
    N_images = images.shape[0]



###############################################################################
# for each image calculate the contour and integrate to get flame area
flame_area_mm2 = np.nan*np.zeros(N_images)
flame_y        = np.arange(0,images.shape[1])
flame_ledge    = np.nan*np.zeros((N_images,images.shape[1]))
flame_redge    = np.nan*np.zeros((N_images,images.shape[1]))
flame_diameter = np.nan*np.zeros((N_images,images.shape[1]))
flame_center   = np.nan*np.zeros((N_images,images.shape[1]))
flame_gradient = []
for j,image in enumerate(images):
    print(f"\rProcessing frame {j+1:4d} of {N_images:4d}", end="", flush=True)
    
    if debug and j>10:
        break
    
    flame_gradient.append(gaussian(sobel(image), sigma=2.0))
    
    first = True
    for i in range(int(jet_exit_plane_pix-pix_offset_base),-1,-1):
        scan_line = image[i,:]
        grad_line = flame_gradient[-1][i,:]
        
        _scan_cutoff = cutoff*(np.nanmax(image)-np.nanmin(image))+np.nanmin(image)
        _grad_cutoff = cutoff*(np.nanmax(grad_line)-np.nanmin(grad_line))+np.nanmin(grad_line)
        _index = []
        for k in np.arange(10,1000):
            if  grad_line[k]>grad_line[k-1] and\
                grad_line[k]>grad_line[k+1] and\
                grad_line[k]>_grad_cutoff   and\
                scan_line[k]>_scan_cutoff:
                _index.append(k)
        
        _flame_front = []
        for index in _index:
            _x = np.arange(index-1,index+2)
            _y = grad_line[_x]
            p  = np.polyfit(_x,_y,deg=2)
            _i = -p[1]/(2.*p[0])
            if abs(index-_i)<1:
                _flame_front.append(_i)
            else:
                _flame_front.append(index)
        
        if len(_flame_front)>1:
            flame_ledge[j][i]    =                         np.min(_flame_front)
            flame_redge[j][i]    =  np.max(_flame_front)
            flame_diameter[j][i] =  np.max(_flame_front) - np.min(_flame_front)
            flame_center[j][i]   = (np.max(_flame_front) + np.min(_flame_front) )/2
        elif len(_flame_front)<=1:
            flame_ledge[j][i]    = flame_center[j][i-1]
            flame_redge[j][i]    = flame_center[j][i-1]
            flame_diameter[j][i] = np.nan
            flame_center[j][i]   = flame_center[j][i-1]
        
        if first:
            if np.isnan(flame_ledge[j][i]) or np.isnan(flame_redge[j][i]):
                continue
            else:
                first = False
    
    flame_area_mm2[j] = 0.
    for i in range(image.shape[1]-1,0,-1):
        R_1 = flame_diameter[j][i  ]/2  * spatial_cal_mm_pix
        R_2 = flame_diameter[j][i-1]/2  * spatial_cal_mm_pix
        h   = 1.                        * spatial_cal_mm_pix
        A = np.pi * (R_1+R_2) * np.sqrt( (R_1-R_2)**2 + h**2 )
        if np.isfinite(A):
            flame_area_mm2[j] += A

###############################################################################
# POST-PROCESSING: Gap-filling and spike removal on flame edges
###############################################################################

def clean_flame_edge(edge, max_gap=20, spike_window=20, spike_threshold=5):
    """
    Clean a single flame edge array (flame_ledge or flame_redge for one frame).
    
    Steps:
      1. Identify the valid (non-NaN) region of the flame — ignore NaNs outside
         the flame extent entirely, only fill gaps within it.
      2. Remove spikes using a rolling median comparison.
      3. Interpolate across any remaining NaN gaps within the flame extent.
    
    Parameters
    ----------
    edge             : 1D array of pixel positions (NaN where not detected)
    max_gap          : maximum gap size (rows) to fill via interpolation
    spike_window     : number of rows used for rolling median (must be odd)
    spike_threshold  : deviation from rolling median (pixels) to flag as spike
    
    Returns
    -------
    edge_clean : cleaned 1D array
    n_spikes   : number of spike-flagged points
    n_gaps     : number of gap-filled points
    """
    edge_clean = edge.copy()
    n_spikes   = 0
    n_gaps     = 0
    
    # --- find the valid flame extent (first and last finite row) ---
    finite_idx = np.where(np.isfinite(edge_clean))[0]
    if len(finite_idx) < 3:
        return edge_clean, 0, 0          # not enough data to do anything
    
    i_start = finite_idx[0]
    i_end   = finite_idx[-1]
    flame_slice = slice(i_start, i_end + 1)
    
    # --- step 1: spike removal via rolling median ---
    half = spike_window // 2
    for i in range(i_start + half, i_end - half + 1):
        window = edge_clean[i - half : i + half + 1]
        finite_window = window[np.isfinite(window)]
        if len(finite_window) < 3:
            continue
        median_val = np.median(finite_window)
        if np.isfinite(edge_clean[i]) and abs(edge_clean[i] - median_val) > spike_threshold:
            edge_clean[i] = np.nan      # flag spike as NaN for interpolation
            n_spikes += 1
    
    # --- step 2: gap filling via linear interpolation (within flame extent only) ---
    rows = np.arange(len(edge_clean))
    finite_idx2 = np.where(np.isfinite(edge_clean[flame_slice]))[0] + i_start
    
    if len(finite_idx2) < 2:
        return edge_clean, n_spikes, 0
    
    # find NaN runs within the flame extent and only fill short ones
    nan_mask = ~np.isfinite(edge_clean)
    i = i_start
    while i <= i_end:
        if nan_mask[i]:
            # find the end of this NaN run
            j = i
            while j <= i_end and nan_mask[j]:
                j += 1
            gap_size = j - i
            if gap_size <= max_gap:
                # interpolate across this gap using nearest finite neighbors
                i_before = i - 1
                i_after  = j
                if i_before >= i_start and i_after <= i_end and \
                   np.isfinite(edge_clean[i_before]) and np.isfinite(edge_clean[i_after]):
                    for k in range(i, j):
                        t = (k - i_before) / (i_after - i_before)
                        edge_clean[k] = (1 - t) * edge_clean[i_before] + t * edge_clean[i_after]
                        n_gaps += 1
            i = j
        else:
            i += 1
    
    return edge_clean, n_spikes, n_gaps


###############################################################################
# Apply cleaning to all frames

print('\nCleaning flame edges (gap-fill + spike removal)...')

flame_ledge_clean    = np.copy(flame_ledge)
flame_redge_clean    = np.copy(flame_redge)
flame_diameter_clean = np.copy(flame_diameter)
flame_center_clean   = np.copy(flame_center)
flame_area_mm2_clean = np.nan * np.zeros(N_images)

total_spikes = 0
total_gaps   = 0

for j, image in enumerate(images):
    print(f"\rCleaning frame {j+1:4d} of {N_images:4d}", end="", flush=True)
    
    # clean left and right edges independently
    flame_ledge_clean[j], n_spikes_l, n_gaps_l = clean_flame_edge(flame_ledge[j])
    flame_redge_clean[j], n_spikes_r, n_gaps_r = clean_flame_edge(flame_redge[j])
    total_spikes += n_spikes_l + n_spikes_r
    total_gaps   += n_gaps_l   + n_gaps_r
    
    # recompute diameter and center from cleaned edges
    flame_diameter_clean[j] = flame_redge_clean[j] - flame_ledge_clean[j]
    flame_center_clean[j]   = (flame_redge_clean[j] + flame_ledge_clean[j]) / 2
    
    # recompute flame area from cleaned diameter
    flame_area_mm2_clean[j] = 0.
    for i in range(image.shape[1] - 1, 0, -1):
        R_1 = flame_diameter_clean[j][i  ] / 2 * spatial_cal_mm_pix
        R_2 = flame_diameter_clean[j][i-1] / 2 * spatial_cal_mm_pix
        h   = 1.                              * spatial_cal_mm_pix
        A   = np.pi * (R_1 + R_2) * np.sqrt((R_1 - R_2)**2 + h**2)
        if np.isfinite(A):
            flame_area_mm2_clean[j] += A

print(f'\nEdge cleaning complete.')
print(f'  Total spikes removed: {total_spikes}')
print(f'  Total gaps filled:    {total_gaps}')
print(f'  Mean area before cleaning: {np.nanmean(flame_area_mm2):8.2f} mm^2')
print(f'  Mean area after  cleaning: {np.nanmean(flame_area_mm2_clean):8.2f} mm^2')


###############################################################################
# INTERACTIVE REVIEW: Let user include/exclude frames from the average
###############################################################################

print('\n\n--- Interactive Frame Review ---')
print('For each frame: press  I  to Include,  X  to Exclude,  Q  to stop reviewing early.')
print('Frames not reviewed will be INCLUDED by default.\n')

include_mask = np.ones(N_images, dtype=bool)

fig_review, ax_review = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(bottom=0.2)

status_text = fig_review.text(0.5, 0.05, '', ha='center', va='center',
                               fontsize=13, fontweight='bold',
                               transform=fig_review.transFigure)

decision = {'value': None}

def on_key(event):
    if event.key in ('i', 'I', 'x', 'X', 'q', 'Q'):
        decision['value'] = event.key.upper()   # signal the waiting loop — do NOT close the figure

fig_review.canvas.mpl_connect('key_press_event', on_key)

stop_early = False
for j in range(N_images):
    decision['value'] = None

    # Left panel: raw image with contour overlay
    ax_review[0].cla()
    ax_review[0].imshow(images[j], cmap=cmap)
    ax_review[0].plot(flame_ledge_clean[j], flame_y, lc, linewidth=lw)
    ax_review[0].plot(flame_redge_clean[j], flame_y, lc, linewidth=lw)
    ax_review[0].set_xlim(jet_exit_centerline_pix - 30/spatial_cal_mm_pix,
                          jet_exit_centerline_pix + 30/spatial_cal_mm_pix)
    ax_review[0].set_ylim(jet_exit_plane_pix, jet_exit_plane_pix - 180/spatial_cal_mm_pix)
    ax_review[0].set_title(f'Frame {j+1}/{N_images}  |  Area = {flame_area_mm2_clean[j]:.1f} mm²')
    ax_review[0].set_xlabel('Distance [mm]')
    ax_review[0].set_ylabel('Distance [mm]')

    # Right panel: flame area time series with current frame highlighted
    ax_review[1].cla()
    ax_review[1].plot(np.arange(N_images)+1, flame_area_mm2_clean, 'b.-', linewidth=0.8, markersize=3)
    ax_review[1].axvline(j+1, color='r', linewidth=1.5, label='Current frame')
    ax_review[1].set_xlabel('Frame #')
    ax_review[1].set_ylabel('Flame Area [mm²]')
    ax_review[1].set_title('Flame Area — All Frames')
    ax_review[1].legend(fontsize=9)

    status_text.set_text(f'Frame {j+1}/{N_images}  |  Press  I = Include    X = Exclude    Q = Quit review')

    fig_review.canvas.draw()
    fig_review.canvas.flush_events()

    # Block until user presses a valid key
    while decision['value'] is None:
        plt.pause(0.05)

    if decision['value'] == 'Q':
        print(f'  Review stopped early at frame {j+1}. Remaining frames included by default.')
        stop_early = True
        break
    elif decision['value'] == 'X':
        include_mask[j] = False
        #print(f'  Frame {j+1:4d}: EXCLUDED  (area = {flame_area_mm2[j]:.1f} mm²)')
    #else:
        #print(f'  Frame {j+1:4d}: included  (area = {flame_area_mm2[j]:.1f} mm²)')

plt.close(fig_review)   # close only once, after the loop is done

n_included = np.sum(include_mask)
n_excluded = np.sum(~include_mask)
print(f'\nReview complete: {n_included} frames included, {n_excluded} excluded.')

# Apply mask — set excluded frames to NaN so nanmean ignores them
flame_area_mm2_filtered = flame_area_mm2_clean.copy()
flame_area_mm2_filtered[~include_mask] = np.nan

flame_diameter_filtered = flame_diameter.copy()
flame_diameter_filtered[~include_mask] = np.nan
###############################################################################
# End of interactive review — remainder of script uses filtered arrays
###############################################################################
#  Save include_mask and key features to csv for ML training
##############################################################################
import pandas as pd

# extract per-frame features for ML training
feature_rows = []
for j in range(N_images):
    img = np.array(images[j], dtype=float)
    grad = flame_gradient[j]
    
    row = {
        # identifiers
        'filename'  : filename,
        'frame'     : j,
        'label'     : int(include_mask[j]),   # 1=include, 0=exclude
        
        # raw image statistics
        'img_mean'  : np.nanmean(img),
        'img_std'   : np.nanstd(img),
        'img_max'   : np.nanmax(img),
        'img_p10'   : np.nanpercentile(img, 10),
        'img_p90'   : np.nanpercentile(img, 90),
        
        # gradient statistics
        'grad_mean' : np.nanmean(grad),
        'grad_std'  : np.nanstd(grad),
        'grad_max'  : np.nanmax(grad),
        
        # flame area (raw and cleaned)
        'area_raw'  : flame_area_mm2[j],
        'area_clean': flame_area_mm2_clean[j],
        
        # contour quality metrics
        'ledge_nan_frac' : np.mean(~np.isfinite(flame_ledge[j])),   # fraction of rows with no detection
        'redge_nan_frac' : np.mean(~np.isfinite(flame_redge[j])),
        'ledge_std'      : np.nanstd(np.diff(flame_ledge_clean[j])),  # smoothness of edge
        'redge_std'      : np.nanstd(np.diff(flame_redge_clean[j])),
        'diameter_std'   : np.nanstd(flame_diameter_clean[j]),
        'center_std'     : np.nanstd(flame_center_clean[j]),
        'area_change'    : flame_area_mm2_clean[j] - flame_area_mm2[j],  # how much cleaning changed things
    }
    feature_rows.append(row)

df_new = pd.DataFrame(feature_rows)

# append to a master CSV that accumulates across all your sessions
master_csv = os.path.join(filepath, 'ml_training_data.csv')
if os.path.exists(master_csv):
    df_existing = pd.read_csv(master_csv)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_combined = df_new

df_combined.to_csv(master_csv, index=False)
print(f'\nTraining data saved: {len(df_combined)} total labeled frames in {master_csv}')
##########################################################################################
# End of ML Training Script
##########################################################################################



print(f'\n Results for {filename} with cutoff={cutoff:0.2f}')
print(f'    Flame Average Area (all frames):      {np.nanmean(flame_area_mm2):8.2f} mm^2')
print(f'    Flame Average Area (included frames): {np.nanmean(flame_area_mm2_filtered):8.2f} mm^2')
print(f'    Flame Std Dev Area  (included frames): {np.nanstd(flame_area_mm2_filtered):8.2f} mm^2')
###############################################################################
# Additional troubleshooting

flame_height_mm = np.nan*np.zeros(N_images)
for j,_flame_diameter in enumerate(flame_diameter):
    _val = np.where(np.isfinite(flame_diameter[0]))[0]
    _val = np.max(_val) - np.min(_val)
    flame_height_mm[j] = _val * spatial_cal_mm_pix

_flame_diameter_max_mm = np.nanmax(flame_diameter_filtered*spatial_cal_mm_pix)
print('\n Debug information')
print(f'    Maximum observed flame diameter: {_flame_diameter_max_mm:8.2f} mm')
print(f'    Flame Average Height:            {np.mean(flame_height_mm):8.2f} mm')

_H = np.mean(flame_height_mm)
_R = _flame_diameter_max_mm / 2
print(f'    Flame area for tangent ogive:    {integrateOgive(_R, _H):8.2f} mm^2')
print(f'    Flame area for cone:             {np.pi*_R*np.sqrt(_H**2 + _R**2):8.2f} mm^2')


#######################################
## Debug plot  (uses i_img — picks first included frame if i_img was excluded)
if not include_mask[i_img]:
    i_img = np.where(include_mask)[0][0]
    print(f'\n(i_img was excluded; switching debug plot to first included frame: {i_img})')

fig1,ax = plt.subplots(1,6,figsize=figsize)

im = ax[0].imshow(images[i_img],cmap=cmap)
ax[0].set_xlabel('Distance [mm]')
ax[0].set_ylabel('Distance [mm]')
ax[0].set_title('Single Image')

im = ax[1].imshow(flame_gradient[i_img],cmap=cmap)
ax[1].set_xlabel('Distance [mm]')
ax[1].set_ylabel('Distance [mm]')
ax[1].set_title('Sobel Transform')

im = ax[2].imshow(images[i_img],cmap=cmap)
ax[2].set_xlabel('Distance [mm]')
ax[2].set_ylabel('Distance [mm]')
ax[2].set_title('Single Image + Contour')

im = ax[3].imshow(np.mean(np.asarray(images,dtype=float),axis=0),cmap=cmap)
ax[3].set_xlabel('Distance [mm]')
ax[3].set_ylabel('Distance [mm]')   
ax[3].set_title('Averaged Image + Contours')

ax[4].set_axis_off()
ax[5].set_axis_off()

fig1.text(.7,.5,f'Results for {filename} with cutoff={cutoff:0.2f}\n'+
                f'  Frames included/excluded: {n_included}/{n_excluded}\n'+
                f'  Flame Avg Area (all):      {np.nanmean(flame_area_mm2):8.2f} mm$^2$\n'+
                f'  Flame Avg Area (filtered): {np.nanmean(flame_area_mm2_filtered):8.2f} mm$^2$\n'+
                '\nDebug information\n'+
                f'  Flame Maximum Diameter:       {_flame_diameter_max_mm:8.2f} mm\n'+
                f'  Flame Average Height:         {np.nanmean(flame_height_mm):8.2f} mm\n'+
                f'  Flame area for tangent ogive: {integrateOgive(_R, _H):8.2f} mm$^2$\n'+
                f'  Flame area for cone:          {np.pi*_R*np.sqrt(_H**2 + _R**2):8.2f} mm$^2$\n',
                fontfamily='monospace')

for i in range(4):
    ax[i].set_xlim(jet_exit_centerline_pix-30/spatial_cal_mm_pix,
                   jet_exit_centerline_pix+30/spatial_cal_mm_pix)
    ax[i].set_xticks([jet_exit_centerline_pix-20/spatial_cal_mm_pix ,
                      jet_exit_centerline_pix                       ,
                      jet_exit_centerline_pix+20/spatial_cal_mm_pix ])
    ax[i].set_xticklabels(['-20','0','20'])
    ax[i].set_yticks([jet_exit_plane_pix                      ,
                      jet_exit_plane_pix-20/spatial_cal_mm_pix,
                      jet_exit_plane_pix-40/spatial_cal_mm_pix,
                      jet_exit_plane_pix-60/spatial_cal_mm_pix,
                      jet_exit_plane_pix-80/spatial_cal_mm_pix,
                      jet_exit_plane_pix-100/spatial_cal_mm_pix,
                      jet_exit_plane_pix-120/spatial_cal_mm_pix,
                      jet_exit_plane_pix-140/spatial_cal_mm_pix,
                      jet_exit_plane_pix-160/spatial_cal_mm_pix,
                      jet_exit_plane_pix-180/spatial_cal_mm_pix  ])
    ax[i].set_yticklabels(['0','20','40','60','80','100','120','140','160','180'])

ax[2].plot(flame_ledge[i_img],flame_y,lc,linewidth=lw)
ax[2].plot(flame_redge[i_img],flame_y,lc,linewidth=lw)

for j, (_flame_ledge, _flame_redge) in enumerate(zip(flame_ledge,flame_redge)):
    _lw = lw/2 if include_mask[j] else 0   # don't draw excluded frames
    ax[3].plot(_flame_ledge,flame_y,lc,linewidth=_lw)
    ax[3].plot(_flame_redge,flame_y,lc,linewidth=_lw)

fig1.tight_layout()

if save_fig:
    _filename = 'flamearea_'+os.path.splitext(filename)[0]+'.png'
    print(f'\nSaving fig out to: {_filename}')
    fig1.savefig(_filename,dpi=150)