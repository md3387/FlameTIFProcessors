import tifffile
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.animation as animation

plt.close('all')
#######################################
## Prerequisites
# FFMpegWriter 

#######################################
## User inputs
filepath = r'D:\Downloads\AE471_CY2026_Spring'
filename = '20260505_1_vis.tif'

#######################################
## Tunable Parameters
# fps - frames per second in output
# dpi - resolution of each frame
#figsize - aspect ratio of the output (8, 6) is 8x6inches
#bitrate - video quality inf the FFMpegWriter [kbps]

#######################################
## Load the TIFF file
full_path = os.path.join(filepath, filename)
tif_data = tifffile.imread(full_path)

print(f"Loaded TIFF shape: {tif_data.shape}")
print(f"Data type: {tif_data.dtype}")
print(f"Min: {tif_data.min()}, Max: {tif_data.max()}")

#######################################
## Normalize data for display
# Handle multi-frame (3D) or single-frame (2D) TIFFs
if tif_data.ndim == 2:
    # Single frame — wrap in a list so the loop below still works
    frames = [tif_data]
elif tif_data.ndim == 3:
    # Could be (frames, H, W) or (H, W, channels)
    if tif_data.shape[2] in [3, 4]:  # RGB or RGBA
        frames = [tif_data]
    else:
        frames = [tif_data[i] for i in range(tif_data.shape[0])]
elif tif_data.ndim == 4:
    # (frames, H, W, channels)
    frames = [tif_data[i] for i in range(tif_data.shape[0])]
else:
    raise ValueError(f"Unexpected TIFF dimensions: {tif_data.ndim}")

print(f"Number of frames: {len(frames)}")

def normalize_frame(frame):
    """Normalize a frame to 0–1 range for display."""
    f = frame.astype(np.float64)
    f_min, f_max = f.min(), f.max()
    if f_max > f_min:
        f = (f - f_min) / (f_max - f_min)
    return f

#######################################
## Build the animation
fps        = 10          # frames per second — adjust as needed
output_mp4 = os.path.join(filepath, filename.replace('.tif', '.mp4'))

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
ax.axis('off')

# Use 'gray' colormap for single-channel; ignored for RGB
cmap = 'gray' if frames[0].ndim == 2 else None

im = ax.imshow(normalize_frame(frames[0]), cmap=cmap,
               vmin=0, vmax=1, interpolation='nearest')

title = ax.set_title('Frame 1', fontsize=10)

def update(i):
    im.set_data(normalize_frame(frames[i]))
    title.set_text(f'Frame {i + 1} / {len(frames)}')
    return [im, title]

ani = animation.FuncAnimation(
    fig, update,
    frames=len(frames),
    interval=1000 / fps,
    blit=True
)

#######################################
## Save to MP4  (requires ffmpeg on PATH)
writer = animation.FFMpegWriter(fps=fps, bitrate=2000,
                                extra_args=['-vcodec', 'libx264',
                                            '-pix_fmt', 'yuv420p'])
ani.save(output_mp4, writer=writer)
plt.close(fig)

print(f"\nSaved MP4 to: {output_mp4}")