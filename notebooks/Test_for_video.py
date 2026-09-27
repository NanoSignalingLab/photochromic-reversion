#test if we can make video of tracks:




import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import numpy as np

import tifffile as tiff   # added 



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tifffile as tiff
from matplotlib.collections import LineCollection


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import tifffile as tiff

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import tifffile as tiff


def animate_track(
    df,
    track_id,
    tiff_path,
    dt=0.05,               # seconds per frame
    frame_rate=5,
    trail_length=None,     # None = show full history
    track_scale=10,        # use 10 if your coordinates need scaling
    save_path=None
):

    # -------------------------
    # Track
    # -------------------------
    track = df[df["TRACK_ID"] == track_id].copy()

    if len(track) < 2:
        print("Track too short.")
        return

    track = track.sort_values("POSITION_T")

    x = track["POSITION_X"].values * track_scale
    y = track["POSITION_Y"].values * track_scale
    t = track["POSITION_T"].values

    padding = 10 # pixels

    xmin = x.min() - padding
    xmax = x.max() + padding

    ymin = y.min() - padding
    ymax = y.max() + padding

    # Convert time -> TIFF frame
    frame_idx = np.round(t / dt).astype(int)

    # -------------------------
    # TIFF
    # -------------------------
    img_stack = tiff.imread(tiff_path)

    n_img_frames = len(img_stack)

    # Keep only valid frames
    valid = frame_idx < n_img_frames

    x = x[valid]
    y = y[valid]
    t = t[valid]
    frame_idx = frame_idx[valid]

    if len(frame_idx) == 0:
        print("No valid frames found.")
        return

    # -------------------------
    # Animate ONLY relevant TIFF frames
    # -------------------------
    start_frame = frame_idx.min()
    end_frame = frame_idx.max()

    movie_frames = np.arange(start_frame, end_frame + 1)

    # -------------------------
    # Figure
    # -------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    im = ax.imshow(
        img_stack[start_frame],
        cmap="gray",
        origin="upper"
    )

    point, = ax.plot([], [], "ro", markersize=5)

    trail = LineCollection([], linewidths=2)
    ax.add_collection(trail)

    title = ax.set_title("")

    ax.set_aspect("equal")


    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)  # origin='upper'

    # -------------------------
    # Update
    # -------------------------
    def update(movie_frame):

        frame_img = img_stack[movie_frame]
        im.set_data(frame_img)

        # all track positions observed up to current TIFF frame
        idx = np.where(frame_idx <= movie_frame)[0]

        if len(idx) == 0:
            point.set_data([], [])
            title.set_text(
                f"Frame {movie_frame} | t={movie_frame*dt:.2f}s"
            )
            return im, point, trail, title

        current = idx[-1]

        cx = x[current]
        cy = y[current]

        point.set_data([cx], [cy])

        # -------------------------
        # Zoom around particle
        # -------------------------
        zoom = 10

        #ax.set_xlim(cx - zoom, cx + zoom)
        #ax.set_ylim(cy + zoom, cy - zoom)

        # -------------------------
        # Fading trail
        # -------------------------
        start_idx = 0

        if trail_length is not None:
            start_idx = max(0, current - trail_length)

        segments = []
        colors = []

        for j in range(start_idx, current):

            segments.append([[x[j], y[j]], [x[j + 1], y[j + 1]]])
            #alpha = (j - start_idx + 1) / (current - start_idx + 1)
            alpha=1

            colors.append((1, 0, 0, alpha))

        trail.set_segments(segments)
        trail.set_colors(colors)

        title.set_text(
            f"Frame {movie_frame} | "
            f"t={movie_frame*dt:.2f}s | "
            f"Track t={t[current]:.2f}s"
        )

        return im, point, trail, title

    # -------------------------
    # Animation
    # -------------------------
    anim = FuncAnimation(
        fig,
        update,
        frames=movie_frames,
        interval=1000 / frame_rate,
        blit=False
    )

    # -------------------------
    # Save / Show
    # -------------------------
    if save_path:
        anim.save(save_path, writer="pillow", fps=frame_rate)
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return anim

    


    
    #df = pd.read_csv(r"C:\Users\miche\Desktop\track_annotator\test\cleaned_trackmate_p1_001_allspots_per_position.csv")

df=pd.read_csv(r"C:\Users\miche\Desktop\track_annotator\1474\cleaned_trackmate_1474_25_488_per_position.csv")
tiff_p=r"Y:\Research\Members\Michelle\CASTA_MS\TIRFM\231130\1474-025_20uWConv-1_50-1000.tif"
save_p=r"Y:\Research\Members\Michelle\CASTA_MS\TIRFM\231130\track0_987.gif"

    
    # works below for getting TIRFs 
    #animate_track(df,track_id=987,tiff_path=tiff_p,dt=0.05, track_scale=10,   save_path=save_p) # scaling: set to 1 if scaling is not needed
    



    #### add function to also animate track without background image:


    #def animate_track(df, track_id, frame_rate=30, trail_length=10, save_path=None):


    # Animate a single track as a video-like playback.
    
    # Parameters:
    #     df: DataFrame with TRACK_ID, POSITION_X, POSITION_Y columns
    #     track_id: The track to animate
    #     frame_rate: Frames per second for playback
    #     trail_length: Number of past positions to show as a fading trail
    #     save_path: If provided, saves animation to this path (e.g., 'track.mp4')

def animate_track_blank(df, track_id,frame_rate,trail_length, save_path):

    # Select one track
    track = df[df['TRACK_ID'] == track_id].reset_index(drop=True)

    # Extract coordinates
    x = track['POSITION_X'].to_numpy()
    y = track['POSITION_Y'].to_numpy()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set axis limits
    padding = 1
    ax.set_xlim(x.min() - padding, x.max() + padding)
    ax.set_ylim(y.min() - padding, y.max() + padding)

    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title(f"Track {track_id}")

    # Current particle position
    point, = ax.plot([], [], 'ro', markersize=8)

    # Trail object
    trail = LineCollection([], cmap='hot', linewidths=2)
    ax.add_collection(trail)

    # Frame text
    frame_text = ax.text(
        0.02, 0.95, '',
        transform=ax.transAxes,
        fontsize=10
    )

    # Initialization
    def init_blank():
        point.set_data([], [])
        trail.set_segments([])
        frame_text.set_text('')
        return point, trail, frame_text

    # Update function
    def update_blank(frame):

        # Current point
        point.set_data([x[frame]], [y[frame]])

        # Trail
        start = max(0, frame - trail_length)

        segments = []
        colors = []

        for i in range(start, frame):

            seg = [[x[i], y[i]], [x[i + 1], y[i + 1]]]
            segments.append(seg)

            colors.append(i - start)

        trail.set_segments(segments)

        if len(colors) > 0:
            trail.set_array(np.array(colors))

        frame_text.set_text(f'Frame: {frame + 1}/{len(x)}')

        return point, trail, frame_text

    # Animation
    interval = 1000 / frame_rate

    anim = FuncAnimation(
        fig,
        update_blank,
        frames=len(x),
        init_func=init_blank,
        interval=interval,
        blit=True,
        repeat=True
    )

    # Save or display
    if save_path is not None:

        anim.save(save_path, writer='ffmpeg', fps=frame_rate)

        print(f"Saved animation to: {save_path}")

    else:
        plt.show()

    return anim



df=pd.read_csv(r"C:\Users\miche\Desktop\track_annotator\1474\cleaned_trackmate_1474_25_488_per_position.csv")
outpath=r"Y:\Research\Members\Michelle\CASTA_MS\TIRFM\231130\cell_25\track84_blank.gif"
animate_track_blank(df, track_id=84,frame_rate=30, trail_length=200, save_path=outpath)




