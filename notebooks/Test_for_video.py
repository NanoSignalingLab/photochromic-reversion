#test if we can make video of tracks:




import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import numpy as np

import tifffile as tiff   # added 

#def animate_track(df, track_id, frame_rate=30, trail_length=10, save_path=None):
""" def animate_track(df, track_id, tiff_path, frame_rate=30, trail_length=10, save_path=None):

    
    Animate a single track as a video-like playback.
    
    Parameters:
        df: DataFrame with TRACK_ID, POSITION_X, POSITION_Y columns
        track_id: The track to animate
        frame_rate: Frames per second for playback
        trail_length: Number of past positions to show as a fading trail
        save_path: If provided, saves animation to this path (e.g., 'track.mp4')

    track = df[df['TRACK_ID'] == track_id].reset_index(drop=True)
    
    if len(track) < 2:
        print(f"Track {track_id} has fewer than 2 points, skipping.")
        return
    
    x = track['POSITION_X'].values
    y = track['POSITION_Y'].values

    ## added
     # OPTIONAL: if you have frame column, use it
    if 'POSITION_T' in track.columns:
        #frames = track['POSITION_T'].values.astype(int)
        dt = 0.05  # seconds per frame

        frames = np.round(track['POSITION_T'].values / dt).astype(int)

    tiff_stack = tiff.imread(tiff_path)
 
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate bounds with padding
    padding = 0.1 * max(x.ptp(), y.ptp()) or 1
    ax.set_xlim(x.min() - padding, x.max() + padding)
    ax.set_ylim(y.min() - padding, y.max() + padding)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Track {track_id}')
    
    # Plot full trajectory as faint background
    ax.plot(x, y, 'lightgray', linewidth=0.5, alpha=0.5)
    
    # Current position marker
    point, = ax.plot([], [], 'ro', markersize=10)
    
    # Trail as a LineCollection for color gradient
    trail = LineCollection([], cmap='hot', linewidths=2)
    ax.add_collection(trail)
    
    # Frame counter
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                         verticalalignment='top', fontsize=10)
     """
    

""" def init():
        point.set_data([], [])
        trail.set_segments([])
        frame_text.set_text('')
        return point, trail, frame_text
    
    def update(frame):
        # Current position
        point.set_data([x[frame]], [y[frame]])
        
        # Trail with fading effect
        start = max(0, frame - trail_length)
        if frame > 0:
            segments = []
            colors = []
            for i in range(start, frame):
                segments.append([[x[i], y[i]], [x[i+1], y[i+1]]])
                # Color intensity increases toward current position
                colors.append((i - start) / trail_length)
            trail.set_segments(segments)
            trail.set_array(np.array(colors))
        
        frame_text.set_text(f'Frame: {frame + 1}/{len(x)}')
        return point, trail, frame_text
    
    interval = 1000 / frame_rate  # milliseconds between frames
    anim = FuncAnimation(fig, update, frames=len(x), init_func=init,
                         blit=True, interval=interval, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=frame_rate)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    return anim """

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
    frame_rate=30,
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
        zoom = 80

        ax.set_xlim(cx - zoom, cx + zoom)
        ax.set_ylim(cy + zoom, cy - zoom)

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
            alpha = (j - start_idx + 1) / (current - start_idx + 1)

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

    



def animate_all_tracks(csv_path, frame_rate=30, save_dir=None):
    """Animate all tracks in the CSV file."""
    df = pd.read_csv(csv_path)
    track_ids = df['TRACK_ID'].unique()
    
    print(f"Found {len(track_ids)} tracks")
    
    for tid in track_ids:
        save_path = f"{save_dir}/track_{tid}.gif" if save_dir else None
        animate_track(df, tid, frame_rate=frame_rate, save_path=save_path)
# Usage
if __name__ == "__main__":
    # Load your data
    #df = pd.read_csv(r"Y:\Research\Members\Michelle\CASTA_MS\MS_new\CASTA_handlabeled_groundtruth\1474\cleaned_trackmate_1474_25_488_per_position.csv")
    
    # Animate a specific track at 30 fps
    #animate_track(df, track_id=1, frame_rate=30)
    
    # Or save it as a video file
    #animate_track(df, track_id=1, frame_rate=30, save_path="track_1.gif")
    df = pd.read_csv(r"C:\Users\miche\Desktop\track_annotator\test\cleaned_trackmate_p1_001_allspots_per_position.csv")
    tiff_p=r"C:\Users\miche\Desktop\track_annotator\test\p1_001.tif"


    

    #animate_track(df,track_id=118,tiff_path=tiff_p, frame_rate=30   )
    animate_track(
    df,
    track_id=118,
    tiff_path=r"C:\Users\miche\Desktop\track_annotator\test\p1_001.tif",
    dt=0.05,
    track_scale=10,   # set to 1 if scaling is not needed
    save_path=None
)
    
    # Or animate all tracks
    # animate_all_tracks("your_sptpalm_data.csv", frame_rate=30, save_dir="animations")