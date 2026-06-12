#test if we can make video of tracks:




import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import numpy as np

import tifffile as tiff   # added 

def animate_track(df, track_id, frame_rate=30, trail_length=10, save_path=None):
    """
    Animate a single track as a video-like playback.
    
    Parameters:
        df: DataFrame with TRACK_ID, POSITION_X, POSITION_Y columns
        track_id: The track to animate
        frame_rate: Frames per second for playback
        trail_length: Number of past positions to show as a fading trail
        save_path: If provided, saves animation to this path (e.g., 'track.mp4')
    """
    track = df[df['TRACK_ID'] == track_id].reset_index(drop=True)
    
    if len(track) < 2:
        print(f"Track {track_id} has fewer than 2 points, skipping.")
        return
    
    x = track['POSITION_X'].values
    y = track['POSITION_Y'].values
    
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
    
    def init():
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
    df = pd.read_csv(r"Y:\Research\Members\Michelle\CASTA_MS\MS_new\CASTA_handlabeled_groundtruth\1474\cleaned_trackmate_1474_25_488_per_position.csv")
    
    # Animate a specific track at 30 fps
    #animate_track(df, track_id=1, frame_rate=30)
    
    # Or save it as a video file
    animate_track(df, track_id=1, frame_rate=30, save_path="track_1.gif")
    
    # Or animate all tracks
    # animate_all_tracks("your_sptpalm_data.csv", frame_rate=30, save_dir="animations")