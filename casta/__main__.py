import argparse
from casta.main_for_mva_hmm_STA_12 import calculate_spatial_transient_wrapper

def main():
    parser = argparse.ArgumentParser(
        description='Calculate spatial transient analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument('dir', help='Path to the folder containing track data')
    
    # Optional
    parser.add_argument('--dt', type=float, default=0.05, help='Time step for analysis')
    parser.add_argument('--min-track-length', type=int, default=25, help='Minimum track length for analysis')
    parser.add_argument('--plot', type=int, default=0, help='Enable plotting (0=off, 1=on)')
    parser.add_argument('--image-format', choices=['svg', 'tiff'], default='svg', help='Image saving format')
    
    args = parser.parse_args()
    
    calculate_spatial_transient_wrapper(
        args.dir,
        args.min_track_length,
        args.dt,
        args.plot,
        args.image_format
    )

if __name__ == "__main__":
    main()
