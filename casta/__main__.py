from casta.main_for_mva_hmm_STA_12 import calculate_spatial_transient_wrapper

if __name__ == "__main__":
    plotting_flag=0
    dt=0.05
    min_track_length=25
    plotting_saving_nice_image_flag=0
    
    image_saving_flag="svg"
    image_saving_flag="tiff"

    #folderpath1=r"C:\Users\miche\Desktop\simualted tracks\test_real_tracks"
    #folderpath1=r"D:\photochromic_reversion_data\tst"
    folderpath1=r"/Users/schulzp9/Documents/casta"

    calculate_spatial_transient_wrapper(folderpath1, min_track_length, dt, plotting_flag, image_saving_flag)
