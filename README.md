# CASTA - Computational Analysis of Spatial Transient Arrest


This repository contains the codebase to automatically identify and quantify transient spatial arrest events in long-term single molecule tracks. 

To run the analysis on a CSV file with the track information, simply run `main.py`. An example file with the required input file structure is provided (`example.csv`). These input files can be generated with any standard image analysis tool, e.g. TrackMate (ImageJ). Adjust the frame rate, minimum track length and image saving option (tiff or svg) at the bottom.

As an output, an Excel file with the results of the analyzed tracks is saved at the location of the input file. This file contains information such as the number of spatial arrest events (STA), the number of tracking points, the time spent in each spatial arrest, as well as the area and averages across all events in all tracks for a given file.
Additionally, the graphs visualizing the tracks and the detected events are saved at the specified path location. 