# photochromic-reversion

This repository contains the codebase to automatically identify and quantify transient spatial arrest events in long-term single molecule tracks. Building on previously published code for [Diffusional Fingerprinting](https://github.com/hatzakislab/Diffusional-Fingerprinting), it also computes additional features of the molecule track to recognize transient spatial arrest events. 

To run the analysis on a CSV file with the track information, simply run `main.py`. An example file with the required input file structure is provided (`example.csv`). These input files can be generated with any standard image analysis tool, e.g. TrackMate (ImageJ).
Global variables like the CSV input file path, image save path, minimum track length and track time step length (acquisition rate) can be adjusted near the top of the main script.

As an output, an Excel file with the results of the analyzed tracks is saved at the location of the input file. This file contains information such as the number of spatial arrest events, the number of track points and the time spent in each spatial arrest, and averages across all events in all tracks for a given file.
Additionally, the graphs visualizing the tracks and the detected events are saved at the specified path location. 