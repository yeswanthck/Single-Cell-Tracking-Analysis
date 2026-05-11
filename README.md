Single-Cell-Tracking-Analysis aims to quantitatively analyze confocal timelapse microscopy images of Mycobacteria and Pseudomonas to specifically track cell division, cell death and cell stasis.


System Requirements

Hardware requirements

Single-Cell-Tracking-Analysis package requires a standard computer with enough RAM to support the in-memory options.

Software requirements

OS Requirements

This package is supported for macOS and windows. The package has been tested on the following systems:

macOS: Tahoe 26.3,
Windows: Windows 11

Python Dependencies

This package mainly depends on the Python scientific stack. 

numpy
cv2,
matplotlib,
pathlib,
scipy,
scikit-image.

For those wanting to try it out: The best place to start is the ipython/jupiter notebook. This is what you need:

1. A working version of python with all the dependencies installed. 
2. The data. 

Download the github file to the directory of choice: 

git clone https://github.com/yeswanthck/Single-Cell-Tracking-Analysis

Unzip the data and put it in the data folder. Once the data is in place, the program can be run in the iPython notebook.  

Sample data included: 
14dC01-04
14dP01-04
(Supplementary Fig 6, 14-day pseudolysogeny)

Currently, all the sample data must be run manually by replacing the folder name with appropriate folder name on line 468. 

The program will analyze frame by frame and output the quantification and perform a quality check by outputting the visualization at each step of the program. The output files will be located in the project folder. The entire process should take a few minutes for each sample data.

License

This project is covered under the Apache 2.0 License.
