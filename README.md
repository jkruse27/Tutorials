# Tutorials for Heart Rate Variability Analysis

This repository contains a collection of tutorials that cover topics from data cleaning and traditional HRV feature extraction all the way to more complex applications. The goal here is to provide an understanding of how the features and methods work by introducing documented Python implementations for all methods covered.

## Covered topics
- Cleaning the data
- Time domain analysis
- Frequency domain analysis
- Non linear analysis
- DMA
- Non-gaussianity
- Applications of deep learning

## Structure
This repository is organized as follows:

- data: Contains sample data that is used as demonstration during the tutorials.
- notebooks: Jupyter notebooks containing the actual tutorials.
- utils: Python script files with the implemented functions in a more accessible/concise manner, so that they can be imported into other projects.

## How to use

Using Python environments (such as those in venvs or conda) is recommended to keep your workspace and packages organized. With all requirements installed, you can simply run the notebooks and follow them along. For the DMA code, because it was implemented in Cython for speed, a few more steps are required. First, make sure you have a C compiler installed (for Linux users gcc usually already comes installed. For Mac OS, gcc can be installed via Apple XCode. For Windows users, MSVC is the recommended compiler, and it can be installed with Visual Studio as well. For further info you can refer to [this page](http://docs.cython.org/en/latest/src/quickstart/install.html)). Make sure you also have Python and Jupyter notebook installed in your machine.   

The requirements.txt file contains all python packages that are required throughout the tutorial, and it can be installed by running the command:

```
pip install -r requirements.txt
```

This package contains some main functions such as:
- *read_file*: Given the path to a csv, this function reads the data and pre-processes it (for HRV)
- *create_scales*: Generates a sequence of scales in which the DMA will be evaluated.  
- *dma*: Computes the generalized variance for each of the given scales.  

Some examples are given in the `examples` folder.

## Credits
The tutorials are part of the work made at Kiyono Lab, Osaka University.