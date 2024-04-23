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
For starters, make sure you have Python and Jupyter notebook installed in your machine. The requirements.txt file contains all python packages that are required throughout the tutorial, and it can be installed by running the command:

```
pip install -r requirements.txt
```

Using Python environments (such as those in venvs or conda) is recommended to keep your workspace and packages organized. With all requirements installed, you can simply run the notebooks and follow them along.

## Credits
The tutorials are part of the work made at Kiyono Lab, Osaka University.