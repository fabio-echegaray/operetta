# Image Processing in Python for PerkinElmer's Operetta Microscope
 This is a (beta) version of a library that I wrote to import image files from the file structure created by Harmony and to further process them in a sequential or parallel way.

## Table of contents
* [General info](#general-info)
* [Screenshots](#screenshots)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Status](#status)
* [Contact](#contact)
* [Licence](#licence)

## General info
Given that the Harmony software for the Operetta microscope has a rather obscure pipeline, which is also not open source, I've decided to write a pipeline on my own. The current version let's you import the files into a Python script, do image analysis and compile the results into a Pandas dataframe. 
Some tools for cell cycle gating and plotting utilities are still available, but will be removed soon to give this package a unique purpose.

## Screenshots
> PENDING!
![Example screenshot](./img/screenshot.png)

## Technologies
Requires Python 3.6 or greater. There are also some packages that this library depends on. For specifics, see requirements.txt, but the main packages required are
* Pandas - version >0.23.4
* Scikit image - version >0.14.0
* Shapely - version >1.6.4
* Matplotlib - version >3.0.2
* Seaborn - version 0.9.0

## Setup
`git clone https://github.com/fabio-echegaray/operetta.git`
Then, on the working folder run: `pip install -r requirements.txt`
    

## Code Examples
### Export of Harmony images
PENDING
### Package usage example
Look at batch.py and position.py

## Features
List of features ready and TODOs for future development
* Importing and csv generation (powered by Pandas) of all the files in the Harmony file output.
* Parallel processing capable. HPC cluster script provided as an example.
* Cell Cycle gating feature (provided two DNA markers, ex. DAPI & EbU)
* Manual measurements of ring-like structures.
* Some plotting utilities.

To-do list:
* Improve processing pipeline.
* Move interactive applications to separate project.

## Status
Project is _in progress_. It still doesn't comply 100% with Harmony file format (would be grateful if anyone has more documentation of it), and working on better processing support.

## Contact
Created by [@fabioechegaray](https://twitter.com/fabioechegaray)
* [fabio.echegaray@gmail.com](mailto:fabio.echegaray@gmail.com)
* [github](https://github.com/fabio-echegaray)
Feel free to contact me!

## Licence
    Image Processing in Python for PerkinElmer's Operetta Microscope
    Copyright (C) 2020  Fabio Echegaray

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.