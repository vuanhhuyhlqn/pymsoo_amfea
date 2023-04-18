# Multifactorial Evolution Algorithms - MFEA 
This is a library for multifactorial evolution algorithms. 


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Requirements](#requirements)
* [Usage](#usage)
* [Project Status](#project-status)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->


## General Information
A library that includes utilities for using and upgrading versions of the MFEA.

## Requirements
  All in file requirement.yml

## Usage
### Run docker tutorial 
- Create folder that contain 2 files: **run.py** to run code, **cfg.yaml** contains config parameter (see MFEA_DaS, SM_MFEA_SBX). Name folder is name of algorithms.
- Use: `docker build -t pymsoo:lastest . ` to build images docker 
- In file **run.sh**, change `MODEL` value to name of folder created
- In file **run_docker.txt**, change path `LOCAL_SAVE_PATH` and `DOCKER_SAVE_PATH` to the path want to save.  
- Copy all line in file **run_docker.txt** and paste to cmd and run
## Project Status
The project is in process of update and fix


## Acknowledgements
- This project was trained and tutorial by [MSO Lab](http://mso.soict.hust.edu.vn/)
- This project was based on these paper below: 
  - [Multifactorial Evolution: Toward Evolutionary Multitasking](https://ieeexplore.ieee.org/abstract/document/7161358)
  - ...

## Contact
- Lê Trung Kiên - kien.letrung610@gmail.com
- Đinh Tấn Minh - tanminh8b@gmail.com
