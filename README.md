# remove_4ovek
This is my Graduation Project in Elsys

# Description
Aovek is system for making photos of background without people.

Aovek can be used for making photos of very visited landmark around which there are people all the time.

# Usage
## Installing the packages
### Create virtual environment
```bash
conda env create -f environment.yml --name aovek
```
### Activate virtual environment
```bash
source activate aovek
```
### Deactivate virtual environment
```bash
source deactivate
```
### Delete virtual environment
```bash
conda env remove –name aovek
```
## Process dataset and train CNN for object detection
File aovek.py is responsible for controlling CNN for object detection.
### All possible options of aovek.py
![Alt text](./images/options.png?raw=true)
### Download dataset, Process dataset, Train CNN for object detection
```bash
python3 aovek.py -config_file ./config.json -dataset_download -processes_dataset -train
```
## Starting web application
### Migrate
```bash
python3 web/manage.py migrate
```
### Run server
```bash
python3 web/manage.py runserver
```
