# am_OVD
A package aims to make Open-vocabulary Detect (OVD).

## Installation
Follow `modules/Detic` to install Detic.

## Functionality

## ChangeLog
### V1.0.0
+ Use Realsense to make open-vocabulary detection.
+ Modify `Detic/detic/predictor.py` to make relative path available used everywhere.
### V1.1.0
+ When input custom vocabulary list, DeticModule will change to custom mode automatically.
+ Add online custom vocabulary change (Modify `Detic/detic/predictor.py` to solve the global unchangeable meta issue).

## TODO 
- [ ] Not change files in the module but add wrapper to solve relative path issue (or make a self-contained package).
