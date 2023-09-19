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
### V1.1.1
+ Add transform_listener and static_transform_broadcaster into `OVD_node` for later use.
+ `DeicModule` return prediction result rather than only masked image.
## TODO 
- [ ] Not change files in the module but add wrapper to solve relative path issue (or make a self-contained package).
