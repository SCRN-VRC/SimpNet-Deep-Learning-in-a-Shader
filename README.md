# SimpNet
<img src="Images/SimpNetRender.png"/>
A trainable convolutional neural network coded inside a fragment shader.

## Overview
* Three version of SimpNet: Python, C++, HLSL
* Python + Keras version is the high level overview of the network structure, also does the offline training.
* C++ + OpenCV version is a low level version to help me convert the network into HLSL.
* HLSL version is the one used in-game in VRChat.
* Four pre-trained networks included: Fruits, Age Classification, Hololive VTubers, and VRC Devs.

<img src="Images/Example.png"/>
The age classifier does not work well with anime faces.

## Live Demo
* https://www.vrchat.com/home/launch?worldId=wrld_4cbc4ccb-3c0d-419d-bc8b-e370c951edd2

## Setup for VRChat
#### Prerequisites
* [VRC SDK 2](https://vrchat.com/home/download) setup in a Unity project
* [VRChat layers must be setup already](https://docs.vrchat.com/docs/)
* [Post Processing V2](https://github.com/Unity-Technologies/PostProcessing) (Not needed, but you do need to remove the missing scripts)
### Avatars
1. Sorry I didn't make a version you can carry around
### Worlds
1. Clone the repository
2. Open the Unity project
3. Import VRCSDK2
4. Remove any missing scripts (Post Processing V2)

OR

1. Open a new Unity project
2. Import VRCSDK2
3. Import the SimpNet.unitypackage in [Releases](https://github.com/SCRN-VRC/SimpNet-Deep-Learning-in-a-Shader/releases)
4. Remove any missing scripts (Post Processing V2)

I will be converting everything to Udon at some point.

# Python Code
If you wish to run the Python code, here's what you need.
* Anaconda
* Python 3.7
* TensorFlow
* Keras

You can follow a guide on Keras + Anaconda installations here https://inmachineswetrust.com/posts/deep-learning-setup/

# C++ Code
If you wish to run the C++ code.
* OpenCV – 4.0.1 or above

You can follow a guide on OpenCV + Visual Studio here https://www.deciphertechnic.com/install-opencv-with-visual-studio/

## How it Works
<img src="Images/RenderTexture.png"/>
