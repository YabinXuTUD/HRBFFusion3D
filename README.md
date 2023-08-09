# HRBFFusion3D
HRBF-Fusion: Accurate 3D Reconstruction from RGB-D Data Using On-the-Fly Implicits

The system is built based on [ElasticFusion](https://github.com/mp3guy/ElasticFusion) and [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2), please refer to them for dependencies.

First, build Third-party of ORB_SLAM2: DBoW2 and g2o;

Move to project directory, and type the following command:

```bash
cd ./Core/src/ORB_SLAM2_m/Thirdparty/DBoW2/
mkdir build 
cmake ..
make 
```
```bash
cd ../../g2o/
mkdir build
cd ./build
cmake ..
make
```

Then move to folder ORB_SLAM2_m, type following command(before that,comment "${HRBFFUSION_LIBRARY}" in CMakeLists.txt):
```bash
mkdir build
cd ./build
cmake ..
make
```

uncomment "${HRBFFUSION_LIBRARY}" in CMakeLists.txt

Move to Core folder, type following command:
```bash
mkdir build
cd ./build
cmake ../src/
make
```

set OPENNI2 Path:
set(OPENNI2_INCLUDE_DIR path_to_your_OPENNI/Include)
set(OPENNI2_LIBRARY path_to_your_OPENNI//Bin/x64-Release/libOpenNI2.so)

Move to GUI folder, type following command:
```bash
mkdir build
cd ./build
cmake ../src/
make
```

Then the code could be compiled.








