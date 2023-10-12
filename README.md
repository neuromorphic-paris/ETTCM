# ETTCM

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Code for [Time-to-Contact Map by Joint Estimation of Up-to-Scale Inverse Depth and Global Motion using a Single Event Camera, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Nunes_Time-to-Contact_Map_by_Joint_Estimation_of_Up-to-Scale_Inverse_Depth_and_ICCV_2023_paper.html)

```bibtex
@inproceedings{nunesTimeToContact2023,
	title = {Time-to-Contact Map by Joint Estimation of Up-to-Scale Inverse Depth and Global Motion using a Single Event Camera},
	booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
	author = {Nunes, Urbano Miguel and Perrinet, Laurent Udo and Ieng, Sio-Hoi},
	year = {2023},
	pages = {23653-23663},
}
```

The authors provide this code in the hope it will be useful for understanding the proposed method, as well as for reproducibility of the results.

For more information and more open-source software please visit the neuromorphic-paris' Github page: <https://github.com/neuromorphic-paris>.

# Datasets

We provide all the sequences evaluated in the paper ready to be used: [VL](https://spik.xyz/nc/index.php/s/ER6GMim9qPQRxaF) (104.2MB) or run on a terminal

```bash
wget --header 'Host: spik.xyz' --header 'Sec-GPC: 1' 'https://spik.xyz/nc/index.php/s/ER6GMim9qPQRxaF/download/VL.zip' --output-document 'VL.zip'
```

The original data can be found [here](https://github.com/s-mcleod/ventral-landing-event-dataset).

# Manual Installation

This code was tested on Ubuntu 20.04 distro.

## Dependencies

For a complete list of the dependencies you can also refer to the [Dockerfile](./Dockerfile).

- Base dependencies:

```bash
sudo apt-get install build-essential cmake git graphviz pkg-config libeigen3-dev
```

- Specific version of [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download) that needs to be installed separately, e.g., in `.local` folder:

```bash
git clone https://gitlab.com/libeigen/eigen.git && cd eigen && git checkout 27367017bd0aef15a67ce76b8e263a94c2508a1c
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=~/.local
cmake --build build
cmake --build build --target install
```

- Specific version of [OpenCV](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html):

```bash
git clone https://github.com/opencv/opencv.git && cd opencv && git checkout 82ac7ea23620fb13b7b6be225fa1b0e848f5e72d
cd ..
git clone https://github.com/opencv/opencv_contrib.git && cd opencv_contrib && git checkout c4027ab7f912a3053175477d41b1a62d0078bc5f
cd ../opencv
cmake -S . -B build -DOPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -DCMAKE_INSTALL_PREFIX=~/.local
cmake --build build
cmake --build build --target install
```

## General

After all the dependencies have been installed, to compile this code, assuming you are on the code directory and the specific versions of [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download) and [OpenCV](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html) were installed on the `.local` folder, run on a terminal:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DEigen3_DIR=~/.local/share/eigen3/cmake -DOpenCV_DIR=~/.local/lib/cmake/opencv4 -DETTCM_BUILD_DOC=OFF
cmake --build build
```

# Dockerfile

Just install [Docker](https://www.docker.com/) and then build a container by running on a terminal:

```bash
docker build -t <name-of-the-container> .
```

This will take a while (~6100s or ~1h40m in a standard laptop) since it will build all the necessary packages, including [OpenCV](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html).
After everything is built, to run inside the container, on a terminal run:

```bash
docker run -it <name-of-the-container>
```

# Documentation

To build the documentation, you need to have [Doxygen](https://www.doxygen.nl/) installed:

```bash
sudo apt-get install doxygen
```

Then, recompile:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DEigen3_DIR=~/.local/share/eigen3/cmake -DOpenCV_DIR=~/.local/lib/cmake/opencv4 -DETTCM_BUILD_DOC=ON
cmake --build build
make -C build doc
```

The documentation is built by default using the [Docker](https://www.docker.com/) file.
It can be accessed inside the folder [doc](./build/doc/doxygen).

# Experimental Evaluation

For the following, we assume you are either inside the [Docker](https://www.docker.com/) container or in the root of the code after manually compiling.

## Tab. 2 Partial Results

Only the 2D-odd and 3D sequences provided are evaluated.
To get the partial results in terms of accuracy shown in Tab. 2 just run on a terminal:

```bash
bash table_2_partial_results.sh
```

## Tab. 2 Complete Results

You need to download the data provided [here](https://spik.xyz/nc/index.php/s/ER6GMim9qPQRxaF), uncompress the zip file and move it to the [datasets](./datasets) folder.
To get the results in terms of accuracy shown in Tab. 2 just run on a terminal:

```bash
bash table_2_results.sh
```

# License

The ETTCM code is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Commercial usage is not permitted.
