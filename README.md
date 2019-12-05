# CONSIM - CONtact SIMulation

CONSIM is C++ library for simulation of rigid multi-body dynamics with contacts, based on [Pinocchio](https://github.com/stack-of-tasks/pinocchio).

## Dependencies
* boost (unit_test_framework)
* eigen3
* [pinocchio](https://github.com/stack-of-tasks/pinocchio)

To install eigen3 on Ubuntu you can use apt-get:
  `sudo apt-get install libeigen3-dev`

To install [pinocchio](https://github.com/stack-of-tasks/pinocchio) follow the instruction on its website.

## Installation

    cd $DEVEL/openrobots/src/
    git clone --recursive git@github.com:andreadelprete/consim.git
    cd consim
    mkdir _build-RELEASE
    cd _build-RELEASE
    cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=$DEVEL/openrobots
    make install

## Python Bindings
To use this library in python, we offer python bindings based on Boost.Python and EigenPy.

To install EigenPy you can compile the source code:

    git clone https://github.com/stack-of-tasks/eigenpy
    
or, on Ubuntu, you can use apt-get:

    sudo apt-get install robotpkg-py27-eigenpy
     
For testing the python bindings, you can run the unit test scripts in the `script` folder, for instance:

    ipython script/test_consim.py
    
