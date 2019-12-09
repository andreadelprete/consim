# CONSIM - CONtact SIMulation

CONSIM is C++ library for simulation of rigid multi-body dynamics with contacts, based on [Pinocchio](https://github.com/stack-of-tasks/pinocchio).

## Dependencies
For the cpp code:
* boost (unit_test_framework)
* eigen3
* [pinocchio](https://github.com/stack-of-tasks/pinocchio)

For the python code:
* [gepetto-viewer-corba](https://github.com/Gepetto/gepetto-viewer-corba)
* [example-robot-data](https://github.com/Gepetto/example-robot-data)
* [tsid](https://github.com/stack-of-tasks/tsid)
    
## Installation
First you need to install all dependencies.
To install [pinocchio](https://github.com/stack-of-tasks/pinocchio) and the other dependencies you can follow the instruction on the associated website, or you can install them
via apt:
   `sudo apt install robotpkg-py27-example-robot-data robotpkg-collada-dom robotpkg-gepetto-viewer robotpkg-gepetto-viewer-corba robotpkg-hpp-fcl+doc robotpkg-osg-dae robotpkg-py27-pinocchio`

To build this library

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

    ipython script/test_solo_cpp_sim.py

To test the python version of the simulator (with matrix exponential) you can run:

    python scripts/test_exp_integrator_with_quadruped.py
        
