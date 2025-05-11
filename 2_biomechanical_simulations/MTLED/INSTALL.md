# ExplicitSim Installation Guide

## Get the source code

Clone the ExplicitSim repository:

    git clone https://username@bitbucket.org/explicitsim/explicitsim.git

or download the latest version from
[here](https://bitbucket.org/explicitsim/explicitsim/get/master.zip).

## Linux installation

### Prerequisites

#### Debian

Install dependencies using APT:

    apt install cmake
    apt install doxygen graphviz
    apt install libboost-dev libboost-filesystem-dev libboost-thread-dev
    apt install libeigen3-dev
    apt install libtinyxml2-dev

If you want to use CGAL for nearest neighbor search:

    apt install libcgal-dev

### Build and install ExplicitSim

Create a build directory:

    mkdir build
    cd build

Configure with CMake

    cmake ..

Alternatively, use `-DCMAKE_INSTALL_PREFIX=/path/to/installation`
to change the installation directory, e.g:

    cmake -DCMAKE_INSTALL_PREFIX=/opt/explicitsim ..

Build (with documentation) and install:

    make
    make doc
    make install

## macOS installation

### Prerequisites

Install the required build tools and dependencies with
[Homebrew](https://brew.sh):

    brew install cmake
    brew install boost eigen tinyxml2

or [MacPorts](https://www.macports.org).

    sudo port install cmake
    sudo port install boost eigen3

The `TinyXML-2` library is not available in MacPorts and
may need to be installed separately.

If you want to use `CGAL` for nearest neighbor search
you also need to install `CGAL` using either homebrew:

    brew install cgal

or macports:

    sudo port install cgal

### Build and install ExplicitSim

Create a build directory:

    mkdir build
    cd build

Configure with CMake

    cmake ..

Build (with documentation) and install:

    make
    make doc
    make install

## Windows installation

TODO
