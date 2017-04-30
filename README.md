# EDSDCA
Efficient Distributed SDCA

* Jeff Hajewski
* Mingrui Liu

## Quickstart

### Dependencies
The project requires the following to build:

* cmake
* C++ 14

Optionally, you may use the Ninja build system if you have it installed.

### Building and Running
To run the project, first move into the ``build`` directory (e.g., ``cd build``
 from project ``root``). From within the build directory, run the command

 ```
 cmake ..
 make
 ```

 To run the executable, run:

 ```
 ./edsdca_main
 ```

### Using Ninja
If you have the Ninja bulid system installed, you can select Ninja via:
```
cmake -GNinja ..
ninja
```

Running the executable is the same as above.

## Testing
There are also test executables that are created (depending on available
hardware). These are run in the same manner as ``edsdca_main``

## Modifying EDSDCA Main
To use a different dataset in ``edsdca_main``, modify the path to the specified
dataset in the file ``src/edsdca/run_models.cc``. Make sure the new dataset has
a format similar to the other datasets. The main supported formats are:

* CSV
* LibSVM sparse representation
