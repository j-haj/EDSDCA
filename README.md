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