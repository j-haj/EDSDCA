# Build Instructions
From this directory, issue the following commnads, based on your preferred build
system

## make

```
cmake ..
make
```

## ninja
To use the ``ninja`` build system you must first have it installed on your
seystem. To install on macOS, for example, simply run the command ``brew install
ninja`` assuming Homebrew is installed on the system. To build the project under
ninja, use the following commands:

```
cmake -GNina ..
ninja
```
