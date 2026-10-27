# RandomizedTensorCompressionCXX
CXX implementations of Star-M tensor compression algorithms.

Install Just and Visual Studio 2022 or 2026 with **Desktop development with C++**,
a Windows SDK, and CMake 3.25 or newer.

Then run the following commands from a PowerShell prompt in the project root directory:

```powershell
just run                 # Build and run optimized code with PDB debug information
just nocompile           # Run the existing RelWithDebInfo executable without building
just debug               # Build and run unoptimized Debug code
just build               # Incrementally build RelWithDebInfo without running
just configure           # Generate the Windows x64 build tree
just fetch-deps          # Download, build, and install the dependencies locally
just clean               # Clean the application, retaining built dependencies
just run "D:/other data" # Inspect a different directory
```
