# Droplet Simulation

![Simulation Output](results/output_20250121_1632.gif)

## Introduction
This project simulates mechanical and osmotic interactions between droplets. It is implemented using C, C++, and CUDA. A graphical user interface (GUI) developed with OpenGL enables real-time monitoring of the simulation. The simulation and rendering processes are executed in separate threads to ensure optimal performance, preventing delays when either process becomes computationally intensive.

## Requirements
To compile and run this project, you need:
- **CMake** (Tested with version 3.31.5)
- **Visual Studio** (Recommended for Windows, tested with version 17.9.0)
- **CUDA-compatible GPU** (Tested with Nvidia GeForce RTX 3080 Ti)
- **Windows 11** (Recommended, tested with version 24H2)

All the dependencies are included in the external directory. There's no need of additional downloads besides this repository.

## Compiling
### **Method 1: Using Visual Studio (Windows)**
1. Use **CMake** to generate a **Visual Studio solution**.
2. Open the solution in **Visual Studio** and build the project.
### **Method 2: Using Standard CMake Procedure**
This method applies to **both Windows and Linux**, but only tested on windows for now.
1. **Open a terminal (or Command Prompt on Windows)**.
2. **Navigate to the project directory**:
   
   ```sh
   cd /path/to/DropletSimulation
   
3. Create a `build` directory and enter it:
   
   ```sh
   mkdir build && cd build
   
4. Run CMake to generate build files:
   
   ```sh
   cmake ..
   
5. Compile the project:
   
   ```sh
   cmake --build .
   
Usually, the compiling time should be less than 5 minutes.

## Usage
An example simulation is provided in `main.cu`. This example creates a 2D specimen measuring 25 mm x 5 mm, containing 20 x 4 droplets. The top two layers of droplets have a higher concentration than the bottom two layers, causing the top layers to shrink and the bottom layers to expand. This results in the specimen folding up, as illustrated in the GIF above. To reproduce the simulation in our paper, change the dropet number in line-67 of main.cu to 2000.

### Running the Simulation
1. To run the simulation:  
   - If using **Visual Studio**, navigate to the directory containing `main.exe` and execute it.  
   - If using **CMake**, navigate to the `build` directory and run one of the following commands, depending on your build configuration:
     
     ```sh
     ./example/Debug/main  # For the Debug build  
     ./example/Release/main  # For the Release build  
     ```

2. The initial state of the specimen will appear in a separate window.  
3. Press the `Enter` key to start the simulation.  
4. Press `Enter` again to pause the simulation. 

### Camera Controls
You can adjust the observation position and angle using the following controls:

#### Movement:
- **W**: Move forward
- **A**: Move left
- **S**: Move backward
- **D**: Move right
- **X**: Move downward
- **Space**: Move upward

#### Rotation:
- Hold the **right mouse button** and move the mouse to change the observation angle.
