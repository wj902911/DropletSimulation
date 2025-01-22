# Droplet Simulation

![Simulation Output](results/output_20250121_1632.gif)

## Introduction
This project simulates mechanical and osmotic interactions between droplets. It is implemented using C, C++, and CUDA. A graphical user interface (GUI) developed with OpenGL enables real-time monitoring of the simulation. The simulation and rendering processes are executed in separate threads to ensure optimal performance, preventing delays when either process becomes computationally intensive.

## Compiling
The project can be compiled on Windows. Use CMake to generate a Visual Studio solution, then build the examples using Visual Studio.

## Usage
An example simulation is provided in `main.cu`. This example creates a 2D specimen measuring 25 mm x 5 mm, containing 20 x 4 droplets. The top two layers of droplets have a higher concentration than the bottom two layers, causing the top layers to shrink and the bottom layers to expand. This results in the specimen folding up, as illustrated in the GIF above.

### Running the Simulation
1. Build the project and run the example.
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