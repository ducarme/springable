## Getting started
### Preparation
I recommend to use PyCharm for setting up the program.
1. Download and install Python 3.10 or more recent
2. Download and install PyCharm
3. In Pycharm, create a new project:
   * Select a directory where the project will live in
   * For the Python interpreter:
     * Pick **New environment** using **Virtualenv**
     * Location: `<project_directory>/venv`
     * Base interpreter: select Python interpreter corresponding to the newly downloaded Python (3.10 or up).
       > Windows: select the file name **python.exe**. It is usually located at `C:\Users\<username>\AppData\Local\Programs\Python\Python310`.
       
       > MACOSX: select the file named **python**. It is usually located at `/usr/local/bin/python3.10`.
     * Leave checkboxes **Inherit...** and **Make available...** unticked.
   * Do *not* create a `main.py` welcome script
   * Click **Create** button (A project should now be created)
4. Still in PyCharm, Go to *File>Settings* (or open *Preferences* under MACOSX). In the left panel, go to *Project:<project_name>/Python interpreter* and click the `+` button to install the following packages:
   * package **numpy**
   * package **scipy**
   * package **matplotlib**
5. Using the file explorer (or Finder on MACOSX), copy the folder named `truss_analysis_project`, the file `requirements.txt` and the `README.md` files. In PyCharm, right-click on the project directory shown in the left panel and select **Paste**.
6. In PyCharm (still), click *Add configuration...*. This opens a dialog window: click the `+` button then *Python*. In the right-panel, leave everything unchanged except for *Script path* where you select file `<project_directory>/truss_analysis_project/start_simulation.py`, then click **OK**. Repeat this step with file `<project_directory>/truss_analysis_project/run_tests.py`.
7. In PyCharm (still), open file `<project_directory>/truss_analysis_project/src/post_processing/graphics_maker.py`. Locate line `mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\ducarme\\Documents\\truss-analysis\\ffmpeg\\ffmpeg-5.1.2' \
                                        r'-essentials_build\\bin\\ffmpeg.exe'`. Replace the right hand side with the path of your ffmpeg executable. If not installed on your machine, you can download it from https://ffmpeg.org/download.html. It is necessary to save animations
8. For a sanity check, run file `run_tests.py` (by clicking the green *play* triangle in the top bar. Make sure the selected script is `run_test.py`). This script automatically runs multiple spring simulations and displays the force-displacement curves obtained numerically with the analytical solutions. For each test, an animation is also generated. Manually close the plot windows to allow the script to continue in-between each test. The result files are stored under `<project_directory>/truss_analysis_project/_test_executions`.
9. Using the file explorer (or Finder on MACOSX), copy the folders `FF60A_SET_2`, `FF70A_SET_2`, `SS950_SET_1`, `SS950_SET2`. In PyCharm, paste them under `<project_directory>/truss_project_analysis/_unit_libraries`.

### Running a simulation
To start a simulation, the Python file `start_simulation.py` needs to be run.
In this file, insert the following command:
```
simulate_parametric_truss_statically(truss_problem_path, simulation_name)
```
* The `truss_problem_path` argument is the path (relative to folder `truss_analysis_project`) leading to the file that describes the spring structure you want to simulate). How to create such a file is described in the section **Creating a CSV file describing the spring structure**.
* The `simulation_name` argument is simply the name of the folder under which the simulation results will be saved (relative to the `truss_analysis_project/_result_data`) directory. If the name already exists, then the results will be stored under the `simulation_name` string appended with `-1`, or `-2`, etc.

Some examples are already available in the `start_simulation.py` file.

Many settings can be tuned before running a simulation. They can be changed at the start of the file `<project_directory>/truss_analysis_project/src/simulation/truss_simulator.py`. The most used parameters are:
* `reference_load_parameter` and `radius` from dictionary `_default_static_solver_parameters`. Small values (around 0.01) = finer solution, but slow. Larger values (around 0.5)
= coarser solution, but faster.
* `i_max` from dictionary `_default_static_solver_parameters` which is the maximum number of iterations. A large value is completely fine, but some simulations might take some time to finish. To make sure, they don't last too long, a smaller value of `i_max` can abort them in a clean manner (results are stored normally after the max number of iterations has been reached).
* parameters from dictionary `_graphics_options` to control what is generated from the simulation results (plots, animations, plot settings, animation settings (fps, total number of frames, etc),  etc).
### Creating a CSV file describing the spring structure


The CSV file looks like this:

~~~~
PARAMETERS
<parameter name>, <parameter value>
<parameter name>, <parameter value>
...
NODES
<node index>, <x>, <y>, <constrained x>, <constrained y>
<node index>, <x>, <y>, <constrained x>, <constrained y>
...
SPRING ELEMENTS
<node index>, <node index>, <stiffness>, [natural length]
<node index>, <node index>, <stiffness>, [natural length]
...
FLEXIBLE HINGE ELEMENTS
<node index>, <node index>, <node index>, <torsional stiffness>, [natural angle]
<node index>, <node index>, <node index>, <torsional stiffness>, [natural angle]
...
NONLINEAR ELEMENTS
<node index>, <node index>, BEZIER(u1=<u1>|f1=<f1>|u2=<u2>|f2=<f2>|u3=<u3>|f3=<f3>), [natural length]
<node index>, <node index>, BEZIER(u1=<u1>|f1=<f1>|u2=<u2>|f2=<f2>|u3=<u3>|f3=<f3>), [natural length]
...
UNITS ELEMENTS
<node index>, <node index>, (unit library; unit name; [perturbation]; [fit model]), [natural length]
<node index>, <node index>, (unit library; unit name; [perturbation]; [fit model]), [natural length]
...
STATIC LOADMAP
<node index>, <force>, <direction>, [max displacement]
<node index>, <force>, <direction>, [max displacement]
...
~~~~
NB:
* `<...>`: required field
* `[...]`: optional field
#### The `PARAMETERS` section
The `PARAMETERS` section serves to define some parameters that can be used to in the next sections. To define a parameter, a line with the following structure is added to the section `PARAMETERS`:\
`<parameter name>, <parameter value>`.
* `<parameter name>` is the name of the parameter (character string **without** quotes).
* `<parameter value>` is the value of the parameter. It can be either a float or a string (**with** simple quotes)
#### The `NODES` section
The `NODES` section serves to define the nodes composing the spring structure, by specifying their index,
their coordinates and the fact they are constrained or not. To define a node, a line with the following structure is added to the section `NODES`:\
`<node index>, <x>, <y>, <constrained x>, <constrained y>`.
* `<node index>` is a positive integer (`0`, `1`, `2`, ...) representing the index of the node being defined. Two nodes cannot have the same index.
When nodes are defined, indices cannot be skipped (if there are four nodes in total, the indices must be `0`, `1`, `2` and `3`). The order in which the nodes are defined does not matter.
* `<x>` is the horizontal coordinate of the node (float).
* `<y>` is the vertical coordinate of the node (float).
* `<constrained x>` is either
  * `1` if the node cannot move horizontally,
  * `0` if the node is free to move horizontally.
* `<constrained y>` is either
  * `1` if the node cannot move vertically,
  * `0` if the node is free to move vertically.
#### The `SPRING ELEMENTS` section
The `SPRING ELEMENTS` section serves to define the topology of the spring structure, by specifying which nodes are connected by linear springs.
When two nodes are connected, the stiffness of the spring connecting them must be specified, and optionally the natural length of that spring.
If no natural length is provided, the natural length is automatically set to the distance between the nodes connected by the spring.
To define a spring element, a line with the following structure is added to the section ` SPRING ELEMENTS`:\
`<node index>, <node index>, <stiffness>, [natural length]`.
* `<node index>` is the index of one of the node connected to the spring.
* `<node index>` is the index of the other node connected to the spring.
* `<stiffness>` is the stiffness of the spring connecting both nodes (positive float).
* `[natural length]` is the natural length of the spring connecting both nodes (positive float). 
It is an optional parameter; if not provided the natural length of the spring element will automatically be set to the distance between both nodes as created in the `NODES` section.
#### The `FLEXIBLE HINGE ELEMENTS` section
The `FLEXIBLE HINGE ELEMENTS` section serves to define the flexible hinges (torsional springs) in the spring structure, by specifying three nodes A, B and C, which together, define the angle ABC (B is the vertex of the angle). More precisely, the angle ABC is the angle by which the segment BA must rotate counterclockwise (about B) to align with segment BC. The angle is always between 0 and 2π. Thereby, extra loops must be avoided, because angles larger than 2π cannot be obtained.
When a flexible hinge is defined the torsional stiffness must be specified, and optionally the natural angle of that hinge (in radians).
If no natural angle is provided, the natural angle is automatically set to the angle defined by the three specified nodes.
To define a flexible hinge element, a line with the following structure is added to the section `FLEXIBLE HINGE ELEMENTS`:\
`<node index>, <node index>, <node index>, <torsional stiffness>, [natural angle]`.
* `<node index>` is the index of node A.
* `<node index>` is the index of node B.
* `<node index>` is the index of node C.
* `<torsional stiffness>` is the torsional stiffness of the hinge (positive float).
* `[natural angle]` is the natural angle of the hinge in radians (float in [0, 2π[). 
It is an optional parameter; if not provided the natural angle of the flexible hinge element will automatically be set to the angle defined by nodes A, B and C as created in the `NODES` section.
#### The `STATIC LOADMAP` section
The `STATIC LOADMAP` section serves to set the forces applied on some specific nodes along a specific direction (horizontal or vertical).
To define a horizontal or vertical force on a node, a line with the following structure is added to the section `STATIC LOADMAP`:\
`<node index>, <force>, <direction>, [max displacement]`.
* `<node index>` is index of the node on which the force is applied.
* `<force>` the signed amplitude of the force (negative or positive float).
* `<direction>` is either:
  * `H` for a horizontal force,
  * `V` for a vertical force.
* `[max displacement]` is the maximum displacement of the specified node along the specified direction (positive float).
Above this value, the simulation will finish (even if the force has not been reached yet). It is an optional parameter; if not provided, there is no maximum displacement at which the simulation can finish prematurely (it is the same as setting the maximum displacement to a value close to infinity).

#### Example: two-spring-and-one-hinge structure

![test](https://github.com/ducarme/truss-analysis/blob/main/truss_analysis_project/_media/two_spring_one_hinge.png)

_two_spring_one_hinge.csv_
~~~~
PARAMETERS
hinge_stiffness, 3.0
NODES
0, 0.0, 0.0, 1, 1
1, 6.0, 0.0, 1, 1
2, 3.0, 4.0, 1, 0
SPRING ELEMENTS
0, 2, 1.0, 5.0
1, 2, 1.0, 5.0
FLEXIBLE HINGE ELEMENTS
0, 2, 1, hinge_stiffness
LOADMAP
2, -4.0, V
~~~~
Note that the natural lengths of the springs could have been omitted since the distance between the connected nodes as defined in the `NODES` section is already 5.0.
Therefore, _two_spring_one_hinge.csv_ could have been this (see herebelow), which defines the exact same spring structure:
~~~~
PARAMETERS
hinge_stiffness, 3.0
NODES
0, 0.0, 0.0, 1, 1
1, 6.0, 0.0, 1, 1
2, 3.0, 4.0, 1, 0
SPRING ELEMENTS
0, 2, 1.0
1, 2, 1.0
FLEXIBLE HINGE ELEMENTS
0, 2, 1, hinge_stiffness
LOADMAP
2, -4.0, V
~~~~





