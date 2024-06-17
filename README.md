# springable


The Python package `springable` allows you to simulate how assemblies of springs deform under loads.
By accounting for any geometrical changes (as large as they may be), the simulation allows you to explore the richness
of nonlinear mechanics, beyond the (boring) linear regime.

The implementation of the package is geared towards understanding how spring assemblies lead to mechanical behaviors
ranging from simple monotonic responses to complex, highly-nonlinear ones, such as snapping instabilities, sequencing,
buckling, symmetry-breaking or restabilization.

In its core, `springable` deals with **springs**, that we define as any entity that can store [elastic energy](https://en.wikipedia.org/wiki/Elastic_energy). **Springs** therefore include longitudinal springs (compression and extension), rotation springs (bending), area springs (useful to model fluids and pneumatic loading) and more (*see gallery*).
## Installation

Use `pip` to install. In the terminal simply enter

    pip install springable

and you are good to go!

It is supported on **Python 3.11 and above**.
## How to use


### Running a simulation
To start a simulation, create a Python file named - let's say - `my_first_simulation.py`, then run it.

```python
"""
my_first_simulation.py
Example to learn how to use the package springable
"""
import springable.simulation as ss

ss.simulate_model(model_path='my_spring_model.csv', save_dir='my_simulation_result')
```

The function `ss.simulate_model()` takes in two necessary arguments:
* The `model_path` argument is the path leading to the [CSV file](https://en.wikipedia.org/wiki/Comma-separated_values)
that describes the model you want to simulate.
How to create such a file is described in the section **Creating a CSV file describing the spring model**.
* The `save_dir` argument is simply the name of the folder under which the simulation results will be saved.

CSV file examples that describe spring assemblies are already available
[here on GitHub](https://github.com/ducarme/springable/tree/main/examples) for inspiration or to download.

A simple linear spring under a tensile load can be described as follows.
```csv
# my_simple_spring_model.csv

PARAMETERS
stiffness, 1.0
NODES
0, 0.0, 0.0, 1, 1
1, 1.0, 0.0, 0, 1
SPRINGS
0-1, stiffness
LOADING
1, X, 2.0
```

Many settings can be tuned before running a simulation. See section **Configuring simulation settings** for more details.




### Creating a CSV file describing the spring model


The CSV file describing a valid spring model is specified as follows:

~~~~
PARAMETERS
<parameter name>, <parameter value>
<parameter name>, <parameter value>
...
NODES
<node index>, <x>, <y>, <constrained x>, <constrained y>
<node index>, <x>, <y>, <constrained x>, <constrained y>
...
SPRINGS
<node index>-<node index>, <mechanical behavior>, [natural length]
<node index>-<node index>, <mechanical behavior>, [natural length]
...
ROTATION SPRINGS
<node index>-<node index>-<node index>, <mechanical behavior>, [natural angle]
<node index>-<node index>-<node index>, <mechanical behavior>, [natural angle]
...
AREA SPRINGS
<node index>-<node index>-<node index>-..., <mechanical behavior>, [natural area]
<node index>-<node index>-<node index>-..., <mechanical behavior>, [natural area]
...
LOADING
<node index>, <direction>, <force>, [max displacement]
<node index>, <direction>, <force>, [max displacement]
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

Example:
```csv
PARAMETERS
width, 2.0
height, 1.0
stiffness, 7.3
```
*Three parameters - `width`, `height` and `stiffness` - are defined and set to values `2.0`, `1.0` and `7.3` respectively.*

#### The `NODES` section
The `NODES` section serves to define the nodes composing the spring assembly, by specifying their index,
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

Example:
```csv
NODES
0, -width/2, 0.0, 1, 1
1, +width/2, 0.0, 1, 1
2, 0.0, height, 1, 0
```
*Three nodes labelled `0`, `1` and `2` are defined at coordinates `(-width/2, 0.0)`, `(+width/2, 0.0)`, `(0.0, height)`
respectively. Nodes `0` and `1` are constrained horizontally and vertically, while node `2` is constrained horizontally
but free to move vertically.*

#### The `SPRINGS` section
The `SPRINGS` section serves to define **longitudinal springs**. Each longitudinal spring is defined by the
**two nodes** it connects, a **mechanical behavior**, and optionally its **natural length**,
that is the length it has if not constrained nor loaded. If no natural length is provided, the natural length is
automatically set to the distance between the nodes connected by the spring.

The mechanical behavior describes its intrinsic axial force-displacement relation. It can be a linear behavior
(the spring follows [the Hooke's law](https://en.wikipedia.org/wiki/Hooke%27s_law)) or a nonlinear one
(see section **Specifying a Nonlinear Mechanical Behavior**).

To define a longitudinal spring, a line with the following structure is added to the section `SPRINGS`:\
`<node index>-<node index>, <mechanical behavior>, [natural length]`.
* `<node index>` is the index of one of the node connected to the spring.
* `<node index>` is the index of the other node connected to the spring.
* `<mechanical behavior>` is the axial mechanical behavior of the spring. To specify a **linear** longitudinal spring,
the mechanical behavior is simply the **spring constant** (positive float).
* `[natural length]` is the natural length of the spring connecting both nodes (positive float). 
It is an optional parameter; if not provided the natural length of the spring element will automatically be set to the
distance between both nodes as created in the `NODES` section.

Example:
```csv
SPRINGS
0-2, stiffness
1-2, stiffness
```
*Two linear longitudinal springs are defined. Both are characterized by the spring constant value `stiffness`.
No natural length was provided, so their natural length will be automatically set to the distance between nodes `0`
and `2`, and between nodes `1` and `2` as defined in the section `NODES`, respectively.*

#### The `ROTATION SPRINGS` section
The `ROTATION SPRINGS` section serves to define **rotation springs**
(also known as [torsion springs](https://en.wikipedia.org/wiki/Torsion_spring)), by specifying **three nodes** A, B and C,
which together, define the angle ABC (B is the vertex of the angle). More precisely, the angle ABC is the angle by which
the segment BA must rotate counterclockwise (about B) to align with segment BC. The angle is always between 0 and 2π.
Along with its three nodes, the **mechanical behavior** must be specified, and optionally the natural angle of the rotation
spring (in radians). If no natural angle is provided, the natural angle is automatically set to the angle defined by the
three specified nodes.
To define a rotation spring, a line with the following structure is added to the section `ROTATION SPRINGS`:\
`<node index>-<node index>-<node index>, <nechanical behavior>, [natural angle]`.
* `<node index>` is the index of node A.
* `<node index>` is the index of node B.
* `<node index>` is the index of node C.
* `<mechanical behavior>` is the angular mechanical behavior of the rotation spring. To specify a **linear** rotation spring,
the mechanical behavior is simply the **spring constant** (positive float), that is the slope of its (torque)-(angle-change) curve.
* `[natural angle]` is the natural angle of the rotation spring in radians (float in [0, 2π[). 
It is an optional parameter; if not provided the natural angle of the rotation spring will automatically be set to the
angle defined by nodes A, B and C as created in the `NODES` section.

Example:
```csv
ROTATION SPRINGS
0-2-1, 1.5, PI/2
```
*A linear rotation spring is defined. The torque it creates will be determined by the difference between the angle 021
(vertex at node `2`) and its natural angle `PI/2` (90 deg). The angle-difference versus torque relation is defined
by the spring constant set to `1.5`.
Note that if no natural angle was specified, the natural angle would have been automatically set to the angle defined by
the nodes `0`, `1` and `2` as defined in the section `NODES`.*

#### The `LOADING` section
The `LOADING` section serves to set the forces applied on some specific nodes along a specific direction (horizontal or vertical).
To define a horizontal or vertical force on a node, a line with the following structure is added to the section `LOADING`:\
`<node index>, <direction>, <force>, [max displacement]`.
* `<node index>` is the index of the node on which the force is applied.
* `<direction>` is either:
  * `X` for a horizontal force,
  * `Y` for a vertical force.
* `<force>` the signed amplitude of the force (negative or positive float)
* `[max displacement]` is the maximum displacement of the specified node along the specified direction (positive float).
Beyond this value, the simulation will finish (even if the force has not been reached yet).
It is an optional parameter; if not provided, there is no maximum displacement at which the simulation can finish
prematurely (it is the same as setting the maximum displacement to a value close to infinity).

Example:
```csv
LOADING
2, Y, -10.0, -3.0
```
*A force is applied on node `2`, along the `Y`-direction (vertical).
The magnitude of the force is `-10` (it is a negative value, so the force points downwards).
The maximum displacement is set to `-3.0`, meaning that if node `2` is displaced downward by more that `3.0`,
the simulation is assumed to have completed.*


Complete example
```csv
# spring model example (this is a comment)

PARAMETERS
width, 2.0
height, 1.0
stiffness, 7.3

NODES
0, -width/2, 0.0, 1, 1
1, +width/2, 0.0, 1, 1
2, 0.0, height, 1, 0

SPRINGS
0-2, stiffness
1-2, stiffness
ROTATION SPRINGS
0-2-1, 1.5, PI/2

LOADING
2, Y, -10.0, -3.0
```

#### Additional notes
* Empty lines have no semantic meaning. Adding/removing some will not change the spring model.
* `#` is used to indicate a line comment. Each line starting with `#` will be ignored when reading the file.
* Parameters can be combined in mathematical expression in all sections but `PARAMETERS`. Supported operations include
`(...)`, `+`, `-`, `*`, `/`, `SIN(...)` (sine), `COS(...)` (cosine), `TAN(...)` (tangent), `SQRT(...)` (square root). 
Value π can be used without defining it in the section `PARAMETERS` with the keyword `PI`.
* If your spring assembly does not include a certain type of spring, feel free to leave the corresponding section empty (header only)
or to omit it completely (no header and no content).


### Configuring simulation settings
Many settings can be tuned before running a simulation. They fall into two categories: **solver settings**
or **graphics settings**.

Solver settings only affect the solver (that is the part responsible to solve the nonlinear equations,
by computing all the equilibrium points), while graphics settings determines what is generated to visualize a result
(previously computed by the solver), and how it is going to look like.

All default solver and graphics settings along with useful documentation are listed in the files
`default_solver_settings.toml` and `default_graphics_settings.toml` respectively. They are available
[here on GitHub](https://github.com/ducarme/springable).


When you wish to change a setting (let's say a solver setting),
create a [TOML file](https://toml.io/en/), that is a text file saved with extension *.toml*. You can use the NotePad (Windows)
or TextEdit (MacOS) to do that, for example. The file will look similar to this:
```toml
# custom_solver_settings.toml
radius = 0.01
reference_load_parameter = 0.01
```
*Low values for `radius` and `reference_load_parameter` can be used to refine the solution,
at the cost of increasing the solving duration. Default values are 0.05 and 0.05, respectively.*

To use these custom solver settings, use the path to `custom_solver_settings.toml`
as an extra argument of the `ss.simulate_model()` function, as follows:

```python
"""
my_first_simulation.py
Example to learn how to use the package springable
"""
import springable.simulation as ss

ss.simulate_model(model_path='my_spring_model.csv',
                  save_dir='my_simulation_result',
                  solver_settings_path='custom_solver_settings.toml')
```


Similarly, when you wish to modify a graphics setting, create another TOML file and include the settings you wish to modify
```toml
# custom_graphics_settings.toml
[animation_options]
nb_frames = 240
fps = 60

[plot_options]
show_snapping_arrows = true
drive_mode = "force"
```
*Animation settings `nb_frames` and `fps` determine the number of frames and the frame rate (in frame per second) of the animation showing
the spring assembly deforming. Plot settings `show_snapping_arrows = true` combined with `drive_mode = "force"` means that
you want to indicate with arrows the (potential) snapping transitions under controlled force in the force-displacement plot.
To indicate, snapping transitions under controlled displacement use `show_snapping_arrows = true` combined with `drive_mode = "displacement"`
instead.*

To use these custom graphics settings, use the path to `custom_graphics_settings.toml`
as an extra argument of the `ss.simulate_model()` function, as follows:

```python
"""
my_first_simulation.py
Example to learn how to use the package springable
"""
import springable.simulation as ss

ss.simulate_model(model_path='my_spring_model.csv',
                  save_dir='my_simulation_result',
                  solver_settings_path='custom_solver_settings.toml',
                  graphics_settings_path='custom_graphics_settings.toml')
```




#### Additional notes
* A custom settings file does not need to contain all the possible settings; just include the one you wish to modify.
* Graphics settings are divided into 4 sections of settings (indicated by `[...]` in TOML files):
  * general options (determines _what_ should be generated and directly shown (drawing, animation, plot))
  * plot options (determines _how_ plots will look like)
  * animation options (determines _how_ animations will look like)
  * assembly appearance (determines _how_ the spring assembly will be depicted)




