
[![PyPI version](https://img.shields.io/pypi/v/springable)](https://pypi.org/project/springable/)
![GitHub License](https://img.shields.io/github/license/ducarme/springable)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ducarme/springable/blob/main/docs/examples/example01_getting_started/example01_getting_started.ipynb)
[![Research Group](https://img.shields.io/badge/Research%20group-Soft%20robotic%20matter%20%40%20AMOLF-67CD00)](https://overvelde.com/)
[![Made with love in Amsterdam (NL)](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20in-Amsterdam%20(NL)-ece4fc)](https://amolf.nl/)

<p align="center"><img src="https://github.com/user-attachments/assets/a9c967f2-dac9-4ce9-ad31-a8a403fd74bf" height="180"/></p>
<p align="center">Library for nonlinear spring simulations</p>

![](https://github.com/user-attachments/assets/110da33e-266a-4fca-a0f0-3ac793804bed)


**Springable** is a library for **simulations of nonlinear springs**. It allows you to simulate how mechanical structures made out of (non)linear springs deform when subject to forces.
By accounting for any geometrical changes (as large as they may be), the simulation allows you to explore the richness
of nonlinear mechanics, beyond the (boring) linear regime.

The implementation of the library is geared towards understanding how spring assemblies lead to mechanical behaviors
ranging from simple monotonic responses to complex, highly-nonlinear ones, such as snapping instabilities, sequencing,
buckling, symmetry-breaking or restabilization.

In its core, `springable` deals with **springs**, that we define as any entity that can store [elastic energy](https://en.wikipedia.org/wiki/Elastic_energy).
*Springs* therefore include longitudinal springs (compression and extension),
angular springs (bending), area springs (useful to model fluids and pneumatic loading), path springs (useful to model cable-driven systems), and more!
On top of that, the library allows you to define the energy potential of each individual spring to make them intrinsically linear or nonlinear, thereby generating a whole ecosystem of springs, ready to be assembled and simulated!

   
## Installation

Use `pip` to install. In the terminal simply enter

    python -m pip install springable

and you are good to go!

It is supported on **Python 3.10 and above**.

## Don't want to install it right now? Try the Online Notebook
Try `springable` online, without any installation in an [interactive online notebook](https://colab.research.google.com/github/ducarme/springable/blob/main/docs/examples/example01_getting_started/example01_getting_started.ipynb)

## Getting started with examples
After you've installed the library, run the following python script.
It will show you a lot of examples that will help you create and simulate your own spring models!

```python
from springable.discover import show_examples

show_examples()
```


## How to use

### Running a simulation
To start a simulation, we first create the file 
that will describe the spring model we want to simulate. To do that, we create a simple
[CSV file](https://en.wikipedia.org/wiki/Comma-separated_values),
(a text file saved with extension *.csv*), that looks like this for example:
```csv
# my_spring_model.csv

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
This file defines a spring structure composed of only one horizontal spring, clamped on the left and loaded in tension from the right.

How to read or make such a file is described in the paragraph [Creating a CSV file describing the spring model](#creating-a-csv-file-describing-the-spring-model).
Many CSV file examples that describe spring structures are already available
[here on GitHub](https://github.com/ducarme/springable/tree/main/src/springable/data/examples_csv_models) for inspiration or to download.

Next, we create a Python script (a text file saved with the extension *.py*), with the following content

```python
"""
Python script 'my_first_simulation.py'
Example to learn how to use the package springable
"""
import springable.simulation as ss

ss.simulate_model(model_path='my_spring_model.csv', save_dir='my_simulation_result')
```
and save it under the name - let's say - `my_first_spring_simulation.py`

The function `ss.simulate_model()` takes in two arguments:
* The `model_path` argument is the path leading to the [CSV file](https://en.wikipedia.org/wiki/Comma-separated_values)
that describes the model you want to simulate. In this case, we used the CSV file `my_spring_model.csv` that we have just created.
* The `save_dir` argument is simply the name of the folder under which the simulation results will be saved. It should not exist yet; it will be created automatically when we run the script.
It is an optional argument, if not specified, a folder will be created automatically in the working directory to store the result files.

Finally, we run the Python script. This can be done in the terminal by simply executing

    python my_first_spring_simulation.py



>[!TIP]
> Many settings can be tuned before running a simulation. See paragraph [Configuring simulation settings](#configuring-simulation-settings) for more details.


## Documentation
The full documentation is available on our website [https://ducarme.github.io/springable/](https://ducarme.github.io/springable/)

Herein below, the essentials!

### Creating a CSV file describing the spring model


The CSV file describing a valid spring model is specified as follows:

~~~~
PARAMETERS
<parameter name>, <parameter value>, [range]
<parameter name>, <parameter value>, [range]
...
NODES
<node index>, <x>, <y>, <constrained x>, <constrained y>
<node index>, <x>, <y>, <constrained x>, <constrained y>
...
SPRINGS
<node index>-<node index>, <mechanical behavior>, [natural length]
<node index>-<node index>, <mechanical behavior>, [natural length]
...
ANGULAR SPRINGS
<node index>-<node index>-<node index>, <mechanical behavior>, [natural angle]
<node index>-<node index>-<node index>, <mechanical behavior>, [natural angle]
...
AREA SPRINGS
<node index>-<node index>-<node index>-..., <mechanical behavior>, [natural area]
<node index>-<node index>-<node index>-..., <mechanical behavior>, [natural area]
...
PATH SPRINGS
<node index>-<node index>-<node index>-..., <mechanical behavior>, [natural length]
<node index>-<node index>-<node index>-..., <mechanical behavior>, [natural length]
...
DISTANCE SPRINGS
<node index>-<node index>-<node index>, <mechanical behavior>, [natural distance]
<node index>-<node index>-<node index>, <mechanical behavior>, [natural distance]
...
LOADING
<node index>, <direction>, <force>, [max displacement]
<node index>, <direction>, <force>, [max displacement]
...
~~~~
NB:
* `<...>`: required field
* `[...]`: optional field

Each section is described in details herein below.

+ [The `PARAMETERS` section](#the-parameters-section)
+ [The `NODES` section](#the-nodes-section)
+ [The `SPRINGS` section](#the-springs-section)
+ [The `ANGULAR SPRINGS` section](#the-angular-springs-section)
+ [The `AREA SPRINGS` section](#the-area-springs-section)
+ [The `PATH SPRINGS` section](#the-line-springs-section)
+ [The `LOADING` section](#the-loading-section)
+ [A complete example](#a-complete-example)
+ [Additional notes](#additional-notes)


#### The `PARAMETERS` section
The `PARAMETERS` section serves to define some parameters that can be used to in the next sections. To define a parameter, a line with the following structure is added to the section `PARAMETERS`:\
`<parameter name>, <parameter value>`.
* `<parameter name>` is the name of the parameter (character string **without** quotes).
* `<parameter value>` is the value of the parameter. It can be either a float or a string (**with** simple quotes)
* `[range]` (optional) is a vector of possible values the parameter can have. This field is used only when [scanning the parameter space](#scanning-parameters). The range can be specified in two different ways:
  * either as a vector of n regularly-spaced values between two float. Syntax: `[low bound; high bound; n]`. Example: `radius, 2.1, [2.0; 5.0; 4]`.
  * either as a list of possible values. Syntax: `{value1; value2; value3; ...}`. Example: `radius, 2.1, {1.0; 7.0; 8.0; 2.0}`.

Example:
```csv
PARAMETERS
width, 2.0
height, 1.0
stiffness, 7.3
```
>Three parameters - `width`, `height` and `stiffness` - are defined
> and set to values `2.0`, `1.0` and `7.3` respectively.

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
>Three nodes labelled `0`, `1` and `2` are defined at coordinates `(-width/2, 0.0)`, `(+width/2, 0.0)`, `(0.0, height)`
respectively. Nodes `0` and `1` are constrained horizontally and vertically, while node `2` is constrained horizontally
but free to move vertically.

#### The `SPRINGS` section
The `SPRINGS` section serves to define **longitudinal springs**, that is, springs whose elastic energy is a function of their length.
Each longitudinal spring is defined by the
**two nodes** it connects, a **mechanical behavior**, and optionally its **natural length**,
that is the length it has if not constrained nor loaded. If no natural length is provided, the natural length is
automatically set to the distance between the nodes connected by the spring.

The mechanical behavior describes its intrinsic axial force-displacement relation. It can be a linear behavior
(the spring follows [Hooke's law](https://en.wikipedia.org/wiki/Hooke%27s_law)) or a nonlinear one
(see section [Specifying a nonlinear mechanical behavior](#specifying-a-nonlinear-mechanical-behavior)).

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
>Two linear longitudinal springs are defined. Both are characterized by the spring constant value `stiffness`.
No natural length was provided, so their natural length will be automatically set to the distance between nodes `0`
and `2`, and between nodes `1` and `2` as defined in the section `NODES`, respectively.

#### The `ANGULAR SPRINGS` section
The `ANGULAR SPRINGS` section serves to define **angular springs**
(also known as [torsion springs](https://en.wikipedia.org/wiki/Torsion_spring)), that is, springs whose elastic energy is a function of an angle. They are useful when modelling mechanical systems involving elastic bending, such as flexures for example.
Those springs are defined by specifying **three nodes** A, B and C,
which together, define the angle ABC (B is the vertex of the angle). More precisely, the angle ABC is the angle by which
the segment BA must rotate counterclockwise (about B) to align with segment BC. The angle is always between 0 and 2π.

Along with its three nodes, the **mechanical behavior** must be specified, and optionally the natural angle of the angular
spring (in radians). If no natural angle is provided, the natural angle is automatically set to the angle defined by the
three specified nodes. The mechanical behavior describes its intrinsic (torque)-(angle-change) relation. It can be a linear behavior
(the angular spring follows [Hooke's law](https://en.wikipedia.org/wiki/Hooke%27s_law)) or a nonlinear one
(see section [Specifying a nonlinear mechanical behavior](#specifying-a-nonlinear-mechanical-behavior)).

To define a angular spring, a line with the following structure is added to the section `ANGULAR SPRINGS`:\
`<node index>-<node index>-<node index>, <mechanical behavior>, [natural angle]`.
* `<node index>` is the index of node A.
* `<node index>` is the index of node B.
* `<node index>` is the index of node C.
* `<mechanical behavior>` is the angular mechanical behavior of the angular spring. To specify a **linear** angular spring,
the mechanical behavior is simply the **spring constant** (positive float), that is the slope of its (torque)-(angle-change) curve.
* `[natural angle]` is the natural angle of the angular spring in radians (float in [0, 2π[). 
It is an optional parameter; if not provided the natural angle of the angular spring will automatically be set to the
angle defined by nodes A, B and C as created in the `NODES` section.

Example:
```csv
ANGULAR SPRINGS
0-2-1, 1.5, PI/2
```
>A linear angular spring is defined. The torque it creates will be determined by the difference between the angle 021
(vertex at node `2`) and its natural angle `PI/2` (90 deg). The angle-difference versus torque relation is defined
by the spring constant set to `1.5`.
Note that if no natural angle was specified, the natural angle would have been automatically set to the angle defined by
the nodes `0`, `1` and `2` as defined in the section `NODES`.


#### The `AREA SPRINGS` section
The `AREA SPRINGS` section serves to define **area springs**, that is, springs whose elastic energy is a function of their area. They are useful when modelling mechanical systems involving fluids and pneumatic or hydraulic components.
Those springs are defined by specifying **n nodes** (n>=3), which together define the area of a
[simple polygon](https://en.wikipedia.org/wiki/Simple_polygon). More precisely, the nodes are the vertices listed sequentially that form the single closed boundary of the polygon.
The sequence of nodes should *not* be ending with the starting node. The polygon can be convex or concave, but not self-intersecting. The boundary of the polygon can be specified by listing the vertices clockwise or counterclockwise.

Along with its n nodes, the **mechanical behavior** must be specified, and optionally the natural area of the area
spring. If no natural area is provided, the natural area is automatically set to the area defined by the
n specified nodes. The mechanical behavior describes its intrinsic (2d-pressure)-(area-change) relation. It can be a linear behavior
(the area spring follows [Hooke's law](https://en.wikipedia.org/wiki/Hooke%27s_law)) or a nonlinear one
(see section [Specifying a nonlinear mechanical behavior](#specifying-a-nonlinear-mechanical-behavior)).

To define an area spring, a line with the following structure is added to the section `AREA SPRINGS`:\
`<node index>-<node index>-<node index>-..., <mechanical behavior>, [natural area]`.
* `<node index>` is the index of a first node that form the boundary of the polygon,
* `<node index>` is the index of the second node, following the first node along the boundary (clockwise or counter-clockwise),
* `<node index>` is the index of the third node following the second node along the boundary (clockwise or counter-clockwise),
* `<node index>` ... etc.
* `<mechanical behavior>` is the areal mechanical behavior of the area spring. To specify a **linear** area spring,
the mechanical behavior is simply the **spring constant** (positive float), that is the slope of its (2d-pressure)-(area-change) curve.
* `[natural area]` is the natural area of the area spring (float). 
It is an optional parameter; if not provided the natural area of the area spring will automatically be set to the
area defined by the n nodes as created in the `NODES` section.

> [!NOTE]
> To define an area spring associated to a **[polygon with holes](https://en.wikipedia.org/wiki/Polygon_with_holes)**,
please refer to the [full documentation](#documentation) for more info.

Example:
```csv
AREA SPRINGS
0-2-1, 3.0
```
>A linear area spring is defined. The 2d-pressure it creates will be determined by the difference between the area of the polygon 0210
and its natural area. The area-change versus 2d-pressure relation is defined
by the spring constant set to `3.0`.
Here, no natural area was provided, so the natural area will be automatically set to
the area of the polygon defined by the nodes `0`, `2`, and `1` as defined in the section `NODES`.

#### The `PATH SPRINGS` section
The `PATH SPRINGS` section serves to define **path springs**, that is, springs whose elastic energy is a function of their [polygonal chain](https://en.wikipedia.org/wiki/Polygonal_chain)'s length.
They are useful when modelling mechanical systems involving cable-driven actuation or [pulleys](https://en.wikipedia.org/wiki/Pulley).
Those springs are defined by specifying **n nodes** (n>=2), which together define a polygonal chain. More precisely, the nodes are the vertices listed sequentially that form the chain.
The sequence of nodes does not need to (but can) be closed (first and last node can be different or identical).

Along with its n nodes, the **mechanical behavior** must be specified, and optionally the natural length of the line
spring. If no natural length is provided, the natural length is automatically set to the length defined by the
n specified nodes. The mechanical behavior describes its intrinsic tension-displacement relation. It can be a linear behavior
(the path spring follows [Hooke's law](https://en.wikipedia.org/wiki/Hooke%27s_law)) or a nonlinear one
(see section [Specifying a nonlinear mechanical behavior](#specifying-a-nonlinear-mechanical-behavior)).

To define a path spring, a line with the following structure is added to the section `PATH SPRINGS`:\
`<node index>-<node index>-<node index>-..., <mechanical behavior>, [natural length]`.
* `<node index>` is the index of a first node that form the polygonal chain,
* `<node index>` is the index of the second node, following the first node along the chain,
* `<node index>` is the index of the third node following the second node along the chain,
* `<node index>` ... etc.
* `<mechanical behavior>` is the mechanical behavior of the path spring. To specify a **linear** path spring,
the mechanical behavior is simply the **spring constant** (positive float), that is the slope of its tension-displacement curve.
* `[natural length]` is the natural length of the path spring (float). 
It is an optional parameter; if not provided the natural length of the path spring will automatically be set to the
length defined by the n nodes as created in the `NODES` section.


Example:
```csv
PATH SPRINGS
0-2-1, 1.0
```
>A linear path spring is defined. The tension it creates will be determined by the difference between its current and natural lengths.
The displacement versus tension relation is defined by the spring constant set to `1.0`.
Here, no natural length was provided, so the natural length will be automatically set to
the length of the polygonal chain defined by the nodes `0`, `2`, and `1` as defined in the section `NODES`.


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
>A force is applied on node `2`, along the `Y`-direction (vertical).
The magnitude of the force is `-10` (it is a negative value, so the force points downwards).
The maximum displacement is set to `-3.0`, meaning that if node `2` is displaced downward by more that `3.0`,
the simulation is assumed to have completed.

> [!NOTE]
> More complex loading can be specified (preloading, multiple loading steps, blocking nodes).
Please refer to the [full documentation](#documentation) for more info

#### A complete example
This example describes a spring structure composed of two inclined linear longitudinal springs connected in the center,
and hinging through a linear angular spring.
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
ANGULAR SPRINGS
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

### Specifying a nonlinear mechanical behavior
In `springable`, each spring (longitudinal, angular, etc) has its own intrinsic mechanical behavior.
An intrinsic mechanical behavior is fully characterized by a **generalized force-displacement curve**.
For a longitudinal spring, that curve will be interpreted as a *force-displacement* curve. For a angular spring, as a
*torque-angle change* curve. For an area spring, as a *2d pressure-area change* curve. Etc.

> [!NOTE]
> Mathematically speaking, the generalized force $F$ is defined as the derivative of the elastic energy with respect to the *measure* $\alpha$
> of the spring. The measure $\alpha$ is the *length* for a longitudinal spring, the *angle* for a angular spring, the *area* for an area spring, etc.
> The generalized displacement $U$ is defined as the difference between the current measure $\alpha$ and the *natural* measure $\alpha_0$, that is, the measure
> of the spring in its default configuration.


Multiple types of intrinsic mechanical behavior can be specified in the [spring model CSV file](#creating-a-csv-file-describing-the-spring-model).
The faster way to get familiar with them is by running the **behavior creation graphical user interface**. To do that create the following python script

```python
"""
Python script 'start_behavior_creation_interface.py'
"""
from springable.behavior_creation import start

start()
```
and run it (in the terminal, that would be done using `python start_behavior_creation_interface.py`).

A window named *Behavior creation* should pop up on your screen
<p align="center"><img src="https://github.com/user-attachments/assets/20f715f0-d1cb-405b-9760-65faf9dbd7ae" height="320px"/></p>

By playing around with the interface, you will be able to create various generalized force-displacement curves and generate the corresponding code to use in the
[spring model CSV file](#creating-a-csv-file-describing-the-spring-model). Herein below, more details are provided about the various types of mechanical behavior.

+ [Linear behavior](#linear-behavior)
+ [Logarithmic behavior](#logarithmic-behavior)
+ [Bezier behavior](#bezier-behavior)
+ [Bezier2 behavior](#bezier2-behavior)
+ [Piecewise behavior](#piecewise-behavior)
+ [Zigzag behavior](#zigzag-behavior)
+ [Zigzag2 behavior](#zigzag2-behavior)
+ [Contact behavior](#contact-behavior)
+ [Isothermal gas behavior](#isothermal-behavior)
+ [Isentropic gas behavior](#isentropic-behavior)
+ [Additional notes](#additional-notes-1)


#### Linear behavior
For a **linear** generalized force-displacement curve $F=kU$, where $k$ is the spring constant.

`LINEAR(k=<value>)` or `<value>`

Example: `..., LINEAR(k=2.0)` or equivalently `..., 2.0`
> A spring with a linear behavior characterized by a spring constant `2.0` is defined.

The unit of $k$ should be the unit of the generalized force $F$ divided by the unit of the generalized displacement $U$.

#### Logarithmic behavior
A **logarithmic** behavior is defined by a generalized force-displacement curve given by
$F=k\alpha_0\ln(\alpha/\alpha_0)$, $U=\alpha-\alpha_0$. It is useful to prevent springs from having a zero measure
(longitudinal springs with zero length, angular springs with zero angle, etc),
as the generalized force approaches infinity as the measure gets close to zero.

`LOGARITHMIC(k=<spring constant>)`

Example: `... , LOGARITHMIC(k=2.0)`
> A spring with a natural behavior characterized by $k$ equals `2.0` is defined.

It seems like we are missing the parameter $\alpha_0$ in the specification (only $k$ is provided). This is not a problem; remember that the value of $\alpha_0$ is
in fact automatically set to the value of the spring measure in the state defined by [the `NODES` section](#the-node-section), when not provided.
If you want to assign a value for $\alpha_0$, you can do it by adding a comma followed by the $\alpha_0$ value

Example: `... , LOGARITHMIC(k=2.0), 1.0`.

> A spring is defined with a behavior of type `LOGARITHMIC` with `k=2.0` and a natural measure `1.0`.

The unit of $k$ should be the unit of the generalized force $F$ divided by the unit of the generalized displacement $U$.

#### Bezier behavior
A **Bezier** behavior is described by a generalized force-displacement curve defined as a [Bezier curve](https://en.wikipedia.org/wiki/B%C3%A9zier_curve).
More precisely, the $F-U$ curve is given by $F(t)=\sum_{i=1}^n f_i b_{i,n}(t)$ and $U(t)=\sum_{i=1}^n u_i b_{i,n}(t)$, where $u_i$ and $f_i$ describe
the coordinates of [control points](https://en.wikipedia.org/wiki/Control_point_(mathematics)),
$b_{i,n}$ are the [Bernstein polynomials](https://en.wikipedia.org/wiki/Bernstein_polynomial) of degree $n$,
and $t$ is the curve parameter that runs from 0 to 1.

`BEZIER(u_i=[<value_11>; <value_12>; ...;<value_1n>]; f_i=[<value_21>; <value_22>; ...; <value_2n>])`

Example: `..., BEZIER(u_i=[1.0;1.2;3.0];f_i=[2.0;-3.0;2.4])`
> A spring is defined with a generalized force-displacement relation described as a Bezier curve of degree 3
> with control points (0, 0), (`1.0`, `2.0`), (`1.2`, `-3.0`) and (`3.0`, `2.4`).

The unit of each $u_i$ should be the unit of the generalized displacement $U$.
The unit of each $f_i$ should be the unit of the generalized force $F$.


> [!NOTE]
> For a generalized displacement $U$ larger than $u_n$, the corresponding generalized force is extrapolated linearly based on the slope at the last control point.
> Also, the generalized force-displacement relation is defined for negative generalized displacements $U<0$ by imposing the symmetry $F(U<0)=-F(|U|)$.

#### Bezier2 behavior
A **Bezier2** behavior is the same as a [Bezier behavior](#bezier-behavior).
The only difference is that, unlike a Bezier behavior,
a Bezier2 behavior is allowed to define a curve that *curves back*,
meaning that at a certain generalized displacement value, multiple force values can exist.

`BEZIER2(u_i=[<value_11>; <value_12>; ...;<value_1n>]; f_i=[<value_21>; <value_22>; ...; <value_2n>])`

Example: `..., BEZIER2(u_i=[2.5;-1.0;2.0];f_i=[2.0;-1.0;1.0])`
> A spring is defined with a generalized force-displacement relation described as a Bezier curve of degree 3
> with control points (0, 0), (`2.5`, `2.0`), (`-1.0`, `-1.0`) and (`2.0`, `1.0`).
> This curve curves back; it cannot be described a function $F(U)$.

> [!IMPORTANT]
> Due to implementation details, the way the curve folds and unfolds should respect some conditions. First, the curve cannot have [cusps](https://en.wikipedia.org/wiki/Cusp_(singularity)).
> Second, the tangent vector along the curve can never point vertically upward, as one moves along the curve from the origin
> (it is perfectly fine for the tangent to point vertically downward).
> 
> Also, a Bezier2 behavior introduces an extra [degree of freedom (DOF)](https://en.wikipedia.org/wiki/Degrees_of_freedom_(mechanics))
> in order to disambiguate the state of the spring, as the generalized displacement $U$ is not enough to fully define its state.
> Using a **Bezier** behavior instead when the curve does not curve back helps keep the number of DOFs low.

#### Piecewise behavior
A **piecewise** behavior is defined by a [piecewise linear function](https://en.wikipedia.org/wiki/Piecewise_linear_function)
whose corners have been smoothed using a quadratic function. A piecewise curve composed of $n>1$ segments is described by
$n$ slopes $k_i$ and $n-1$ transition points $u_i$ at which the segments would connect. The quantity $u_s$ describes how smooth
each corner must be. More precisely, around each corner $i$ located at $u_i$, the curve is given by a quadratic function on the interval
$\[u_i-u_s, u_i+u_s\]$, instead of linear segments.
The smoothing quadratic functions are tuned to be [C1 continuous](https://en.wikipedia.org/wiki/Smoothness)
with the segments they connect.

`PIECEWISE(k_i=[<value_11>; <value_12>; ...;<value_1n>]; u_i=[<value_21>; <value_22>; ...; <value_2(n-1)>]; us=<value>])`

Example: `..., PIECEWISE(k_i=[1.0;-1.0;2.0]; u_i=[1.0;2.0]; us=0.2)`
> A spring is defined with a generalized force-displacement relation described as a smoothed piecewise linear curve
> composed of three segments with slopes `1.0`, `-1.0` and `2.0`,
> with the transition between the first and second segment at `1.0`
> and the transition between the second and third segment at `2.0`. The amount of smoothing is set to `0.2`.

> [!NOTE]
> The quantity $u_s$ must be positive and lower than $\min((2u_1-0.0), (u_2-u_1), ..., (u_{n-1}-u_{n-2}))/2$.
> Also, the generalized force-displacement relation is defined for negative generalized displacements $U<0$
> by imposing the symmetry $F(U<0)=-F(|U|)$.

#### Zigzag behavior
A **zigzag** behavior is described by a generalized force-displacement curve defined as a
[polygonal chain](https://en.wikipedia.org/wiki/Polygonal_chain) with smoothed corners.
It is specified by providing the control points' coordinates $(u_i, f_i)$
(coordinates of the corners of the non-smoothed polygonal chain),
and a smoothing parameter $0<\epsilon<1$.

`ZIGZAG(u_i=[<value_11>; <value_12>; ...;<value_1n>]; f_i=[<value_21>; <value_22>; ...; <value_2n>]; epsilon=<value>)`

Example: `..., ZIGZAG(u_i=[1.0; 2.0; 3.0]; f_i=[1.0; -0.5; 2.0]; epsilon=0.8)`
> A spring is defined with a generalized force-displacement relation described as a smoothed zigzag curve
> defined by 4 control points (0, 0), (`1.0`, `1.0`), (`2.0`, `-0.5`) and (`3.0`, `2.0`). The corners are smoothed using
> $\epsilon=$`0.8`

> [!NOTE]
> The generalized force-displacement relation is defined for negative generalized displacements $U<0$
> by imposing the symmetry $F(U<0)=-F(|U|)$.

#### Zigzag2 behavior
A **zigzag2** behavior is the same as a [zigzag behavior](#zigzag-behavior).
The only difference is that, unlike a zigzag behavior,
a zigzag2 behavior is allowed to define a curve that *curves back*,
meaning that at a certain generalized displacement value, multiple force values can exist.

`ZIGZAG2(u_i=[<value_11>; <value_12>; ...;<value_1n>]; f_i=[<value_21>; <value_22>; ...; <value_2n>]; epsilon=<value>)`

Example: `..., ZIGZAG2(u_i=[2.0; 1.0; 3.0]; f_i=[2.0; 0.0; 1.0]; epsilon=0.4)`
> A spring is defined with a generalized force-displacement relation described as a smoothed zigzag curve
> with control points (0, 0), (`2.0`, `2.0`), (`1.0`, `0.0`) and (`3.0`, `1.0`).
> The corners are smoothed using $\epsilon=$`0.4`
> This curve curves back; it cannot be described a function $F(U)$.

> [!IMPORTANT]
> Due to implementation details, the way the curve folds and unfolds should respect some conditions. First, the curve cannot have [cusps](https://en.wikipedia.org/wiki/Cusp_(singularity)).
> Second, the tangent vector along the curve can never point vertically upward, as one moves along the curve from the origin
> (it is perfectly fine for the tangent to point vertically downward).
> 
> Also, a zigzag2 behavior introduces an extra [degree of freedom (DOF)](https://en.wikipedia.org/wiki/Degrees_of_freedom_(mechanics))
> in order to disambiguate the state of the spring, as the generalized displacement $U$ is not enough to fully define its state.
> Using a **zigzag** behavior instead when the curve does not curve back helps keep the number of DOFs low.


#### Contact behavior
#### Isothermal behavior
#### Isentropic behavior
#### Additional notes


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
```
*A lower value for `radius` can be used to refine the solution,
at the cost of increasing the solving duration. Default values is 0.05.*

To use these custom solver settings, use the path to `custom_solver_settings.toml`
as an extra argument of the `ss.simulate_model()` function, as follows:

```python
"""
my_first_simulation.py
Example to learn how to use the package springable
"""
import springable.simulation as ss

ss.simulate_model(model_path='my_spring_model.csv', save_dir='my_simulation_result',
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

ss.simulate_model(model_path='my_spring_model.csv', save_dir='my_simulation_result',
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




