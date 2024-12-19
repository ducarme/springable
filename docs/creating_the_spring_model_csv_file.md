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
ROTATION SPRINGS
<node index>-<node index>-<node index>, <mechanical behavior>, [natural angle]
<node index>-<node index>-<node index>, <mechanical behavior>, [natural angle]
...
AREA SPRINGS
<node index>-<node index>-<node index>-..., <mechanical behavior>, [natural area]
<node index>-<node index>-<node index>-..., <mechanical behavior>, [natural area]
...
LINE SPRINGS
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

Legend: `<...>`: required field, `[...]`: optional field

Each section is described in details herein below.

+ [The `PARAMETERS` section](#the-parameters-section)
+ [The `NODES` section](#the-nodes-section)
+ [The `SPRINGS` section](#the-springs-section)
+ [The `ROTATION SPRINGS` section](#the-rotation-springs-section)
+ [The `AREA SPRINGS` section](#the-area-springs-section)
+ [The `LINE SPRINGS` section](#the-line-springs-section)
+ [The `LOADING` section](#the-loading-section)
+ [A complete example](#a-complete-example)
+ [Additional notes](#additional-notes)


#### The `PARAMETERS` section
The `PARAMETERS` section serves to define some parameters that can be used to in the next sections. To define a parameter, a line with the following structure is added to the section `PARAMETERS`:

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

#### The `ROTATION SPRINGS` section
The `ROTATION SPRINGS` section serves to define **rotation springs**
(also known as [torsion springs](https://en.wikipedia.org/wiki/Torsion_spring)), that is, springs whose elastic energy is a function of an angle. They are useful when modelling mechanical systems involving elastic bending, such as flexures for example.
Those springs are defined by specifying **three nodes** A, B and C,
which together, define the angle ABC (B is the vertex of the angle). More precisely, the angle ABC is the angle by which
the segment BA must rotate counterclockwise (about B) to align with segment BC. The angle is always between 0 and 2π.

Along with its three nodes, the **mechanical behavior** must be specified, and optionally the natural angle of the rotation
spring (in radians). If no natural angle is provided, the natural angle is automatically set to the angle defined by the
three specified nodes. The mechanical behavior describes its intrinsic (torque)-(angle-change) relation. It can be a linear behavior
(the rotation spring follows [Hooke's law](https://en.wikipedia.org/wiki/Hooke%27s_law)) or a nonlinear one
(see section [Specifying a nonlinear mechanical behavior](#specifying-a-nonlinear-mechanical-behavior)).

To define a rotation spring, a line with the following structure is added to the section `ROTATION SPRINGS`:\
`<node index>-<node index>-<node index>, <mechanical behavior>, [natural angle]`.
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
>A linear rotation spring is defined. The torque it creates will be determined by the difference between the angle 021
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

!!! note
    To define an area spring associated to a **[polygon with holes](https://en.wikipedia.org/wiki/Polygon_with_holes)**,
    please refer to [Area springs with holes](#area-spring-with-holes) paragraph in the [Advanced topics](#advanced-topics).


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

#### The `LINE SPRINGS` section
The `LINE SPRINGS` section serves to define **line springs**, that is, springs whose elastic energy is a function of their [polygonal chain](https://en.wikipedia.org/wiki/Polygonal_chain)'s length.
They are useful when modelling mechanical systems involving cable-driven actuation or [pulleys](https://en.wikipedia.org/wiki/Pulley).
Those springs are defined by specifying **n nodes** (n>=2), which together define a polygonal chain. More precisely, the nodes are the vertices listed sequentially that form the chain.
The sequence of nodes does not need to (but can) be closed (first and last node can be different or identical).

Along with its n nodes, the **mechanical behavior** must be specified, and optionally the natural length of the line
spring. If no natural length is provided, the natural length is automatically set to the length defined by the
n specified nodes. The mechanical behavior describes its intrinsic tension-displacement relation. It can be a linear behavior
(the line spring follows [Hooke's law](https://en.wikipedia.org/wiki/Hooke%27s_law)) or a nonlinear one
(see section [Specifying a nonlinear mechanical behavior](#specifying-a-nonlinear-mechanical-behavior)).

To define a line spring, a line with the following structure is added to the section `LINE SPRINGS`:\
`<node index>-<node index>-<node index>-..., <mechanical behavior>, [natural length]`.
* `<node index>` is the index of a first node that form the polygonal chain,
* `<node index>` is the index of the second node, following the first node along the chain,
* `<node index>` is the index of the third node following the second node along the chain,
* `<node index>` ... etc.
* `<mechanical behavior>` is the mechanical behavior of the line spring. To specify a **linear** line spring,
the mechanical behavior is simply the **spring constant** (positive float), that is the slope of its tension-displacement curve.
* `[natural length]` is the natural length of the line spring (float). 
It is an optional parameter; if not provided the natural length of the line spring will automatically be set to the
length defined by the n nodes as created in the `NODES` section.


Example:
```csv
LINE SPRINGS
0-2-1, 1.0
```
>A linear line spring is defined. The tension it creates will be determined by the difference between its current and natural lengths.
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

!!! note
    More complex loading can be specified (preloading, multiple loading steps, blocking nodes).
    Please refer to [Complex loading descriptions](#complex-loading-descriptions) paragraph in the [Advanced topics](#advanced-topics) for more details.


#### A complete example
This example describes a spring structure composed of two inclined linear longitudinal springs connected in the center,
and hinging through a linear rotation spring.
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