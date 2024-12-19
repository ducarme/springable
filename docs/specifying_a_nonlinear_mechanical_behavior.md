In `springable`, each spring (longitudinal, rotational, etc) has its own intrinsic mechanical behavior.
An intrinsic mechanical behavior is fully characterized by a **generalized force-displacement curve**.
For a longitudinal spring, that curve will be interpreted as a *force-displacement* curve. For a rotational spring, as a
*torque-angle change* curve. For an area spring, as a *2d pressure-area change* curve. Etc.

!!! info
    Mathematically speaking, the generalized force $F$ is defined as the derivative of the elastic energy with respect to the *measure* $\alpha$
    of the spring. The measure $\alpha$ is the *length* for a longitudinal spring, the *angle* for a rotation spring, the *area* for an area spring, etc.
    The generalized displacement $U$ is defined as the difference between the current measure $\alpha$ and the *natural* measure $\alpha_0$, that is, the measure
    of the spring in its default configuration.


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
+ [Logarithm behavior](#logarithm-behavior)
+ [Bezier behavior](#bezier-behavior)
+ [Bezier2 behavior](#bezier2-behavior)
+ [Piecewise behavior](#piecewise-behavior)
+ [Zigzag behavior](#zigzag-behavior)
+ [Zigzag2 behavior](#zigzag2-behavior)
+ [Contact behavior](#contact-behavior)
+ [Isothermic gas behavior](#isothermic-behavior)
+ [Isentropic gas behavior](#isentropic-behavior)
+ [Additional notes](#additional-notes-1)


## Linear behavior
For a **linear** generalized force-displacement curve $F=kU$, where $k$ is the spring constant.

`LINEAR(k=<value>)` or `<value>`

Example: `..., LINEAR(k=2.0)` or equivalently `..., 2.0`
> A spring with a linear behavior characterized by a spring constant `2.0` is defined.

The unit of $k$ should be the unit of the generalized force $F$ divided by the unit of the generalized displacement $U$.

## Logarithm behavior
A **logarithm** behavior is defined by a generalized force-displacement curve given by
$F=k\alpha_0\ln(\alpha/\alpha_0)$, $U=\alpha-\alpha_0$. It is useful to prevent springs from having a zero measure
(longitudinal springs with zero length, rotational springs with zero angle, etc),
as the generalized force approaches infinity as the measure gets close to zero.

`LOGARITHM(k=<spring constant>)`

Example: `... , LOGARITHM(k=2.0)`
> A spring with a natural behavior characterized by $k$ equals `2.0` is defined.

The unit of $k$ should be the unit of the generalized force $F$ divided by the unit of the generalized displacement $U$.


??? question "Isn't the parameter $\alpha_0$ missing?"
    It seems like we are missing the parameter $\alpha_0$ in the specification (only $k$ is provided). This is not a problem; remember that the value of $\alpha_0$ is
    in fact automatically set to the value of the spring measure in the state defined by [the `NODES` section](creating_the_spring_model_csv_file.md/#the-nodes-section), when not provided.
    If you want to assign a value for $\alpha_0$, you can do it by adding a comma followed by the $\alpha_0$ value.

    Example: `... , LOGARITHM(k=2.0), 1.0`

    > A spring is defined with a behavior of type `LOGARITHM` with `k=2.0` and a natural measure `1.0`.


## Bezier behavior
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

!!! note
    For a generalized displacement $U$ larger than $u_n$, the corresponding generalized force is extrapolated linearly based on the slope at the last control point.
    Also, the generalized force-displacement relation is defined for negative generalized displacements $U<0$ by imposing the symmetry $F(U<0)=-F(|U|)$.

## Bezier2 behavior
A **Bezier2** behavior is the same as a [Bezier behavior](#bezier-behavior).
The only difference is that, unlike a Bezier behavior,
a Bezier2 behavior is allowed to define a curve that *curves back*,
meaning that at a certain generalized displacement value, multiple force values can exist.

`BEZIER2(u_i=[<value_11>; <value_12>; ...;<value_1n>]; f_i=[<value_21>; <value_22>; ...; <value_2n>])`

Example: `..., BEZIER2(u_i=[2.5;-1.0;2.0];f_i=[2.0;-1.0;1.0])`
> A spring is defined with a generalized force-displacement relation described as a Bezier curve of degree 3
> with control points (0, 0), (`2.5`, `2.0`), (`-1.0`, `-1.0`) and (`2.0`, `1.0`).
> This curve curves back; it cannot be described a function $F(U)$.

!!! note "Important"
    Due to implementation details, the way the curve folds and unfolds should respect some conditions. First, the curve cannot have [cusps](https://en.wikipedia.org/wiki/Cusp_(singularity)).
    Second, the tangent vector along the curve can never point vertically upward, as one moves along the curve from the origin
    (it is perfectly fine for the tangent to point vertically downward).
 
    Also, a Bezier2 behavior introduces an extra [degree of freedom (DOF)](https://en.wikipedia.org/wiki/Degrees_of_freedom_(mechanics))
    in order to disambiguate the state of the spring, as the generalized displacement $U$ is not enough to fully define its state.
    Using a **Bezier** behavior instead when the curve does not curve back helps keep the number of DOFs low.

## Piecewise behavior
A **piecewise** behavior is defined by a [piecewise linear function](https://en.wikipedia.org/wiki/Piecewise_linear_function)
whose corners have been smoothed using quadratic functions. A piecewise curve composed of $n>1$ segments is described by
$n$ slopes $k_i$ and $n-1$ transition points $u_i$ at which the segments would connect. The quantity $u_s$ describes how smooth
each corner must be. More precisely, around each corner $i$ located at $u_i$, the curve is given by a quadratic function on the interval
$\left[u_i-u_s, u_i+u_s\right]$, instead of linear segments.
The smoothing quadratic functions are tuned to be [C1 continuous](https://en.wikipedia.org/wiki/Smoothness)
with the segments they connect.

`PIECEWISE(k_i=[<value_11>; <value_12>; ...;<value_1n>]; u_i=[<value_21>; <value_22>; ...; <value_2(n-1)>]; us=<value>])`


Example: `..., PIECEWISE(k_i=[1.0;-1.0;2.0]; u_i=[1.0;2.0]; us=0.2)`

> A spring is defined with a generalized force-displacement relation described as a smoothed piecewise linear curve
> composed of three segments with slopes `1.0`, `-1.0` and `2.0`,
> with the transition between the first and second segment at `1.0`
> and the transition between the second and third segment at `2.0`. The amount of smoothing is set to `0.2`.

!!! note
    The quantity $u_s$ must be positive and lower than $\min((u_1-0.0), (u_2-u_1), ..., (u_{n-1}-u_{n-2}))$.
    Also, the generalized force-displacement relation is defined for negative generalized displacements $U<0$
    by imposing the symmetry $F(U<0)=-F(|U|)$.

## Zigzag behavior
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

!!! note
    The generalized force-displacement relation is defined for negative generalized displacements $U<0$
    by imposing the symmetry $F(U<0)=-F(|U|)$.

## Zigzag2 behavior
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

!!! note "Important"
    Due to implementation details, the way the curve folds and unfolds should respect some conditions. First, the curve cannot have [cusps](https://en.wikipedia.org/wiki/Cusp_(singularity)).
    Second, the tangent vector along the curve can never point vertically upward, as one moves along the curve from the origin
    (it is perfectly fine for the tangent to point vertically downward).
 
    Also, a zigzag2 behavior introduces an extra [degree of freedom (DOF)](https://en.wikipedia.org/wiki/Degrees_of_freedom_(mechanics))
    in order to disambiguate the state of the spring, as the generalized displacement $U$ is not enough to fully define its state.
    Using a **zigzag** behavior instead when the curve does not curve back helps keep the number of DOFs low.


## Contact behavior
A **contact** behavior is described by a generalized force-displacement curve that
is perfectly zero for large displacement $U \geq u_c$ and yields a relatively strong repulsion force for low displacement $U<u_c$.
More precisely, the generalized force-displacement curve is given by

$$
F(U) = \begin{cases} 
          0 & \text{if} \quad U\geq u_c \\
          -f_0\left(\dfrac{u_c-U}{u_c}\right)^3 & \text{if} \quad U<u_c, 
       \end{cases}
$$

where $f_0$ is the magnitude of the generalized force $F$ when the generalized displacement $U$ is zero,
and $u_c$ is the threshold below which a nonzero force is generated by the behavior.
It is useful to model contact, as a relatively significant force is generated but only below a certain threshold.

`CONTACT(f0=<value1>; uc=<value2>)`


Example: `..., CONTACT(f0=5; uc=0.1)`
> A spring is defined with a contact behavior. When the measure $\alpha$ of the spring becomes less than `0.1`,
> a strong generalized repulsion force is generated, reaching -`5.0` when the measure $\alpha$ is zero.

!!! note "Important"
    The natural measure $\alpha_0$ of a spring defined with a contact behavior is 0.0 by default.
    **This stands in stark contrast with all other types of behaviors**, where the default natural measure $\alpha_0$
    is set to the measure $\alpha$ of the spring in the state described in the NODES section. In other words, when
    a spring is associated to a contact behavior and no natural measure is provided.
    The natural measure of that spring is set to 0. In the example herein above, since no natural measure is provided,
    the generalized displacement and the measure are the same, $U=\alpha-\alpha_0=\alpha$.

## Isothermic behavior
## Isentropic behavior
## Additional notes
