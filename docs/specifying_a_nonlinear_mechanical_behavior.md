In `springable`, each spring (longitudinal, angular, etc) has its own intrinsic mechanical behavior.
An intrinsic mechanical behavior is fully characterized by a **generalized force-displacement curve**.
For a longitudinal spring, that curve will be interpreted as a *force-displacement* curve. For a angular spring, as a
*torque-angle change* curve. For an area spring, as a *2d pressure-area change* curve. Etc.

!!! info
    Mathematically speaking, the generalized force $f$ is defined as the derivative of the elastic energy with respect to the *measure* $\alpha$
    of the spring. The measure $\alpha$ is the *length* for a longitudinal spring, the *angle* for a angular spring, the *area* for an area spring, etc.
    The generalized displacement $u$ is defined as the difference between the current measure $\alpha$ and the *natural* measure $\alpha_0$, that is, the measure
    of the spring in its natural configuration (wherein no force is generated and no elastic energy is stored).


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
<p align="center"><img src="https://github.com/user-attachments/assets/ee0ccc8b-a02e-418b-be0f-9bb68738fb0b" height="320px"/></p>

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


## Linear behavior
For a **linear** generalized force-displacement curve $f=ku$, where $k$ is the spring constant.

`LINEAR(k=<value>)` or `<value>`

Example: `..., LINEAR(k=2.0)` or equivalently `..., 2.0`

> A spring with a linear behavior characterized by a spring constant `2.0` is defined.
>
> ![](https://github.com/user-attachments/assets/0dfeb1ca-3857-4d09-8bf5-36228dbc2a85){ width="300"}
> ///caption
> ///

    





??? question "Units?"
    The unit of $k$ should be the unit of the generalized force $f$
    divided by the unit of the generalized displacement $u$.


## Logarithmic behavior
A **logarithmic** behavior is defined by a generalized force-displacement curve given by
$f=k\alpha_0\ln(\alpha/\alpha_0)$, $u=\alpha-\alpha_0$. It is useful to prevent springs from having a zero measure
(longitudinal springs with zero length, angular springs with zero angle, etc),
as the generalized force approaches infinity as the measure gets close to zero.

`LOGARITHMIC(k=<spring constant>)`


Example: `... , LOGARITHMIC(k=2.0)`

> A spring with a logarithmic behavior characterized by $k$ equals `2.0` is defined.
> 
> ![](https://github.com/user-attachments/assets/c53f1391-81bb-467c-bf56-4a6ebf2f5306){ width="300"}
> /// caption
> ///




??? question "Isn't the parameter $\alpha_0$ missing?"
    It seems like we are missing the parameter $\alpha_0$ in the specification (only $k$ is provided). This is not a problem; remember that the value of $\alpha_0$ is
    in fact automatically set to the value of the spring measure in the state defined by [the `NODES` section](creating_the_spring_model_csv_file.md/#the-nodes-section), when not provided.
    If you want to assign a value for $\alpha_0$, you can do it by adding a comma followed by the $\alpha_0$ value.

    Example: `... , LOGARITHMIC(k=2.0), 2.5`

    > A spring is defined with a behavior of type `LOGARITHMIC` with `k=2.0` and a natural measure `2.5`.

??? question "Units?"
    The unit of $k$ should be the unit of the generalized force $f$
    divided by the unit of the generalized displacement $u$.



## Bezier behavior
A **Bezier** behavior is described by a generalized force-displacement curve defined as a [Bezier curve](https://en.wikipedia.org/wiki/B%C3%A9zier_curve).
More precisely, the $f-u$ curve is given by $f(t)=\sum_{i=1}^n f_i b_{i,n}(t)$ and $u(t)=\sum_{i=1}^n u_i b_{i,n}(t)$, where $u_i$ and $f_i$ describe
the coordinates of [control points](https://en.wikipedia.org/wiki/Control_point_(mathematics)),
$b_{i,n}$ are the [Bernstein polynomials](https://en.wikipedia.org/wiki/Bernstein_polynomial) of degree $n$,
and $t$ is the curve parameter that runs from 0 to 1.

`BEZIER(u_i=[<value_11>; <value_12>; ...;<value_1n>]; f_i=[<value_21>; <value_22>; ...; <value_2n>])`

    
Example: `..., BEZIER(u_i=[1.0;1.2;3.0];f_i=[2.0;-3.0;2.4])`

> A spring is defined with a generalized force-displacement relation described as a Bezier curve of degree 3
> with control points (0, 0), (`1.0`, `2.0`), (`1.2`, `-3.0`) and (`3.0`, `2.4`).
> 
> ![](https://github.com/user-attachments/assets/34446134-1988-4691-a261-865f25290b22){ width="300", align=right}
> ///caption
> ///


??? question "Units?"
    The unit of each $u_i$ should be the unit of the generalized displacement $u$.
    The unit of each $f_i$ should be the unit of the generalized force $f$.

!!! note
    For a generalized displacement $u$ larger than $u_n$, the corresponding generalized force is extrapolated linearly based on the slope at the last control point.
    Also, the generalized force-displacement relation is defined for negative generalized displacements $u<0$ by imposing the symmetry $f(u<0)=-f(|u|)$.




## Bezier2 behavior
A **Bezier2** behavior is the same as a [Bezier behavior](#bezier-behavior).
The only difference is that, unlike a Bezier behavior,
a Bezier2 behavior is allowed to define a curve that *curves back*,
meaning that at a certain generalized displacement value, multiple force values can exist.

`BEZIER2(u_i=[<value_11>; <value_12>; ...;<value_1n>]; f_i=[<value_21>; <value_22>; ...; <value_2n>])`


Example: `..., BEZIER2(u_i=[2.5;-1.0;2.0];f_i=[2.0;-1.0;1.0])`
> A spring is defined with a generalized force-displacement relation described as a Bezier curve of degree 3
> with control points (0, 0), (`2.5`, `2.0`), (`-1.0`, `-1.0`) and (`2.0`, `1.0`).
> This curve curves back; it cannot be described a function $f(u)$.
> 
> ![](https://github.com/user-attachments/assets/8f5dae45-7a2e-46bd-afe1-ff1ef5814a8c){ width="300"}
> ///caption
> ///




??? question "Units?"
    The unit of each $u_i$ should be the unit of the generalized displacement $u$.
    The unit of each $f_i$ should be the unit of the generalized force $f$.

!!! note "Important"
    Due to implementation details, the way the curve folds and unfolds should respect some conditions. First, the curve cannot have [cusps](https://en.wikipedia.org/wiki/Cusp_(singularity)).
    Second, the tangent vector along the curve can never point vertically upward, as one moves along the curve from the origin
    (it is perfectly fine for the tangent to point vertically downward).
 
    Also, a Bezier2 behavior introduces an extra [degree of freedom (DOF)](https://en.wikipedia.org/wiki/Degrees_of_freedom_(mechanics))
    in order to disambiguate the state of the spring, as the generalized displacement $u$ is not enough to fully define its state.
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
> 
>![](https://github.com/user-attachments/assets/66de0d4b-bd8a-4463-9c92-ef6190981e31){ width="300"}
> ///caption
> ///





??? question "Units?"
    The unit of each $k_i$ should be the unit of the generalized force $f$
    divided by the unit of the generalized displacement $u$.
    The unit of each $u_i$ and the unit of $u_s$ should be the unit of the generalized displacement $u$.

!!! note
    The quantity $u_s$ must be positive and lower than $\min((u_1-0.0), (u_2-u_1)/2, ..., (u_{n-1}-u_{n-2})/2)$.
    Also, the generalized force-displacement relation is defined for negative generalized displacements $u<0$
    by imposing the symmetry $f(u<0)=-f(|u|)$.

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
>
> ![](https://github.com/user-attachments/assets/31c61352-53b9-4ab8-b190-0f4d326e4333){ width="300"}
> ///caption
> ///

??? question "Units?"
    The unit of each $u_i$ should be the unit of the generalized displacement $u$.
    The unit of each $f_i$ should be the unit of the generalized force $f$.
    The smoothing parameter $\epsilon$ has no dimension.

!!! note
    The generalized force-displacement relation is defined for negative generalized displacements $u<0$
    by imposing the symmetry $f(u<0)=-f(|u|)$.

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
> This curve curves back; it cannot be described a function $f(u)$.
> 
> ![](https://github.com/user-attachments/assets/2e44e2ca-29eb-4298-88d8-537482379d6a){ width="300"}
> ///caption
> ///

??? question "Units?"
    The unit of each $u_i$ should be the unit of the generalized displacement $u$.
    The unit of each $f_i$ should be the unit of the generalized force $f$.
    The smoothing parameter $\epsilon$ has no dimension.

!!! note "Important"
    Due to implementation details, the way the curve folds and unfolds should respect some conditions. First, the curve cannot have [cusps](https://en.wikipedia.org/wiki/Cusp_(singularity)).
    Second, the tangent vector along the curve can never point vertically upward, as one moves along the curve from the origin
    (it is perfectly fine for the tangent to point vertically downward).
 
    Also, a zigzag2 behavior introduces an extra [degree of freedom (DOF)](https://en.wikipedia.org/wiki/Degrees_of_freedom_(mechanics))
    in order to disambiguate the state of the spring, as the generalized displacement $u$ is not enough to fully define its state.
    Using a **zigzag** behavior instead when the curve does not curve back helps keep the number of DOFs low.


## Contact behavior
A **contact** behavior is described by a generalized force-displacement curve that
is perfectly zero for large displacement $u \geq \alpha_\Delta - \alpha_0$ and yields a relatively strong repulsion force for low displacement $u<\alpha_\Delta - \alpha_0$.
More precisely, the generalized force-displacement curve is given by

$$
f(u) = \begin{cases}
    0&\text{if $u \ge \alpha_\Delta - \alpha_0$}\\
    -f_0\left(\dfrac{\alpha_\Delta - \alpha_0-u}{u_\text{c}}\right)^3&\text{if $u < \alpha_\Delta - \alpha_0$},
\end{cases}
$$

where $f_0$ is the magnitude of the generalized force $f$ when the generalized displacement $u$ is decreased by $u_\text{c}$ from $\alpha_\Delta - \alpha_0$.
It is useful to model contact, as a relatively significant force is generated but only below a certain threshold.

`CONTACT(f0=<value1>; uc=<value2>; delta=<value3>)`


Example: `..., CONTACT(f0=3.0; uc=0.01; delta=0.5)`
> A spring is defined with a contact behavior. When the measure $\alpha$ of the spring becomes less than `delta=0.5`,
> an increasingly strong repulsion generalized force is generated, reaching -`3.0` when the measure is decreased further by `0.01`, that is, when $\alpha=$ `0.5`.
> 
<!-- > ![](https://github.com/user-attachments/assets/a8f17fe5-cd4f-41fe-b7a1-65842625f7d9){width="300"}
> ///caption
> /// -->

??? question "Units?"
    The unit of $u_c$ should be the unit of the generalized displacement $u$.
    The unit of $f_0$ should be the unit of the generalized force $f$.

!!! note
    For a spring with a contact behavior, the natural measure $\alpha_0$ does not have any effect, as the force produced by the contact behavior solely depends on the measure $\alpha$; it is independent of the natural measure $\alpha_0$.

## Isothermal behavior

An **isothermal** behavior is described by a generalized force-displacement curve that respects the pressure-volume
relation of an [ideal gas](https://en.wikipedia.org/wiki/Ideal_gas_law) during an isothermal process (constant temperature).
That relation can be expressed as follows,

$$
p - p_0 = nRT_0 \left( 1/V - 1/V_0\right),
$$

where $p_0$ is the ambient pressure, $V_0$ is the volume of the gas at ambient pressure,
$T_0$ is the temperature of the gas (constant), $p$ is the current pressure, $V$ is the current volume,
$n$ is the amount of substance (constant), $R$ is the gas constant.

More precisely, the generalized force $f$ plays the role of the pressure difference, $f=p_0-p$,
while the generalized displacement plays the role of volume change, $u=V-V_0$. The measure $\alpha$ and
natural measure $\alpha_0$ are mapped to $\alpha=V$ and $\alpha_0=V_0$, respectively. The $f-u$ curve is therefore given by

$$
f = nRT_0\frac{u}{(u+\alpha_0)\alpha_0}.
$$

`ISOTHERMAL(n=<n_value>; R=<R_value>; T0=<T0_value>)`

Example: `..., ISOTHERMAL(n=1.0; R=8.3; T0=300)`
> A spring is defined with an isothermal behavior. Its generalized force-displacement relation follows the behavior
> of `1` mole of an ideal gas at constant temperature $T_0$=`300`K.
> 
> ![](https://github.com/user-attachments/assets/b9d2d7d8-b502-4249-9218-7e79cbf44ebd){width="300"}
> ///caption
> ///

??? question "Isn't the parameter $\alpha_0=V_0$ missing?"
    It seems like we are missing the parameter $\alpha_0$, describing the volume $V_0$ at ambient pressure $p_0$, in the specification.
    This is not a problem; remember that the value of $\alpha_0$ is
    in fact automatically set to the value of the spring measure in the state defined by [the `NODES` section](creating_the_spring_model_csv_file.md/#the-nodes-section),
    when not provided.
    If you want to assign a value for $\alpha_0$, you can do it by adding a comma followed by the $\alpha_0$ value.

    Example: `... , ISOTHERMAL(n=1.0; R=8.3; T0=300), 1.0`

    > A spring is defined with an isothermal behavior. Its generalized force-displacement relation follows the behavior
    > of `1` mole of an ideal gas at constant temperature $T_0$=`300`K. Its *volume*/measure at ambient pressure is `1.0`.

??? question "Units?"
    The unit of the quantity $nRT_0$ should be a unit of energy; more precisely,
    it should be the unit of the generalized force $f$
    multiplied by the unit of the generalized displacement $u$.

!!! note
    A negative generalized force $f<0$ corresponds to a compressed state
    (the pressure is greater than the ambient pressure, $p>p_0$). A positive generalized force $f>0$ corresponds to
    a *vacuumed* state (the pressure is smaller than the ambient pressure, $p<p_0$).




## Isentropic behavior
An **isentropic** behavior is described by a generalized force-displacement curve that respects the pressure-volume
relation of an [ideal gas](https://en.wikipedia.org/wiki/Ideal_gas_law) during an
[isentropic process](https://en.wikipedia.org/wiki/Isentropic_process) (constant entropy).
That relation can be expressed as follows,

$$
 p-p_0 = nRT_0\left(\dfrac1V\left(\dfrac{V_0}{V}\right)^{\gamma-1} - \dfrac1{V_0}\right),
$$

where $p_0$ is the ambient pressure, $V_0$ is the volume of the gas at ambient pressure,
$T_0$ is the temperature of the gas at ambient pressure, $p$ is the current pressure, $V$ is the current volume,
$n$ is the amount of substance (constant), $R$ is the gas constant and $\gamma$ is the
[heat capacity ratio](https://en.wikipedia.org/wiki/Heat_capacity_ratio) (constant).

More precisely, the generalized force $f$ plays the role of the pressure difference, $f=p_0-p$,
while the generalized displacement plays the role of volume change, $u=V-V_0$. The measure $\alpha$ and
natural measure $\alpha_0$ are mapped to $\alpha=V$ and $\alpha_0=V_0$, respectively. The $f-u$ curve is therefore given by

$$
f(u) = nRT_0\left(\dfrac1{\alpha_0} - \dfrac1{u+\alpha_0}\left(\dfrac{\alpha_0}{u+\alpha_0}\right)^{\gamma-1}\right).
$$

`ISENTROPIC(n=<n_value>; R=<R_value>; T0=<T0_value>; gamma=<gamma_value>)`

Example: `..., ISENTROPIC(n=1.0; R=8.3; T0=300; gamma=1.4)`
> A spring is defined with an isentropic behavior. Its generalized force-displacement relation follows the behavior
> of `1` mole of an ideal gas initially at $T_0$=`300`K, with $\gamma$=`1.4` (heat capacity ratio of air), at constant entropy.
>
> ![](https://github.com/user-attachments/assets/986d1eaf-9619-4750-924a-542f27a4c68c){width="300"}
> ///caption
> ///

??? question "Isn't the parameter $\alpha_0=V_0$ missing?"
    It seems like we are missing the parameter $\alpha_0$, describing the volume $V_0$ at ambient pressure $p_0$, in the specification.
    This is not a problem; remember that the value of $\alpha_0$ is
    in fact automatically set to the value of the spring measure in the state defined by [the `NODES` section](creating_the_spring_model_csv_file.md/#the-nodes-section),
    when not provided.
    If you want to assign a value for $\alpha_0$, you can do it by adding a comma followed by the $\alpha_0$ value.

    Example: `... , ISENTROPIC(n=1.0; R=8.3; T0=300; gamma=1.4), 1.0`

    > A spring is defined with an isentropic behavior. Its generalized force-displacement relation follows the behavior
    > of `1` mole of an ideal gas initially at $T_0$=`300`K, with $\gamma$=`1.4`, at constant entropy.
    > Its *volume*/measure at ambient pressure is `1.0`.

??? question "Units?"
    The unit of the quantity $nRT_0$ should be a unit of energy;
    more precisely, it should be the unit of the generalized force $f$
    multiplied by the unit of the generalized displacement $u$. The heat capacity ratio $\gamma$ has no dimension.

!!! note
    A negative generalized force $f<0$ corresponds to a compressed state
    (the pressure is greater than the ambient pressure, $p>p_0$). A positive generalized force $f>0$ corresponds to
    a *vacuumed* state (the pressure is smaller than the ambient pressure, $p<p_0$).




## Additional notes

* A nonlinear behavior can be saved in a separate CSV file and used in a model file using
`FROMFILE(<nonlinear behavior csv file>)`

Example:
`..., FROMFILE('custom_nonlinear_behavior.csv')`

where the `custom_nonlinear_behavior.csv` is for example:
`BEZIER2(u_i=[0.21; -0.1; 3.14]; f_i=[1.0; -2.0; +3.0])`

The file path to the behavior is relative to the working directory, that is, the directory from where the script is run. If the CSV behavior file lives in a subdirectory `path/to/behavior.csv` relative to the working directory, then we would use
`FROMFILE('path'; 'to'; 'behavior.csv')`

To specify a CSV behavior file that would live in a subdirectory `relative/path/to/behavior.csv` relative to the CSV model file instead, we can use the keyword `HERE` that encodes the directory where the CSV model file lives (relative to the working directory) as follows:
`FROMFILE(HERE; 'relative; 'path'; 'to'; 'behavior.csv')`

*Nonlinear behaviors can be interactively tuned and created using the behavior creation graphical interface, which can be started by running the following Python script
```
from springable.behavior_creation import start
```

start()