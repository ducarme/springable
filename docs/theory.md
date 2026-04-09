
The `springable` Python library is an implementation of the **flexel framework**. This framework defines a method to define and solve mechanical systems composed of energy-storing entities called **flexels**.

## Flexel
A flexel (**flex**ible **el**ement) can be seen as the generalization of a nonlinear spring. In a nonlinear spring, the axial force is found in two steps: first, its length is calculated from the end nodes’ coordinates; then, that length is used in an energy potential whose derivative gives the force. A flexel extends this idea in two main ways.
First, it replaces the notion of length with a more general geometric measure (noted $\alpha$), which can be any scalar value computed from a list of nodes’ coordinates (noted $z_i$); for example, an angle, an area, a total path length, or the distance from a point to a line.
Second, it allows for a wider range of energy potentials, supporting tunable and possibly multi-valued force–displacement curves. These curves can have multiple turning points and intersections, allowing flexels to singlehandedly encode information about their loading history, or capture snapping or countersnapping phenomenona, for instance.

![Graphical abstract](graphical_abstract.png){ align=center height=320px }

The flexel's ability to capture highly nonlinear behaviors without resorting to an assembly of lower-level components stands in stark contrast with other finite element methods. The versatility of flexels to represent different shapes and behaviors allows to model stretch, compression, bending, pneumatic actuation, cable-driven systems, contact... or interacting of those! As all flexels rely on a single formulation, they form an ecosystem which allows to build simpler models, that are easier to solve, interpret and use to gain insight into the mechanics or guide the design process.

## The mathematical formulation, the equations and the algorithm
The complete theory, the algorithm and all the derived equations are available in the research paper that introduces the flexel framework. The full journal proof is freely available [here](https://doi.org/10.1016/j.eml.2026.102476), in Extreme Mechanics Letter. The mathematical formulation and equations are developed in the supplementary information document (see it [here](https://ars.els-cdn.com/content/image/1-s2.0-S2352431626000374-mmc5.pdf)).

If `springable` or the **concept of flexel** has been useful for your work, your research or your projects, we strongly encourage you to cite our paper


> Ducarme, P., Weber, B., van Hecke, M., & Overvelde, J. T. B. (2026). *Flexel ecosystem: Simulating mechanical systems made from entities with arbitrarily complex mechanical responses*. EML (accepted). [https://doi.org/10.1016/j.eml.2026.102476](https://doi.org/10.1016/j.eml.2026.102476)


```bib
@article{ducarme_flexel_2026,
	title = {Flexel ecosystem: {Simulating} mechanical systems made from entities with arbitrarily complex mechanical responses},
	issn = {2352-4316},
	shorttitle = {Flexel ecosystem},
	url = {https://www.sciencedirect.com/science/article/pii/S2352431626000374},
	doi = {10.1016/j.eml.2026.102476},
	abstract = {Nonlinearities and instabilities in mechanical structures have shown great promise for embedding advanced functionalities. However, simulating structures subject to nonlinearities can be challenging due to the complexity of their behavior, such as large shape changes, effect of pre-tension, negative stiffness and instabilities. While traditional finite element analysis is capable of simulating a specific nonlinear structure quantitatively, it can be costly and cumbersome to use due to the high number of degrees of freedom involved. We propose a framework to facilitate the exploration of highly nonlinear structures under quasistatic conditions. In our framework, models are simplified by introducing ‘flexels’, elements capable of intrinsically representing the complex mechanical responses of compound structures. By extending the concept of nonlinear springs, flexels can be characterized by multi-valued response curves, and model various mechanical deformations, interactions and stimuli, e.g., stretching, bending, contact, pneumatic actuation, and cable-driven actuation. We demonstrate that the versatility of the formulation allows to model and simulate, with just a few elements, complex mechanical systems such as pre-stressed tensegrities, tape spring mechanisms, interaction of buckled beams and pneumatic soft gripper actuated using a metafluid. With the implementation of the framework in an easy-to-use Python library, we believe that the flexel formulation will provide a useful modeling approach for understanding and designing nonlinear mechanical structures.},
	urldate = {2026-04-09},
	journal = {Extreme Mechanics Letters},
	author = {Ducarme, Paul and Weber, Bart and Hecke, Martin van and Overvelde, Johannes T. B.},
	month = apr,
	year = {2026},
	keywords = {Instability, Nonlinear spring, Nonlinearity, Reduced order model, Simulation},
	pages = {102476},
}
```

