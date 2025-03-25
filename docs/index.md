<figure markdown="1">
![](https://github.com/user-attachments/assets/5642dbbe-f27c-4c71-bb05-c6ae4785f801){ width="500" }
<figcaption>Library for nonlinear spring simulations</figcaption>
<video autoplay loop muted src="https://github.com/user-attachments/assets/2d5840ef-d182-48ef-9dfb-d0af7aaad448"></video>
</figure>



**Springable** is a library for **mechanical simulations of nonlinear springs**. It allows you to simulate how structures made out of (non)linear springs deform when subject to forces.
By accounting for any geometrical changes (as large as they may be), the simulation allows you to explore the richness
of nonlinear mechanics, beyond the (boring) linear regime.

The implementation of the library is geared towards understanding how spring assemblies lead to mechanical behaviors
ranging from simple monotonic responses to complex, highly-nonlinear ones, such as snapping instabilities, sequencing,
buckling, symmetry-breaking or restabilization.

In its core, `springable` deals with **springs**, that we define as any entity that can store [elastic energy](https://en.wikipedia.org/wiki/Elastic_energy).
*Springs* therefore include longitudinal springs (compression and extension),
rotation springs (bending), area springs (useful to model fluids and pneumatic loading), line springs (useful to model cable-driven systems), and more!
On top of that, the library allows you to define the energy potential of each individual spring to make them intrinsically linear or nonlinear, thereby generating a whole ecosystem of springs, ready to be assembled and simulated!

**Table of contents**

- [Getting started](getting_started)
    - [Installation](getting_started/#installation)
    - [Don't want to install it right now? Try the Online Notebook](getting_started/#dont-want-to-install-it-right-now-try-the-online-notebook)
    - [Running a simulation](getting_started/#running-a-simulation)
- [Creating a CSV file describing the spring model (+examples)](creating_the_spring_model_csv_file)
- [Specifying a nonlinear mechanical behavior (+examples)](specifying_a_nonlinear_mechanical_behavior)
- [Configuring simulation settings (+examples)](configuring_simulation_settings)
- [Advanced topics](advanced_topics)
    + [Area spring with holes](advanced_topics/#area-spring-with-holes)
    + [Complex loading descriptions](advanced_topics/#complex-loading-descriptions)
    + [Scanning parameters](advanced_topics/#scanning-parameters)





