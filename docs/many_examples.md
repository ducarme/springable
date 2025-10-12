# Some examples

Here are shown many examples to help you get started and familiar with defining a spring model in `springable`. To run and simulate an example, copy the corresponding code in a text file and save it under the name `example.csv`. Then, run this Python script:
```python
from springable.simulation import simulate_model

simulate_model('example.csv')
```

??? question "Just want to be shown many examples and simulations?"
    Examples can also be automatically simulated and shown to you by running this Python script:
    ```python
    from springable.discover import show_examples

    show_examples()
    ```

    More complex spring structures can be shown by running:
    ```python
    from springable.discover import show_gallery

    show_gallery()
    ```

Some help if you are not sure how to run a Python script or get set up is provided [here](getting_started.md).

## Single longitudinal springs
### with linear behavior
```
--8<-- "src/springable/data/examples_csv_models/single_linear_spring.csv"
```
### with a piecewise softening behavior
```
--8<-- "src/springable/data/examples_csv_models/single_softening_spring.csv"
```
### with a piecewise stiffening behavior
```
--8<-- "src/springable/data/examples_csv_models/single_stiffening_spring.csv"
```
### with a piecewise nonmonotonic behavior
```
--8<-- "src/springable/data/examples_csv_models/single_nonmonotonic_spring.csv"
```
### with a complex multi-valued zigzag behavior
```
--8<-- "src/springable/data/examples_csv_models/single_highly_nonlinear_spring.csv"
```

## Basic geometrically nonlinear structures
### two springs connected at an angle and a flexure
```
--8<-- "src/springable/data/examples_csv_models/two_springs_at_an_angle.csv"
```
### Von-Mises truss
```
--8<-- "src/springable/data/examples_csv_models/von_mises_spring_truss.csv"
```
### Buckling
```
--8<-- "src/springable/data/examples_csv_models/buckling.csv"
```

## Different type of springs
### angular spring (to model (non)linear flexures)
```
--8<-- "src/springable/data/examples_csv_models/example_nonlinear_angular_spring.csv"
```
### area spring (to model pressure actuation)
```
--8<-- "src/springable/data/examples_csv_models/example_nonlinear_area_spring.csv"
```
### path spring (to model cable, rope and pulleys)
```
--8<-- "src/springable/data/examples_csv_models/example_nonlinear_path_spring.csv"
```
### distance spring (to model contact)
```
--8<-- "src/springable/data/examples_csv_models/example_contact_distance_spring.csv"
```

## Inflatables
### pneunet actuator
```
--8<-- "src/springable/data/examples_csv_models/pneunet.csv"
```
### dome under pressure
```
--8<-- "src/springable/data/gallery_csv_models/arc_shallow.csv"
```
### another dome under pressure
```
--8<-- "src/springable/data/gallery_csv_models/arc_shallow2.csv"
```

## Snapping, sequencing, buckling and contact
### sequencing with nonmonotonic Bezier springs
```
--8<-- "src/springable/data/examples_csv_models/example_nonlinear_spring_assembly.csv"
```
### bumping beams
```
--8<-- "src/springable/data/gallery_csv_models/bumpingbeams.csv"
```
### beam in compression with wall
```
--8<-- "src/springable/data/gallery_csv_models/beam_in_compression_with_one_contact.csv"
```
### beam in compression with two walls
```
--8<-- "src/springable/data/gallery_csv_models/beam_in_compression_with_two_contacts.csv"
```
### snapping square
```
--8<-- "src/springable/data/gallery_csv_models/snapping_square.csv"
```