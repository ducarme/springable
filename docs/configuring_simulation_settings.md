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
```toml title="custom_solver_settings.toml"
radius = 0.01
reference_load_parameter = 0.01
```
> Low values for `radius` and `reference_load_parameter` can be used to refine the solution,
> at the cost of increasing the solving duration. Default values are 0.05 and 0.05, respectively.

To use these custom solver settings, use the path to `custom_solver_settings.toml`
as an extra argument of the `ss.simulate_model()` function, as follows:

```python title="my_first_simulation.py"
"""
Python script example to learn how to use the package springable
"""
import springable.simulation as ss

ss.simulate_model(model_path='my_spring_model.csv',
                  save_dir='my_simulation_result',
                  solver_settings_path='custom_solver_settings.toml')
```


Similarly, when you wish to modify a graphics setting, create another TOML file and include the settings you wish to modify
```toml title="custom_graphics_settings.toml"
[animation_options]
nb_frames = 240
fps = 60

[plot_options]
show_snapping_arrows = true
drive_mode = "force"
```

> Animation settings `nb_frames` and `fps` determine the number of frames and the frame rate (in frame per second) of the animation showing
> the spring assembly deforming. Plot settings `show_snapping_arrows = true` combined with `drive_mode = "force"` means that
> you want to indicate with arrows the (potential) snapping transitions under controlled force in the force-displacement plot.
> To indicate, snapping transitions under controlled displacement use `show_snapping_arrows = true` combined with `drive_mode = "displacement"`
> instead.

To use these custom graphics settings, use the path to `custom_graphics_settings.toml`
as an extra argument of the `ss.simulate_model()` function, as follows:

```python title="my_first_simulation.py"
"""
Python script example to learn how to use the package springable
"""
import springable.simulation as ss

ss.simulate_model(model_path='my_spring_model.csv',
                  save_dir='my_simulation_result',
                  solver_settings_path='custom_solver_settings.toml',
                  graphics_settings_path='custom_graphics_settings.toml')
```


## Additional notes
* A custom settings file does not need to contain all the possible settings; just include the one you wish to modify.
* Graphics settings are divided into 4 sections of settings (indicated by `[...]` in TOML files):
    * general options (determines _what_ should be generated and directly shown (drawing, animation, plot))
    * plot options (determines _how_ plots will look like)
    * animation options (determines _how_ animations will look like)
    * assembly appearance (determines _how_ the spring assembly will be depicted)