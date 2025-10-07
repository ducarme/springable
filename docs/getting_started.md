## Installation

Use `pip` to install. In the terminal simply enter

```
python -m pip install springable
```

and you are good to go!

It is supported on **Python 3.10 and above**.

## Try online without installation
Try `springable` online, without any installation in an [interactive online notebook](https://colab.research.google.com/github/ducarme/springable/blob/main/docs/examples/example01_getting_started/example01_getting_started.ipynb)

## Running a simulation
To start a simulation, we first create the file 
that will describe the spring model we want to simulate. To do that, we create a simple
[CSV file](https://en.wikipedia.org/wiki/Comma-separated_values),
(a text file saved with extension *.csv*), that looks like this for example.

```csv title="my_spring_model.csv"
PARAMETERS
stiffness, 1.0
NODES
0, 0.0, 0.0, 1, 1
1, 1.0, 0.0, 0, 1
SPRINGS
0-1, stiffness
LOADING
1, X, 1.5
```

This file defines a spring structure composed of only one horizontal spring, clamped on the left and loaded in tension from the right.

How to read or make such a file is described in the paragraph [Creating a CSV file describing the spring model](creating_the_spring_model_csv_file.md).
Many CSV file examples that describe spring structures are already available
[here on GitHub](https://github.com/ducarme/springable/tree/main/examples-spring-model-CSV-files) for inspiration or to download.

Next, we create a Python script (a text file saved with the extension *.py*), with the following content

```python title="my_first_simulation.py"
"""
Python script example to learn how to use the package springable
"""
import springable.simulation as ss

ss.simulate_model(model_path='my_spring_model.csv',
                  save_dir='my_simulation_result')
```
and save it under the name - let's say - `my_first_spring_simulation.py`

The function `ss.simulate_model()` takes in two arguments:

* The `model_path` argument is the path leading to the [CSV file](https://en.wikipedia.org/wiki/Comma-separated_values)
that describes the model you want to simulate. In this case, we used the CSV file `my_spring_model.csv` that we have just created.

* The `save_dir` argument is simply the name of the folder under which the simulation results will be saved. It should not exist yet; it will be created automatically when we run the script.
It is an optional argument, if not specified, a folder will be created automatically in the working directory to store the result files.

Finally, we run the Python script. This can be done in the terminal by simply executing
```
python my_first_spring_simulation.py
```

!!! tip
    Many settings can be tuned before running a simulation. See paragraph [Configuring simulation settings](configuring_simulation_settings.md) for more details.

## Quickly viewing the results 

After running the command, three media files are generated, shown and saved.

* The drawing depicting the spring structure about to be simulated (before solver starts)
* The force-displacement curve of the structure (after solver finishes)
* The animation of the model as it is loaded (after solver finishes)

<div class="grid cards" markdown>

-   spring structure drawing
    
    ---
    
    ![](https://github.com/user-attachments/assets/0b51521f-87a2-43ca-a153-7252caca8942)

-   force-displacement curve

    ---

    ![](https://github.com/user-attachments/assets/6420086a-d87c-47ce-984e-ce98c6a475d7)

</div>

<div class="grid cards" markdown>
-    animation

     ---

    <video autoplay loop muted src="https://github.com/user-attachments/assets/8b40afd1-db93-4bd7-adfb-1d106e96e740"></video>
</div>

!!! tip
    Many settings can be tuned to change of the appearance of the spring structure, plots settings, colors, animation fps and resolution, etc.
    See paragraph [Configuring simulation settings](configuring_simulation_settings.md) for more details.