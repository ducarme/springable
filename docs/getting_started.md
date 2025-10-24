## Installation

Use `pip` to install. Open the terminal, enter

```
pip install springable
```
in your activated virtual environment, and you are good to go!

It is supported on **Python 3.10 and above**.

??? tip "Install `springable` in a virtual environment (easy & recommended)"
    It is recommended to install `springable` in a virtual environment. Create a folder, use the terminal to navigate to that folder and enter
    === "Windows"
        ``` console
        python -m venv venv
        ```
        to create the virtual environment. A folder named `venv` is automatically created in your folder. To activate the virtual environment, run

        ``` console
        venv\Scripts\activate
        ```
        in the terminal. From now on, every installation only lives in the virtual environment. Install `springable` by entering
        ``` console
        pip install springable
        ```
        in the terminal.
    === "Macbook/Apple"
        ``` console
        python3 -m venv venv
        ```
        to create the virtual environment. A folder named `venv` is automatically created in your folder. To activate the virtual environment, run
        ``` console
        source venv/bin/activate
        ```
        in the terminal. From now on, every installation only lives in the virtual environment. Install `springable` by entering
        ``` console
        pip install springable
        ```
        in the terminal.

    A [virtual environment](https://www.w3schools.com/python/python_virtualenv.asp) is like a mini Python setup just for one project. It keeps its own copy of Python and its own folder for packages. This way:

    * you can install packages without touching the global Python installation;
    * different projects can use different versions of the same package without conflicts;
    * when you’re done, you can just deactivate the environment by entering
    ``` console
    deactivate
    ```
    in the terminal. You can re-activate it later as shown above (no need to re-create it of course).

??? failure "Error message: 'Could not find a version that satisfies the requirement springable'?"
    If you see a message similar to
    ``` console
    ERROR: Could not find a version that satisfies the requirement springable (from versions: none)
    ERROR: No matching distribution found for springable
    ```
    after entering `pip install springable`, you most likely have a version of Python that is too old for `springable`. This is an easy fix. You can simply upgrade to a more recent version of Python (>= 3.10.0), by following the instructions in the help box named "How to install/upgrade Python"

    Note: newer Python versions are compatible with code running on older versions, so the newer Python version will still be able to run your older code. In any case, you can always create a virtual environment encapsulating the newer Python version, so that it does not affect anything outside that environment.

??? question "Not familiar with using the terminal? No problem!"

    The terminal is a program that allows you to type and run some commands to perform some actions on your computer, such as creating a folder, opening a file, navigating folders, installing other programs, starting a program, etc.

    How to start the terminal?

    === "Windows"
        If you use Windows, simply search for "Command Prompt" in the bottom-left search bar and double-click on the search result to start the terminal. The program is an empty window, waiting for you to type a command and press ++enter++ to run it.
    === "Macbook/Apple"
        If you use a Macbook or another Apple computer, press ++cmd+space++, search for "Terminal" and open the program. The program is an empty window, waiting for you to type a command and press ++enter++ to run it.

??? question "Not familiar with Python? No problem!"

    Python is a program that can read and run Python scripts. A Python script is a text file saved under a name that ends with the extension `.py`, such as `my_python_script.py` for example:
    ```python title="my_python_script.py"
        x = 1
        y = 2
        z = x + y
        print(f"Result: {z}")
    ```
    
    You do not need to have Python installed on your computer to create such a file. You can create a Python script by simply writing some text in a normal text editor such as Notepad (Windows) or TextEdit (Macbook), and save the file with a name that ends with `.py`.
    
    You need however to have Python installed on your computer to run a Python script. You can find instructions on how to install Python on your computer in the next help box.

    The text you write in a Python script needs to follow some rules and syntax in order for Python to be able to understand and execute it. Those rules form the Python programming language. To run spring simulations with `springable`, you won't have to write any Python scripts yourself. You can just copy-paste the scripts shown in this guide. **No knowledge of the Python language is required**. If you want to know more about Python, even though it is not required, you can find great tutorials online, such as [this one](https://realpython.com/learning-paths/python-basics/).


??? question "Do I have Python installed on my computer? If so, which version?"
    The quickest way to check whether Python is installed on your computer is to open the terminal,  type

    === "Windows"
        ``` console
        python --version
        ```
    === "Macbook/Apple"
        ``` console
        python3 --version
        ```
    and press ++enter++.

    If your terminal replies with something similar to
    === "Windows"
        ```console
        'python' is not recognized as an internal or external command, operable program or batch file.
        ```
    === "Macbook/Apple"
        ```console
        command not found: python
        ```
    it means that Python is not installed on your computer. You will have to install it first. It is not complicated: the instructions are available in the next help box.

    Instead, if your terminal replies with
    ```console
    Python a.b.c
    ```
    (where `a`, `b`, and `c` are numbers), then Python is installed on your computer.

    **To use `springable`, the Python version must be more recent than 3.10.0!**
    
    So, if the terminal replied with `Python 2.7.1` or `Python 3.9.12` for example, you will have to install a more recent version of Python. It is not complicated: the instructions are available in the next help box. As Python is backward compatible, upgrading to a more recent version should in principle not affect the way previous Python programs (that you might have already created) run.

    If the terminal replied with `Python 3.10.0` or `Python 3.13.4` for example, you are good to go. We recommend to create a virtual environment first before running `pip install springable`, as detailed the help box named "Install `springable` in a virtual environment".

??? question "How to install/upgrade Python?"
    If you do not have Python installed or if your Python version is older than 3.10, you need to install the latest version of Python. The easiest and recommended way to do that is to simply download Python from the official website [python.org/download](https://www.python.org/downloads/).

    Double-click on what has been downloaded and follow the instructions that will appear on screen.

    After the installation is completed, **re-open the terminal** and run
    === "Windows"
        ``` console
        python --version
        ```
    === "Macbook/Apple"
        ``` console
        python3 --version
        ```
    The terminal should reply with
    ```Python a.b.c```
    (where `a`, `b`, `c` are numbers). The version should be greater than 3.10.0.

    If it shows a version older than 3.10.0, you most likely need to fix the `PATH`. This [tutorial](https://realpython.com/add-python-to-path/) will guide you through it easily.

??? question "Installation via Anaconda/miniconda or in Jupyter notebook?"
    If you want to install `springable` in your `conda` environment, you can simply activate your `conda` environment and install via `pip`:
    ```
    conda activate my_env
    pip install springable
    ```

    The package `springable` also works perfectly fine in a Jupyter notebook. You can install it using
    ```
    !pip install springable
    ```
    in a cell at the start of your document.


## Try online without any installation
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
Many CSV file examples that describe spring structures are already available [on this page](many_examples.md), to help get started and familiar with the syntax and language.

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
    See page [Configuring simulation settings](configuring_simulation_settings.md) for more details.