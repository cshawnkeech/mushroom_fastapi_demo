# Packing Light Activity

Create a virtual environment to support a model that will be deployed

## Objectives

* Create a virtual environment with only the libraries necessary for
  making predictions with the preserved model.
* Move prediction code into `src` directory as a `mushroom_model.py`
  file.
* Confirm the model can be run with the virtual environment
* Export `requirements.txt`

## Review

### Model Preservation Review

Open the first notebook with your development environment (maybe you are
using `conda base`). Review the code as you progress through the entire
notebook. By the end, you should have a model in the `models` directory.

This should feel like review, as this is the same model you interacted
with in the model preservation activity.

Before you continue, confirm:

* The first notebook in the `notebooks` directory has been run completely
  using your development environment (probably `conda base`, unless
  you've made some changes).
* The `artifacts.joblib` file exists in the `models` directory.

### Load model via `joblib`

Open the second notebook with your development environment. Review the
code as you progress through the entire notebook.

Run the code to confirm that the model loads and makes predictions
properly.

## Packing Light

Now we come to the main point of the activity. We want to pack our
environmental requirements and our code in as small a package as
possible.

Jupyter notebooks require quite a few libraries to run & render. There
was also a fairly large dataset we had to download in order to train the
model. The conda environment probably also has linters and other tools
that aren't necessary just to pack and deploy this model.

Let's look at how we can pack light.

## Getting our model to work in a `.py` script

We want to write only the most necessary code in a `.py` Python script.

Create a new `src` directory at the root of the repository. Inside this
directory, create two new files: `__init__.py` and `mushroom_model.py`.

Inside `mushroom_model.py`, stem out some starter code:

```python
"""
mushroom_model

Functions for predicting whether a given mushroom is poisonous or not.
"""


if __name__ == "__main__":

    pass


```

## MVP: replicating the results of `002-secondary_mushroom.ipynb`

In `002-secondary_mushroom.ipynb`, we were able to restore the
preprocessor and model from `artifacts.joblib`. You may remember that
they were both stored in a dictionary with a few other useful objects.

Our MVP is to duplicate this result to work in a `.py` script file using
the virtual environment. It may be helpful to have the
`002-secondary_mushroom.ipynb` notebook open as a reference.

### Restore the model

First add the code to restore the model. You can use the INEM block to
confirm that the artifact is a Python dictionary.

```python
"""
mushroom_model

Functions for predicting whether a given mushroom is poisonous or not.
"""
import joblib

PATH = "models/artifacts.joblib"
ARTIFACT = joblib.load(PATH)


if __name__ == "__main__":

    # confirm ARTIFACT dictionary
    print(
        type(ARTIFACT),
        ARTIFACT.keys(),
        sep="\n")

```

Try running the file from the root of the directory. If it doesn't run
smoothly the first time, reach out to your peers & instructors for help
troubleshooting.

## Writing a function to predict

You'd like to write a function to make the predictions. Take a moment and
think about what the function should take as input (the parameters), what
the function should do, and what the function should return.

* Input: In `002-secondary_mushroom.ipynb`, the new data had to be
  observations stored as a DataFrame.
* What the function does: The function will need to take that raw (but
  valid) DataFrame, preprocess the DataFrame, and then call
  `lr_model.predict()` on the preprocessed DataFrame.
* Returns: The model returns an Numpy `ndarray` by default (`0` if it's
  likely not poisonous, `1` if it is likely poisonous).

We'll need to import Pandas. It may also make things easier to unpack the
model and preprocessor as constants.

### Your turn

Here is some starter code. Create the constants `MODEL` and
`PREPROCESSOR`and the `predict_mushroom` function as described.

```python
# only partial code shown
import joblib
import pandas as pd

PATH = "models/artifacts.joblib"
ARTIFACT = joblib.load(PATH)
MODEL = None
PREPROCESSOR = None


def predict_mushroom(df):
    """
    Returns ndarray of predictions (poisonous=1) 
    given a properly formatted DataFrame of observations.
    """
    pass

# inem block below (not shown)


```

### Test it out

You'll probably want to test out the function in the INEM block. Let's
use the same example observation from `002-secondary_mushroom.ipynb`.

```python
observation = {
    'cap-diameter': [50],
    'stem-height': [20],
    'stem-width': [30],
    'has-ring': ['t'],
    'cap-shape': ['c']
}
```

You'll need to convert this observation to a DataFrame.

```python
# showing only the INEM block

if __name__ == "__main__":

    observation = {
        'cap-diameter': [50],
        'stem-height': [20],
        'stem-width': [30],
        'has-ring': ['t'],
        'cap-shape': ['c']
    }

    single_obs_df = pd.DataFrame(observation)

    print(predict_mushroom(single_obs_df))

```

You will notice that you get a numpy ndarray with a single prediction
out.

## Create a Virtual environment called `venv`

* [venv — Creation of virtual environments — Python 3.12.3
   documentation](https://docs.python.org/3/library/venv.html)
* [Install packages in a virtual environment using pip and venv - Python
   Packaging User
   Guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)

> If you are using a conda environment for development, be sure that
> environment is activated when you create your virtual environment.

You have been using your development environment (possibly `conda base`
so far).

To create a virtual environment, type the code below. Note that the first
`venv` is the Python module (`-m`) you are running, and the second
`.venv` is the name of the directory you are creating. There are a few
common names for virtual environment directories, and `.venv` is among
them.

```bash
python -m venv .venv
```

> Note: If you are using the integrated terminal in VSCode, you may get a
> message reading `We noticed a new environment has been created. Do you
> want to select it for the workspace folder?`. Feel free to select Yes
> if you would like.

Note that a directory named `.venv` is created. If you look closely in
the standard `.gitignore` template (like the one in this directory), the
directory `.venv` is ignored. Other common virtual environment directory
names are ignored in the same section.

This means that your virtual environment **will not** be staged or
committed to git or pushed to GitHub.

## Activate your virtual environment

* [How venvs
   work](https://docs.python.org/3/library/venv.html#how-venvs-work)

* [Activate a virtual environment - Python Packaging User
  Guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#activate-a-virtual-environment)

If you are using a bash style terminal, you can start your virtual
environment like this.

```bash
# to activate
source .venv/bin/activate

# to deactivate
deactivate
```

You will notice that your prompt now has `(.venv)` in front of it. If you
still have `(base)` as well, that's ok.

## Prepare `pip`

Now that we have a virtual environment, we'll need to choose what
libraries are necessary to include.

With your virtual environment activated, install `pip`, the Python
package manager.

```bash
# install pip
python -m pip install --upgrade pip

# you can check your version
# which should also show the python version
python -m pip --version
```

## Install necessary libraries

If you type `pip freeze` while your virtual environment is activated, you
can see what libraries have been installed and are available. If you try
that now, the list will be empty. If you try to run
`src/mushroom_model.py`, you will get a `ModuleNotFoundError`. The
virtual environment doesn't yet have any libraries installed.

You'll need some libraries to run your model. Take a look at the
libraries imported throughout the `002-secondary_mushroom.ipynb`
notebook.

It looks like these libraries are imported.

```text
joblib
pandas
```

It looks like there were two functions (`f1_score` and `accuracy_score`)
imported from `sklearn.metrics`. These are only useful if you know the
true class already. If you are making predictions on completely unseen
data, you won't be able to assign a score. We can leave these out of the
install.

If you are using a conda environment you may want to see what versions of
these libraries you are using. You may remember that `joblib` is pretty
picky; it usually only works if you have the same versions of the
libraries as when you created the files.

Try these in your terminal.

```bash
conda list | grep joblib

conda list | grep pandas
```

You may see something like this

```bash
joblib                    1.2.0           py311hecd8cb5_0
pandas                    2.1.4           py311hdb55bb0_0
```

For this example, it looks like joblib is version `1.2.0` and pandas is
version `2.0.3`. Yours may be a little different, so be sure to use your
versions.

Since we want to install exactly these versions, we'll use `==` to ensure
pip installs the versions we want.

Using these example versions (be sure to change these to your versions),
we'll install these into our virtual environment with pip.

First install `joblib`.

```bash
pip install joblib==1.2.0
```

Type `pip freeze` and you will now see `joblib` included in the output.

```bash
joblib==1.2.0
```

Now install Pandas.

```bash
pip install pandas==2.1.4
```

Pandas took a little longer. While installing, you may have noticed this
part of the output:

<!-- markdownlint-disable MD013 -->

```bash
Installing collected packages: pytz, tzdata, six, numpy, python-dateutil, pandas
Successfully installed numpy-1.26.4 pandas-2.1.4 python-dateutil-2.9.0.post0 pytz-2024.1 six-1.16.0 tzdata-2024.1
```
<!-- markdownlint-enable MD013 -->

Yours may look a little different depending on the version of Pandas you
install. Pandas installed several dependencies-other libraries that
Pandas can't operate without. Try `pip freeze` now.

The output will look something like this.

```bash
joblib==1.2.0
numpy==1.26.4
pandas==2.1.4
python-dateutil==2.9.0.post0
pytz==2024.1
six==1.16.0
tzdata==2024.1
```

This may not be everything the virtual environment requires, but it is a
good time to try running `mushroom_model.py` to see what happens.

## Attempt to run `mushroom_model.py`

Try to run `src/mushroom_model.py` with the virtual environment
activated. It will take a few moments, and you'll likely encounter an
error message.

Here's an example (your file paths will differ).

<!-- markdownlint-disable MD013 -->
<!-- cSpell:disable -->

```bash
Traceback (most recent call last):
  File "/Users/cshawnkeech/galvanize/ddi/assignments/adt-ddi-packing-light-activity/src/mushroom_model.py", line 10, in <module>
    ARTIFACT = joblib.load(PATH)
               ^^^^^^^^^^^^^^^^^
  File "/Users/cshawnkeech/galvanize/ddi/assignments/adt-ddi-packing-light-activity/.venv/lib/python3.11/site-packages/joblib/numpy_pickle.py", line 658, in load
    obj = _unpickle(fobj, filename, mmap_mode)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/cshawnkeech/galvanize/ddi/assignments/adt-ddi-packing-light-activity/.venv/lib/python3.11/site-packages/joblib/numpy_pickle.py", line 577, in _unpickle
    obj = unpickler.load()
          ^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/base_rise/lib/python3.11/pickle.py", line 1213, in load
    dispatch[key[0]](self)
  File "/opt/anaconda3/envs/base_rise/lib/python3.11/pickle.py", line 1538, in load_stack_global
    self.append(self.find_class(module, name))
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/base_rise/lib/python3.11/pickle.py", line 1580, in find_class
    __import__(module, level=0)
ModuleNotFoundError: No module named 'sklearn'

```

<!-- markdownlint-enable MD013 -->
<!-- cSpell:enable -->

It looks like the big problem is that there is no `sklearn` in the
virtual environment.

Let's check the version we used in development and then add that version
to the environment.

This is one occasion where it may be helpful to use the VSCode integrated
terminal for the virtual environment and the regular terminal for
checking the conda environment.

Check your development environment (probably `conda base`).

```bash
# Check which version of scikit-learn
conda list | grep scikit-learn

```

You may have more than one result. For now, install `scikit-learn` only.

```bash
scikit-learn              1.2.2           py311hcec6c5f_1

```

```bash
# be sure to use your version
pip install scikit-learn==1.2.2
```

Note that `scikit-learn` also brought along a few dependencies.

```python
# take a look at what pip has installed so far
pip freeze
```

### Try the code again

If you run the code again, it should be successful. Note that it takes a
moment to load.

## Create `requirements.txt`

One last step. Let's create a file that saves a list of all of the
necessary packages.

Remember: the `.venv` directory is in the `.gitignore` file, so it won't
be in the repository.

When someone else wants to download our code and make their own virtual
environment, they'll need a list of requirements.

From the root of your repository, pipe the results of `pip freeze` into a
new file: `requirements.txt`.

```bash
# Create requirements.txt
pip freeze > requirements.txt

```

The contents will look something like this:

```text
joblib==1.2.0
numpy==1.26.4
pandas==2.1.4
python-dateutil==2.9.0.post0
pytz==2024.1
scikit-learn==1.2.2
scipy==1.13.1
six==1.16.0
threadpoolctl==3.5.0
tzdata==2024.1

```

Take a moment and note what is in there and what is not.

* Libraries and their dependencies installed by pip are present.
* Python and pip are not.

If you type `pip --version`, while your virtual environment is active,
the Python version will be included. You can also look inside your
`.venv` directory, you can look inside either bin and include and see th
eversion of Python. Mine says `python3.11`.

This is not documented in the requirements file, but would be important
to note in your repository (possibly in the README or in the code
documentation itself).

This is a good time to stage/commit/push your work if you haven't
recently.

You are welcome to deactivate the virtual environment.

```bash
deactivate
```

## How do you put it back together?

It's worth running this experiment. Now that you have your code pushed,
try one of two activities:

* clone it again in an entirely different directory
* with the virtual environment deactivated, delete the `.venv` and all of
  its contents

Either way, try creating a new virtual environment by following the
instructions above, including the instructions to upgrade pip.

* create a virtual environment
* activate the environment
* upgrade pip

However, instead of installing packages with `pip` individually, install
them with the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

Run the `src/mushroom_model.py` and confirm it works. Be sure to
deactivate your environment when you are done.

## Conclusions: MVP accomplished

In this activity, you have

* Converted code in a notebook to a Python script (a `.py` file)
* Created a virtual environment and packed only what we absolutely need
to run the code
* Exported those requirements into a `requirements.txt` file.
* Pulled that code and used the `requirements.txt` file to rebuild and
  run the code
