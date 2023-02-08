# Dataset folder
The dataset aggregates five different datasets.
- Scripts_learning_dataset_proscript_use_flatten_generation.pickle
This dataset make proscript data use the original format in the *flatten_output_for_script_generation* column. So for each topic there is only one script.

Load the dataset pickle file use the following code:
```
import pickle

with open('Scripts_learning_dataset_proscript_use_flatten_generation.pickle', 'rb') as handle:
    thisdataset = pickle.load(handle)
```
