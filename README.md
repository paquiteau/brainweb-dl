# Brainweb-DL

A collection of python scripts to download and convert the Brainweb dataset. 

## Features

- Automatic download, caching and formatting of the brainweb dataset, with conversion to nifti format
 

- Generate T1, T2, T2* and PD images from the brainweb dataset
  - NB: The values provided are not true values, but are suppose to be  close enough to be used for testing purposes.
    
## Usage

### Location of the data

The directory for the cached data is by order of priority:
- A user specific argument (`brainweb_dir` in most functions)
- The `BRAINWEB_DIR` environment variable
- The `~/.cache/brainweb` folder

### In a Python Script
```python
from brainweb_dl import get_mri 

data = get_mri(subject=44, contrast="T1")
```

The brainweb dataset is downloaded and cached in the `~/.cache/brainweb` folder by default.

### Using the command line interface

``` bash
$ brainweb-dl 44 --contrast=T1
```

see `brainweb-dl --help` for more information.

## Installation 
    
```bash 
$ pip install brainweb-dl
```
### Development

``` bash
$ git clone git@github.com/paquiteau/brainweb-dl 
$ cd brainweb-dl
$ pip install -e .[dev,test,doc]
```



### TODO 
- [ ] Add unit test.
- [ ] fuzzy / Multiple subjects download in parallel ? 
- [ ] add interface to download as BIDS
- [x] add interface to generate T1, T2, T2* and PD images
- [x] find Location of brainweb dataset (User > Env variable > default location)

## Acknowledgement

 - https://github.com/casperdcl/brainweb/ for the preliminary work and the original idea. Check out his work if you are looking for PET imaging, and registration functionalities.

 - https://brainweb.bic.mni.mcgill.ca/ for providing the dataset to the community.
