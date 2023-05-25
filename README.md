# Brainweb-DL

A collection of python scripts to download and convert the Brainweb dataset. 

## Features

- Automatic download, caching and formatting of the brainweb dataset 
  - Conversion to the Nifti format
 
- Download all the brainweb data and format it as a BIDS dataset. 

- Generate T1, T2, T2* and PD images from the brainweb dataset
  - NB: The values provided are not true values, but are suppose to be  close enough to be used for testing purposes.


## Installation 
    
    ```bash 
    pip install brainweb-dl
    ```
## Usage
### In a Python Script

    `
    ```python
    from brainwebdl import 
   
    ```

The brainweb dataset is downloaded and cached in the `~/.cache/brainweb` folder by default.

### Using the Command line 

``` bash
brainweb-dl --help
```

### TODO 
- [ ] add interface to download as BIDS
- [ ] add interface to generate T1, T2, T2* and PD images
- [ ] fuzzy / Multiple subjects download in parallel ? 
- [ ] find Location of brainweb dataset (User > Env variable > default location)

## Acknowledgement

 - https://github.com/casperdcl/brainweb/ for the preliminary work and the original idea. Check out his work for 

 - https://brainweb.bic.mni.mcgill.ca/ for providing the dataset to the community.
