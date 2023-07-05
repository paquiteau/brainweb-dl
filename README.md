# Brainweb-DL

Welcome to Brainweb-DL, a powerful Python toolkit for downloading and converting the Brainweb dataset. 

<p align="center">
<a href=https://github.com/paquiteau/brainweb-dl/blob/master/LICENSE><img src=https://img.shields.io/github/license/paquiteau/brainweb-dl></a>
<a href=https://www.codefactor.io/repository/github/paquiteau/brainweb-dl><img src=https://www.codefactor.io/repository/github/paquiteau/brainweb-dl/badge></a>
<a href=https://github.com/psf/black><img src=https://img.shields.io/badge/style-black-black></a>
<a href=https://pypi.org/project/brainweb-dl><img src=https://img.shields.io/pypi/v/brainweb-dl></a>
</p>

## Features

- **Effortless Dataset Management:** Automatically download, cache, and format the Brainweb dataset with ease. Convert it to the convenient nifti format or numpy array hassle-free.

- **Multiple Image Generation:** Generate high-quality T1, T2, T2*, and PD images directly from the Brainweb dataset. Perfect for testing purposes, although keep in mind that the values provided are approximate.

## Get Started

### Data Location

The cached data directory follows this priority order:

1. User-specific argument (`brainweb_dir` in most functions)
2. `BRAINWEB_DIR` environment variable
3. `~/.cache/brainweb` folder

### Python Script Usage
```python
from brainweb_dl import get_mri 

data = get_mri(sub_id=44, contrast="T1")
```

The Brainweb dataset is downloaded and cached by default in the `~/.cache/brainweb` folder.

### Command Line Interface

```bash
brainweb-dl 44 --contrast=T1
```

For more information, see `brainweb-dl --help`.

## Installation 

Get up and running quickly!

```bash 
pip install brainweb-dl
```

### Development

Join our community and contribute to Brainweb-DL!

```bash
git clone git@github.com/paquiteau/brainweb-dl 
cd brainweb-dl
pip install -e .[dev,test,doc]
```

### TODO List
Help us improve and shape the future of Brainweb-DL:

- [ ] Add unit tests.
- [ ] Implement fuzzy search and multiple subjects download in parallel.
- [ ] Introduce an interface to download as BIDS format.
- [x] Develop an interface to generate T1, T2, T2*, and PD images.
- [x] Enhance the search for the location of the Brainweb dataset (User > Environment Variable > Default Location).

## Acknowledgement

We extend our gratitude to the following for their contributions:

- [Casper De Clercq](https://github.com/casperdcl/brainweb/) for the preliminary work and original idea. Check out his great work if you are interested in PET imaging and registration functionalities.

- [BrainWeb](https://brainweb.bic.mni.mcgill.ca/) for providing this valuable dataset to the community.


<p align=center> :star2: If you like this work, don't forget to star it and share it 🌟 </p>
