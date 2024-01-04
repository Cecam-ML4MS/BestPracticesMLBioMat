# About

This folder contains tutorials to accompany the best practices paper entitled
"Best Practices for Training and Applying Machine Learning Potentials to Large
and Complex Chemical Systems".

The tutorials here are meant to demonstrate **concepts** and make no claims of
real-life performance.

## Design

These tutorials are distributed as *executable* notebooks but in the MyST
Markdown format (i.e. they are [MyST
NB](https://myst-nb.readthedocs.io/en/latest/index.html) files). This is for a
few reasons:

- Has a good human readable plain-text representation
- Works better with version control
- Can be rendered easily onto web-pages (statically)
  + Without needing a kernel

Jupyter notebooks are the de-facto standard, but they are essentially blobs of
`json` and so it is better to track the markdown variants.

The design of this repository is inspired to a large extent by [the NumPy
tutorials](https://github.com/numpy/numpy-tutorials).

## Usage

We also provide an `environment.yml` file to ensure local reproducibility via
`conda` (or equivalent helpers like `mamba` or `micromamba`).

``` sh
# Create an envirnment named bp_mlbmat
micromamba create -f environment.yml
micromamba activate bp_mlbmat
jupyter lab --ServerApp.allow_remote_access=1 \
    --ServerApp.open_browser=False --port=8889
```

Now navigate to `localhost:8889` to start interacting with the tutorials.
