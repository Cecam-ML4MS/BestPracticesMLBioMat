---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"id": "yOyfVHby2NmH"}

## Tutorial for kernel-based methods
This jupyter notebook is part of the CECAM workshop "Machine-learned potentials in molecular simulation: best practices and tutorials".

*Credits:* The notebook relies on the work of the following repositories:

- https://github.com/qmlcode/qml/tree/develop
- https://github.com/ferchault/APDFT/tree/master
- https://github.com/andersx/oqml-md/tree/master

Please check them out for more detailed examples!

For more details regarding the theory behind kernel methods and more advanced functionality check out:
- https://www.qmlcode.org/index.html
- "FCHL revisited: Faster and more accurate quantum machine learning", Christensen et al. (2020), DOI: `10.1063/1.5126701`.
- "Fast and Accurate Modeling of Molecular Atomization Energies with Machine Learning", Rupp et al. (2012), DOI: `10.1103/PhysRevLett.108.058301`

+++ {"id": "pYyZpvhV2NmJ"}

### Outline

The notebook is divided into three parts:

1. The first part is a self-contained implementation in numpy using a "simple" feature represenation (i.e. coloumb matrix) to predict energies for the qmrxn20 dataset.

2. The second part extends the ideas and used a more advanced feature representation to achieve improved accuracy.

3. The final part uses the advanced features to also predict forces.

+++ {"id": "QOTOns6fIEDD"}

### General settings

```{code-cell} ipython3
:id: qcisl8kVdeb6

import numpy as np
from collections import namedtuple
import functools
import requests
import io
import gzip
import tarfile

np.random.seed(1234)
```

```{code-cell} ipython3
:id: uq37BRKHdehX

# Mapping string to integer for nuclear charges
NUCLEAR_CHARGE = {
 'H'  :     1,
 'He' :     2,
 'Li' :     3,
 'Be' :     4,
 'B'  :     5,
 'C'  :     6,
 'N'  :     7,
 'O'  :     8,
 'F'  :     9,
 'Ne' :    10,
 'Na' :    11,
 'Mg' :    12,
 'Al' :    13,
 'Si' :    14,
 'P'  :    15,
 'S'  :    16,
 'Cl' :    17,
 'Ar' :    18,
 'K'  :    19,
 'Ca' :    20,
 'Sc' :    21,
 'Ti' :    22,
 'V'  :    23,
 'Cr' :    24,
 'Mn' :    25,
 'Fe' :    26,
 'Co' :    27,
 'Ni' :    28,
 'Cu' :    29,
 'Zn' :    30,
 'Ga' :    31,
 'Ge' :    32,
 'As' :    33,
 'Se' :    34,
 'Br' :    35,
 'Kr' :    36,
 'Rb' :    37,
 'Sr' :    38,
 'Y'  :    39,
 'Zr' :    40,
 'Nb' :    41,
 'Mo' :    42,
 'Tc' :    43,
 'Ru' :    44,
 'Rh' :    45,
 'Pd' :    46,
 'Ag' :    47,
 'Cd' :    48,
 'In' :    49,
 'Sn' :    50,
 'Sb' :    51,
 'Te' :    52,
 'I'  :    53,
 'Xe' :    54,
 'Cs' :    55,
 'Ba' :    56,
 'La' :    57,
 'Ce' :    58,
 'Pr' :    59,
 'Nd' :    60,
 'Pm' :    61,
 'Sm' :    62,
 'Eu' :    63,
 'Gd' :    64,
 'Tb' :    65,
 'Dy' :    66,
 'Ho' :    67,
 'Er' :    68,
 'Tm' :    69,
 'Yb' :    70,
 'Lu' :    71,
 'Hf' :    72,
 'Ta' :    73,
 'W'  :    74,
 'Re' :    75,
 'Os' :    76,
 'Ir' :    77,
 'Pt' :    78,
 'Au' :    79,
 'Hg' :    80,
 'Tl' :    81,
 'Pb' :    82,
 'Bi' :    83,
 'Po' :    84,
 'At' :    85,
 'Rn' :    86,
 'Fr' :    87,
 'Ra' :    88,
 'Ac' :    89,
 'Th' :    90,
 'Pa' :    91,
 'U'  :    92,
 'Np' :    93,
 'Pu' :    94,
 'Am' :    95,
 'Cm' :    96,
 'Bk' :    97,
 'Cf' :    98,
 'Es' :    99,
 'Fm' :   100,
 'Md' :   101,
 'No' :   102,
 'Lr' :   103,
 'Rf' :   104,
 'Db' :   105,
 'Sg' :   106,
 'Bh' :   107,
 'Hs' :   108,
 'Mt' :   109,
 'Ds' :   110,
 'Rg' :   111,
 'Cn' :   112,
 'Uuq':   114,
 'Uuh':   116}

# Auxiliary class to store all necessary information for a single compound
Compound = namedtuple('Compound', ['name', 'coordinates', 'nuclear_charges', "energy", "forces"])
```

+++ {"id": "CmpbE_ZP2NmJ"}

### Part 1: Numpy implementation

+++ {"id": "OjSGRIUT2NmL"}

### Load dataset

```{code-cell} ipython3
:id: 4TfDei7ddejv

# Read xyz file to return molecular coordinates and nuclear charges
# Handles a single compound per call
def read_xyz(data):
    if type(data) == str:
        f = open(data, "r")
        lines = f.readlines()
        f.close()
    else:
        lines = data

    natoms = int(lines[0])
    nuclear_charges = np.empty(natoms, dtype=int)
    coordinates = np.empty((natoms, 3), dtype=float)

    for i, line in enumerate(lines[2:natoms+2]):
        tokens = line.split()

        if len(tokens) < 4:
            break

        nuclear_charges[i] = NUCLEAR_CHARGE[tokens[0]]
        coordinates[i] = np.asarray(tokens[1:4], dtype=float)
    return coordinates, nuclear_charges
```

```{code-cell} ipython3
:id: S2_WtjVPdemh

# Load dataset from cloud
# Credits: https://github.com/ferchault/APDFT/blob/master/prototyping/mlmeta.py#L49
@functools.lru_cache(maxsize=1)
def database_qmrxn20():
    """ Reads transitition state geometries from network, https://iopscience.iop.org/article/10.1088/2632-2153/aba822."""
    # energies
    energiesurl = "https://archive.materialscloud.org/record/file?file_id=0eaa6011-b9d7-4c30-b424-2097dd90c77c&filename=energies.txt.gz&record_id=414"
    res = requests.get(energiesurl)
    webfh = io.BytesIO(res.content)
    with gzip.GzipFile(fileobj=webfh) as fh:
        lines = [_.decode("ascii") for _ in fh.readlines()]
    relevant = [
        _ for _ in lines if "transition-states/" in _ and ".xyz" in _ and "lccsd" in _
    ]
    filenames = [line.strip().split(",")[4] for line in relevant]
    energies = np.array([float(line.strip().split(",")[-2]) for line in relevant])
    # geometries
    geometriesurl = "https://archive.materialscloud.org/record/file?file_id=4905b29e-a989-48a3-8429-32e1db989972&filename=geometries.tgz&record_id=414"
    res = requests.get(geometriesurl)
    webfh = io.BytesIO(res.content)
    t = tarfile.open(fileobj=webfh)
    mols = {}
    for fo in t:
        if fo.name in filenames:
            lines = t.extractfile(fo).readlines()
            lines = [_.decode("ascii") for _ in lines]
            coord, nuc_charges = read_xyz(lines)
            mols[fo.name] = (coord, nuc_charges)
    cs = [Compound(name, mols[name][0], mols[name][1], e, None) for name, e in zip(filenames, energies)]
    return cs
```

+++ {"id": "Ta3IMP2V2NmM"}

### Beginning of energy prediction pipeline

```{code-cell} ipython3
:id: cN-x9NBV5KfO

# Solve linear matrix equation
# Returns x for Ax = y
def solve(A, y):
    return np.linalg.solve(A, y)
```

+++ {"id": "kMhIB57i2NmN"}

### Generate input features to perform a similarity measure
A straightforward way of representing a molecule, defined by the geometry $R$ and the nuclear charges $Z$, is the coloumb matrix.
The following function computes per compound a matrix $M$:

\begin{align}
M_{ij} = \left\{
        \begin{array}{cl}
         Z_{i}^{2.4} & \text{if } i = j \\
        \frac{Z_{i}Z_{j}}{| {R}_{i} - {R}_{j}|}       & \text{if } i \neq j
        \end{array}
        \right.
\end{align}

```{code-cell} ipython3
:id: j1ZQK_V3desI


def generate_coulomb_matrix(R, Z, size=23):
    natoms = R.shape[0]

    # add dummy variable to divide by zero
    dist = 1 / (np.linalg.norm(R[..., None, :] - R[None, :, :], axis=-1) + np.eye(natoms)) - np.eye(natoms)

    # scaling off-diagonal elements
    dist *= np.outer(Z[:, None], Z[None, :])

    # add diagonal elements
    dist[np.diag_indices(natoms)]= 0.5 * Z**(2.4)

    output = dist[np.triu_indices(natoms)].reshape(-1)
    output = np.concatenate([output, np.zeros(int((size+1)*size / 2) - output.shape[0])])
    return output
```

+++ {"id": "zYYI1Qqe2NmN"}

### Gaussian kernel function
The next step is to define a kernel function. Below a gaussian is choosen as a similarity metrics. Alternatives would be for example a Laplacian kernel or a linear kernel.

It takes as two datasets $A \in \mathbb{R}^{n_1 \times f}$ and $B \in \mathbb{R}^{n_2 \times f}$ for $n_1$ and $n_2$ being the number of samples of each dataset (representing in our case molecules) and $f$ represents the feature dimension, e.g. the feature dimension of the coulomb matrix.

```{code-cell} ipython3
:id: dEnM-y3g2NmN

def gaussian_kernel(A, B, sigma=1000):
    '''
    Returns a kernel matrix.

            Parameters:
                    A (np.array): 2D array of shape (samples 1, feature represention)
                    B (np.array): 2D array of shape (samples 2, feature represention)
                    sigma (int): Scaling factor for the kernel function
            Returns:
                    K (np.array): 2D array of shape (samples 1, samples 2)
    '''
    norm = np.linalg.norm(A[None, :, :] - B[:, None, :], axis=-1)
    K = np.exp(-norm/ (2*sigma**2))

    return K
```

+++ {"id": "ov0-QqT82NmO"}

### Define settings

```{code-cell} ipython3
:id: j-Dgr7IGdeue

# Define number of samples for training and predictions
# Large number of samples (> 1000) can be resource intensive (memory & compute)
nb_samples = 100

# Load data
compounds = database_qmrxn20()

# Shuffle
np.random.shuffle(compounds)

# Reduce the dataset
# qmrxn20 contains in total ~4k molecules with reference energies
compounds = compounds[:nb_samples]

# Define dataset split
dataset_size = int(0.5 * nb_samples)

# Scaling factor for gaussian kernel function
sigma = 4000

# Generate input features
X = np.stack([generate_coulomb_matrix(c.coordinates, c.nuclear_charges) for c in compounds])

# Prediction labels
y = np.stack([c.energy for c in compounds])

X_train = X[:dataset_size]
y_train = y[:dataset_size]

# Compute gaussian kernel
K_train = gaussian_kernel(X_train, X_train, sigma=sigma)
K_train[np.diag_indices_from(K_train)] += 1e-8

# Solve repression parameters
alpha = solve(K_train, y_train)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: iYQk12QCdew3
outputId: b5051900-adb0-40db-996a-79e00f75d5f0
---
# Prediction of test set
X_test = X[-dataset_size:]
y_test = y[-dataset_size:]

# Compute gaussian kernel between training set and test set
K_pred = gaussian_kernel(X_test, X_train, sigma=sigma)

# Prediction
y_pred = np.dot(alpha, K_pred)

# Calculate mean-absolute-error (MAE):
print(f"Mean absolute error: {np.mean(np.abs(y_pred - y_test))}")
```

+++ {"id": "VzDAHCaF2NmP"}

As one can see that for this rather small training and test set the mean absolute energy is rather inaccurate. This is partially due to a small training dataset but also connected to the coulomb matrix as input feature.

+++ {"id": "f3IZUk1n2NmP"}

### Part 2: Improved input features and kernel function

The following part is relying on the qml package. For installation instruction see their website: https://www.qmlcode.org/index.html
To use specific input features we need to install the developer branch:

`pip install git+https://github.com/qmlcode/qml.git@develop`

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 7M70iV6GhXxU
outputId: cdc011b6-f388-4133-88a7-1e73c50ad8dd
---
!pip install git+https://github.com/qmlcode/qml.git@develop
```

```{code-cell} ipython3
:id: ODj92FAdeGq_

import qml
from qml.representations import generate_fchl_acsf
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: wT6_AKOnoZU4
outputId: 8ccf0e37-d2bf-42f2-c7bc-8819026f62fb
---
# Define number of samples for training and predictions
# Large number of samples (> 1000) can be resource intensive (memory & compute)
nb_samples = 100

# Load data - See above for a description of the data function
compounds = database_qmrxn20()

# Shuffle
np.random.shuffle(compounds)

# Reduce the dataset
compounds = compounds[:nb_samples]

# Define dataset split
dataset_size = int(0.5 * nb_samples)

# Scaling factor for gaussian kernel function
sigma = 4000

# Define the elements appearing in the dataset and maximal number of atoms for all molecules
elements = [1, 35, 6, 7, 8, 9, 17]
nmax = 21
kwargs = {"elements":elements, "nRs2":12, "nRs3":5, "pad":nmax}

# Convert the molecule into input features based on the work:
# "FCHL revisited: Faster and more accurate quantum machine learning", Christensen et al. (2020), 10.1063/1.5126701.
X = np.array([generate_fchl_acsf(c.nuclear_charges, c.coordinates, **kwargs) for c in compounds])
Q = np.array([c.nuclear_charges for c in compounds])
y = np.stack([c.energy for c in compounds])

X_train = X[:dataset_size]
Q_train = Q[:dataset_size]
K_train = qml.kernels.get_local_kernel(X_train,  X_train, Q_train,  Q_train,  2000)

y_train = y[:dataset_size]
y_test = y[-dataset_size:]
```

```{code-cell} ipython3
:id: GcGe4FH23GOh

alpha = solve(K_train, y_train)
```

```{code-cell} ipython3
:id: 0_Ee6XlU3GKx

K_pred = qml.kernels.get_local_kernel(X[-dataset_size:], X_train,  Q[-dataset_size:], Q_train,  2000)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: c2yBqHoD3oB4
outputId: 39eaf2f0-9fcf-491e-e99f-440592bf0588
---
# Make the predictions
y_pred = np.dot(alpha, K_pred)
print("Test MAE:", np.mean(np.abs(y_pred - y_test)))
```

+++ {"id": "JLe_QuKNEgmH"}

### Part 3: Predict forces & energies

*Credits:* https://github.com/andersx/oqml-md/blob/master/python/utils.py

```{code-cell} ipython3
:id: sJr296002NmQ

from qml.math import svd_solve
```

```{code-cell} ipython3
:id: oYp_pUEEFZ5D

raw_data = np.load("h2co_ccsdt_avtz_4001.npz")
```

```{code-cell} ipython3
:id: VowppG9tJvN1

# Define number of samples for training and predictions
# Large number of samples (> 1000) can be resource intensive (memory & compute)
nb_samples = 100

# Scaling of kernel
sigma = 2 # prev. 2000

# Define dataset split
dataset_size = int(0.5 * nb_samples)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: qBfoOBUiF2JC
outputId: a79f81bf-32a7-403d-d5fb-deea56d73f5d
---
from tqdm import tqdm

max_atoms = max([len(_) for _ in raw_data["Z"]])
elements = sorted(list(set(raw_data["Z"].reshape(-1).tolist())))

x_train, x_test, dx_train, dx_test = [], [], [], []

train_idx = list(np.arange(dataset_size))
test_idx = list(np.arange(dataset_size, nb_samples, 1))

# Prepare input features
# "FCHL revisited: Faster and more accurate quantum machine learning", Christensen et al. (2020), DOI: 10.1063/1.5126701.
for i in tqdm(train_idx):
    x1, dx1 = generate_fchl_acsf(raw_data["Z"][i], raw_data["R"][i],
            elements=elements, gradients=True,
            pad=max_atoms)
    x_train.append(x1)
    dx_train.append(dx1)

for i in tqdm(test_idx):
    x1, dx1 = generate_fchl_acsf(raw_data["Z"][i], raw_data["R"][i],
            elements=elements, gradients=True,
            pad=max_atoms)
    x_test.append(x1)
    dx_test.append(dx1)

E = raw_data["E"]
F = raw_data["F"]
nuclear_charges = raw_data["Z"].tolist()

# Trainings data
X_train = np.array(x_train)
dX_train = np.array(dx_train)

E_train = E[train_idx]
F_train = F[train_idx]

Q_train  = [nuclear_charges[i] for i in train_idx]

# Test data
X_test = np.array(x_test)
dX_test = np.array(dx_test)

E_test = E[test_idx]
F_test = F[test_idx]

Q_test  = [nuclear_charges[i] for i in test_idx]
```

```{code-cell} ipython3
:id: Wqs_EQsTGhKQ

# Compute Kernel for energy & forces
Ke_train = qml.kernels.get_atomic_local_kernel(X_train, X_train, Q_train, Q_train, sigma)
Kf_train = qml.kernels.get_atomic_local_gradient_kernel(X_train, X_train, dX_train, Q_train, Q_train, sigma)

C = np.concatenate((Ke_train, Kf_train))
Y = np.concatenate((E_train, F_train.flatten()))

alphas = svd_solve(C, Y, rcond=1e-10)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 97V-2GLjIhf_
outputId: 15ce3c8e-c22f-4ad8-e53f-b220d38d254a
---
# Check if we can predict training set

y_pred_energy = np.dot(Ke_train, alphas)
print(f"Energy MAE: {np.mean(np.abs(y_pred_energy - E_train))}")

y_pred_force = np.dot(Kf_train, alphas)
print(f"Force MAE: {np.mean(np.abs(y_pred_force - F_train.flatten()))}")
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: knb4r7UKTO_X
outputId: 5c0f94b5-61e4-4dad-9b89-eefe649267de
---
# Test error
Ke_test = qml.kernels.get_atomic_local_kernel(X_train, X_test, Q_train, Q_test, sigma)
Kf_test = qml.kernels.get_atomic_local_gradient_kernel(X_train, X_test, dX_test, Q_train, Q_test, sigma)

y_pred_energy = np.dot(Ke_test, alphas)
print(f"Energy MAE: {np.mean(np.abs(y_pred_energy - E_test))}")

y_pred_force = np.dot(Kf_test, alphas)
print(f"Force MAE: {np.mean(np.abs(y_pred_force - F_test.flatten()))}")
```
