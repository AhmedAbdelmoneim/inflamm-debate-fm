
### Raw data

Raw data is saved under raw_processing directory. This contains raw .tar files downloaded from GEO database.

To use the latest probe information and genome builds, custom CDFs are downloaded from Brain Array

#### Datasets

* GSE7404: Samples from mouse burn and trauma
* GSE19668: Samples from mouse sepsis
* GSE20524: Samples from mouse infection
* GSE28750: Samples from human sepsis
* GSE36809: Samples from human trauma
* GSE37069: Samples from human burn

### Pipeline

1. ./preprocess_gses.ipynb: Download and preprocess the raw GEO datasets to create clean AnnData objects, saved as .h5ad files.
2. ./differential_ge.ipynb: Perform differential expression analysis on the preprocessed datasets to identify significant genes associated with inflammation conditions.