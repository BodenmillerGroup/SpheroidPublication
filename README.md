# This is the companion repository for the Spheroid Publication

It allows to run all analysis steps from raw data until the paper figures for the paper:  
"A quantitative analysis of the interplay of environment, neighborhood and cell state in 3D spheroids"  
https://doi.org/10.1101/2020.07.24.219659 

While already functional, the repository is *work in progress* and not yet well documented.

Also the data is currently on a ZENODO sandbox, thus might become unavailable at any time.

# Installation:
This workflow requires `snakemake` (https://snakemake.readthedocs.io/en/stable/getting_started/installation.html) 
as well as `singularity` (https://sylabs.io/guides/3.6/user-guide/quick_start.html#quick-installation-steps) to be installed.

It has only been tested on Ubuntu.

While the workflow can be run locally (>=8 cores, >32 RAM required), it is best run in a cluster environment
(e.g. SLURM, https://github.com/Snakemake-Profiles/slurm).

To retrieve the repository use:

`git clone --recurse-submodules https://github.com/BodenmillerGroup/SpheroidPublication.git`

Due to technical problems with snakemake subworkflows, currently the subworkflows need to be run independently in the order
described bellow.

To run a subworkflow:
- Change into it's main directory: e.g. `cd subworkflows/bf_preproc`
- Run snakemake with `conda` and `singularity` support: `snakemake --use-conda --use-singularity`

# Overview
The currently the workflow is split up into 5 different Snakemake workflows, represented 
by 5 different branches of this repository:
1) bf_preproc: Processing of brightfield images of spheres to quantify the sphere diameters as well as to
   identify misformed spheres. 
   Result:
   - 'results/hq_spheres.csv': A table of morphometric measurements, including diameter, of spheres that were identified
     to not being misformed.
   - 'results/plate_overviews': Brightfield images of all the plates analyzed
   - 'results/well_overviews': 1 .png image per well
   
2) phys_preprocessing: Preprocessing of the 4 cell line IMC dataset.
Segmentation of spheres and cells in IMC data as well as alignment of
IMC images with fluorescent slidescan images. Finally features on the cell level are measured.
    Result:
    - 'results/cpout': A cellprofiler output folder containing measurements:
        These measurments are of individual spheres cropped out of the original IMC images.
        - Object related measurements:
            - cell.csv: Cell mask measurements
            - cyto.csv: Cytoplasm mask measurements
            - nuclei.csv: Nuclei mask measurements
            - nucleiexp.csv: Slightly expanded nuclei mask measurements
        - Image.csv: Measurements/metadata related to 
        - Experiment.csv: Cellprofiler measurement run related metadata
        - Folder imgs/: All measured images as tiff stacks
        - Folder masks/: All measured object masks as greyscale tiffs
        
3) oexp_preprocessing: Preprocessing of the overexpression dataset. 
    Same output as phys_preprocessing.
    
4) phys_analysis: Analysis of the data preprocessed via phys_preproc

5) oexp_analysis: Analysis of the data preprocessed via oexp_preproc  

