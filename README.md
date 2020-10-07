# This is the companion repository for the Spheroid Publication

It allows to run all analysis steps from raw data until the paper figures for the paper:  
"A quantitative analysis of the interplay of environment, neighborhood and cell state in 3D spheroids"  
https://doi.org/10.1101/2020.07.24.219659 

After running this, you can find plots used for figures in the `results/figures` directories of the subworkflows.  
The run `Jupyter` analysis notebooks can be found in the `logs/` directories.

# Installation:
This workflow requires `snakemake` (tested: v5.18, https://snakemake.readthedocs.io/en/stable/getting_started/installation.html, > 5.18) 
as well as `singularity` (tested: v3.2.1, https://sylabs.io/guides/3.6/user-guide/quick_start.html#quick-installation-steps) to be installed.

It has only been tested on Ubuntu 18.04.

While the workflow can be run locally (>=8 cores, >32 RAM required), it is best run in a cluster environment
(e.g. SLURM, https://github.com/Snakemake-Profiles/slurm).

To retrieve the repository use:

`git clone --recurse-submodules https://github.com/BodenmillerGroup/SpheroidPublication.git`

Due to technical problems with snakemake subworkflows, the subworkflows need to be run independently in the order
described bellow, as otherwise the workflows will be run only single-threaded: https://github.com/snakemake/snakemake/issues/208

To run all subworkflows on a local machine you can use:

`make run_all`

to run them all in the correct order.

If you use a `slurm` cluster, it is required to setup a profile for snakemake (https://github.com/Snakemake-Profiles/slurm).

Then one could run all of them as `make run_all_slurm`.


If you want to run subworkflows manually:
- Change into it's main directory: e.g. `cd subworkflows/bf_preproc`
- Run snakemake with `conda` and `singularity` support: `snakemake --use-conda --use-singularity`

# Overview
The currently the workflow is split up into 5 different Snakemake workflows, represented 
by 5 different branches of this repository:
1) bf_preproc: Processing of brightfield images of spheres to quantify the sphere diameters as well as to
   identify misformed spheres. 
   Result:
   - 'results/hq_spheres.csv': A table of morphometric measurements (area, diameter...) of spheres that were identified
     to not being missformed.
   - 'results/plate_overviews': Brightfield overview images of all the plates analyzed
   - 'results/well_overviews': 1 .png image per well
   
2) phys_preproc: Preprocessing of the 4 cell line IMC dataset.
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
        
3) oexp_preproc: Preprocessing of the overexpression dataset. 
    Same output as phys_preproc.
    
4) phys_analysis: Analysis of the data preprocessed via phys_preproc

5) oexp_analysis: Analysis of the data preprocessed via oexp_preproc  

