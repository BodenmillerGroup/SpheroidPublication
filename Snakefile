import urllib.request
import pathlib
import os
import shutil
import sys

sys.path.append("../../")
from scripts import helpers as hpr
from snakemake.utils import validate
from snakemake.io import strip_wildcard_constraints, expand
import pandas as pd

# Cellprofiler/Ilastik rules
include: '../../rules/cellprofiler.smk'
include: '../../rules/ilastik.smk'

# Read Configuration
configfile: 'config/config_bf.yml'
validate(config, "config/config_bf.schema.yml")

# Extract variables from configuration
## Input/output
input_data_folders = config['input_data_folders']
input_file_regexp = config['input_file_regexp']
folder_base = pathlib.Path(config['output_folder'])

fn_dat_input = 'config/input_files.csv'

## Ilastik run config
ilastik_container = config['ilastik_container']

# Cellprofiler default config
cellprofiler_container = config['cellprofiler_container']
cp_plugins = config['cellprofiler_plugins']
# Define hardcoded variables
## Define basic folder structrue
fn_cp_scaling =  os.path.abspath('cp_pipelines/1_rescale_bf.cppipe')
fn_cp_segmentation =  os.path.abspath('cp_pipelines/2_segment_bf.cppipe')
fn_cp_measurement =  os.path.abspath('cp_pipelines/3_measure_bf.cppipe')
fn_ilastik_proj = 'classifiers/bfsegmentation.ilp'

folder_analysis = folder_base / 'tiff'
folder_input = pathlib.Path('input_data')
folder_scaled = folder_analysis / ('scaled')
folder_probab = folder_analysis / ('probab')
folder_mask = folder_analysis / ('mask')
folder_cp = folder_base / ('cpout')
folder_cp_ov = folder_cp / 'overlays'

## Define Output files
fn_image = folder_cp / 'Image.csv'
fn_mask = folder_cp / 'mask.csv'
fn_experiment = folder_cp / 'Experiment.csv'

# Produce a list of all cellprofiler output files
cp_meas_output = [fn_image, fn_mask, fn_experiment]

# Identify a dictionary of input folders/zips containing .mcd files to process
dict_zip_fns = hpr.get_filenames_by_re(input_data_folders, input_file_regexp)

## Define suffixes
suffix_plate = '_p{platenr}'
suffix_scale = '_r5'
suffix_mask = '_mask'
suffix_probablities = '_Probabilities'
suffix_overlay = '_overlay'
suffix_tiff = '.tiff'

## Define derived file patterns
pat_basename_input = '{img_base}_Plate_{platenr, [0-9]+}/TimePoint_1/{img_base}{well}.TIF'
pat_basename_image = '{img_base}{well}'

pat_fn_input= folder_input / pat_basename_input
pat_fn_plate = folder_analysis / (f'{pat_basename_image}{suffix_plate}{suffix_tiff}')
pat_fn_scaled = folder_analysis / (f'{pat_fn_plate.stem}{suffix_scale}{suffix_tiff}')
pat_fn_probabilities = folder_analysis / (f'{pat_fn_scaled.stem}{suffix_probablities}{suffix_tiff}')
pat_fn_mask= folder_analysis / (f'{pat_fn_probabilities.stem}{suffix_mask}{suffix_tiff}')
pat_fn_overlay= folder_cp_ov / (f'{pat_fn_scaled.stem}{suffix_overlay}{suffix_tiff}')

done_rescale = folder_analysis / 'rescale.done'
fol_scale_combined =  folder_analysis / 'cp_rescale'/ 'combined'

FN_DICT = None
def get_dict():
    global FN_DICT
    if FN_DICT is None:
        dat = pd.read_csv(fn_dat_input)
        d = {str(out): str(inp) for inp, out in dat.loc[:, ['input_filename', 'output_filename']].values}
        FN_DICT = d
    else:
        d = FN_DICT
    return d

# Define dynamic files
## Define (dynamic) input file functions
def fkt_fns_input(wildcards):
    """
    Identifies the input images.
    :param wildcards: wildcards dynamically provided by snakemake
    :return: A list of all `.ome.tiffs` generated.
    """
    checkpoints.define_input_files.get()
    fns_output = pd.read_csv(fn_dat_input)['output_filename']
    fns = [str(folder_input / fn) for fn in fns_output]
    return fns

## Define derived (dynamic) input files functions
## This generates functions to define input filenames based on other input filename functions
fkt_fns_scaled = hpr.get_derived_input_fkt(fkt_fns_input, pat_fn_input, pat_fn_scaled)
fkt_fns_probabilities = hpr.get_derived_input_fkt(fkt_fns_scaled, pat_fn_scaled, pat_fn_probabilities)
fkt_fns_mask = hpr.get_derived_input_fkt(fkt_fns_scaled, pat_fn_scaled, pat_fn_mask)
fkt_fns_overlay = hpr.get_derived_input_fkt(fkt_fns_scaled, pat_fn_scaled, pat_fn_overlay)

# Configuration for cellprofiler pipeline steps
# (Please look at rules/cellprofiler.smk for the documentation of this structure)
config_dict_cp = {
    'rescale': {
        'run_size': 500,
        'plugins': cp_plugins,
        'pipeline': fn_cp_scaling,
        'input_files': [fkt_fns_input],
        'output_patterns': [pat_fn_scaled],
    },
    'segmasks': {
        'run_size': 500,
        'plugins': cp_plugins,
        'pipeline': fn_cp_segmentation,
        'input_files': [fkt_fns_probabilities],
        'output_patterns': [pat_fn_mask],
    },
    'measuremasks': {
        'run_size': 500,
        'plugins': cp_plugins,
        'pipeline': fn_cp_measurement,
        'input_files': [fkt_fns_mask, fkt_fns_scaled],
        'output_patterns': cp_meas_output + [pat_fn_overlay],
    }
}



# Configuration for Ilastik steps
# (Please look at rules/cellprofiler.smk for the documentation of this structure)
config_dict_ilastik = {
    'cell':
        {'project': fn_ilastik_proj,
         'run_size': 500,
         'output_format': 'tiff',
         'output_filename': f'{{nickname}}{suffix_probablities}{suffix_tiff}',
         'export_source': 'Probabilities',
         'export_dtype': 'uint16',
         'pipeline_result_drange': '"(0.0, 1.0)"',
         'input_files': fkt_fns_scaled,
         'output_pattern': pat_fn_probabilities
         }
}

# Target rules
rule all:
    input: cp_meas_output

rule scaled_imgs:
    input:
         fkt_fns_scaled

rule input_imgs:
    input:
         fkt_fns_input

checkpoint define_input_files:
    output:
        fn_dat_input
    run:
        fns = hpr.get_filenames_by_re(input_data_folders, input_file_regexp)
        dat_input = (pd.DataFrame({'input_filename': fns})
                 .assign(output_filename= lambda d: d['input_filename']
        .map(lambda fn: fn.relative_to(fn.parent.parent.parent)))
                 )
        dat_input.to_csv('config/input_files.csv', index=False)

def get_fn_from(wildcards):
    rel_fol = expand(strip_wildcard_constraints(str(pathlib.Path(pat_fn_input).relative_to(folder_input))), **wildcards)[0]
    return get_dict()[rel_fol]

rule retrieve_input_files:
    input: fn_dat_input
    output: pat_fn_input
    params:
        fn_input_file = get_fn_from
    threads: 1
    shell:
        'rsync -u {params.fn_input_file} {output[0]}'
    #run:
    #    fn_output = pathlib.Path(output[0])
    #    fn_base_output = fn_output.relative_to(folder_input)
    #    fn_input = params[0][str(fn_base_output)]
    #    pathlib.Path(fn_output).parent.mkdir(exist_ok=True)
    #    shutil.copy(fn_input, fn_output)

rule sync_to_cluster:
    shell:
        'rsync -rtu /home/vitoz/mnt/tmp/scratch/vitoz/Git/SpheroidPublication vizano@cluster.s3it.uzh.ch:/data/vizano/Git/ --progress --exclude=".*"'

## Rules to target Cellprofiler batch runs
define_cellprofiler_rules(config_dict_cp, folder_base, container_cp=cellprofiler_container)
define_ilastik_rules(config_dict_ilastik, folder_base, threads=4,
                     mem_mb=15000, container_ilastik=ilastik_container)

### Varia

rule clean:
    shell:
        "rm -R {folder_base}"