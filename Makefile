.PHONY: run_all, run_all_slurm, prepare_envs

ADD_SINGULARITY_ARGS = #--singularity-args '\-u'
N_JOBS = 500
N_CORES = --cores 16
SNAKEMAKE_OPTS_SLURM = --profile slurm --use-singularity  --jobs $(N_JOBS) $(ADD_SINGULARITY_ARGS) \
		--use-conda --conda-frontend mamba

SNAKEMAKE_OPTS = --use-conda --conda-frontend mamba  --use-singularity $(ADD_SINGULARITY_ARGS)\
	 	$(N_CORES)

SNAKEMAKE_PREPARE =  --use-conda --use-singularity \
	 	$(N_CORES) --conda-create-envs-only

SNAKEMAKE_BIN = snakemake


run_all:
	cd subworkflows/bf_preproc && $(SNAKEMAKE_BIN) $(SNAKEMAKE_OPTS)
	cd subworkflows/phys_preproc && $(SNAKEMAKE_BIN) $(SNAKEMAKE_OPTS)
	cd subworkflows/oexp_preproc && $(SNAKEMAKE_BIN) $(SNAKEMAKE_OPTS)
	cd subworkflows/phys_analysis && $(SNAKEMAKE_BIN) $(SNAKEMAKE_OPTS)
	cd subworkflows/oexp_analysis && $(SNAKEMAKE_BIN) $(SNAKEMAKE_OPTS)

run_all_slurm:
	cd subworkflows/bf_preproc && $(SNAKEMAKE_BIN) $(SNAKEMAKE_OPTS_SLURM)
	cd subworkflows/phys_preproc && $(SNAKEMAKE_BIN) $(SNAKEMAKE_OPTS_SLURM)
	cd subworkflows/oexp_preproc && $(SNAKEMAKE_BIN) $(SNAKEMAKE_OPTS_SLURM)
	cd subworkflows/phys_analysis && $(SNAKEMAKE_BIN) $(SNAKEMAKE_OPTS_SLURM)
	cd subworkflows/oexp_analysis && $(SNAKEMAKE_BIN) $(SNAKEMAKE_OPTS_SLURM)

prepare_envs:
	cd subworkflows/bf_preproc && $(SNAKEMAKE_BIN) $(SNAKEMAKE_PREPARE)
	cd subworkflows/phys_preproc && $(SNAKEMAKE_BIN) $(SNAKEMAKE_PREPARE)
	cd subworkflows/oexp_preproc && $(SNAKEMAKE_BIN) $(SNAKEMAKE_PREPARE)
	cd subworkflows/phys_analysis && $(SNAKEMAKE_BIN) $(SNAKEMAKE_PREPARE)
	cd subworkflows/oexp_analysis && $(SNAKEMAKE_BIN) $(SNAKEMAKE_PREPARE)
