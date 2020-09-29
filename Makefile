.PHONY: run_all, run_all_slurm

SNAKEMAKE_OPTS_SLURM = --profile slurm --use-singularity --singularity-args '\-u' --jobs 500 \
		--use-conda --conda-frontend mamba

SNAKEMAKE_OPTS = --use-conda --conda-frontend mamba  --use-singularity --singularity-args '\-u'\
	 	--cores 16
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
