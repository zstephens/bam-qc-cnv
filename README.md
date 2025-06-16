# bam-qc-cnv
A script for collecting quality control statistics, plotting coverage depths / b-allele frequencies, and reporting CNVs from input BAM + VCF files.

## Installation

```bash
# download repository
git clone https://github.com/zstephens/bam-qc-cnv.git
cd bam-qc-cnv

# create and activate conda environment
conda env create -f conda_env.yaml
conda activate bam-qc-cnv

# test the application
python bam-qc-cnv.py --help
```

## Usage

(1) Get QC statistics and coverage plot:

```bash
python bam-qc-cnv.py -i aligned.bam -o results_dir/
```

(2) Include B-allele frequency plots from variant data:

```bash
python bam-qc-cnv.py -i aligned.bam -o results_dir/ -v variants.vcf
```

(3) Report CNVs (experimental feature):

```bash
python bam-qc-cnv.py -i aligned.bam -o results_dir/ -v variants.vcf --report-cnvs
```
