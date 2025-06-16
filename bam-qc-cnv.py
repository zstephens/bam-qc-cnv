#!/usr/bin/env python
import argparse
import gzip
import joblib
import math
import matplotlib.pyplot as mpl
import numpy as np
import pathlib
import pysam
import re
import sys
import time

from matplotlib import gridspec
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from source.bimodal_gaussian import fit_bimodal_gaussian
from source.globals          import *
from source.ref_info         import REFFILE_NAMES, LEXICO_2_IND
from source.util             import compute_n50, find_indices_in_range, log_px, makedir, reads_2_cov, strip_polymerase_coords


def main(raw_args=None):
    parser = argparse.ArgumentParser(description='bam-qc-cnv.py')
    parser.add_argument('-i',  type=str, required=True,  help="* input.bam",   metavar='input.bam')
    parser.add_argument('-o',  type=str, required=True,  help="* output_dir/", metavar='output_dir/')
    parser.add_argument('-v',  type=str, required=False, help="input.vcf",                                      metavar='',          default='')
    parser.add_argument('-m',  type=str, required=False, help="5mc bedfile",                                    metavar='',          default='')
    parser.add_argument('-s',  type=str, required=False, help="sample name",                                    metavar='',          default='')
    parser.add_argument('-vw', type=str, required=False, help="variant filters (whitelist)",                    metavar='',          default='PASS')
    parser.add_argument('-vb', type=str, required=False, help="variant filters (blacklist)",                    metavar='',          default='weak_evidence,LowQual')
    parser.add_argument('-w',  type=int, required=False, help="window size for coverage",                       metavar='10000',     default=10000)
    parser.add_argument('-wc', type=int, required=False, help="window size for cnv prediction",                 metavar='1000000',   default=1000000)
    parser.add_argument('-wv', type=int, required=False, help="window size for var density",                    metavar='10000',     default=10000)
    parser.add_argument('-wm', type=int, required=False, help="window size for methylation",                    metavar='1000000',   default=1000000)
    parser.add_argument('-cv', type=int, required=False, help="minimum variants per window for cnv prediction", metavar='50',        default=50)
    parser.add_argument('-mv', type=int, required=False, help="minimum CpG sites per window for 5mc plotting",  metavar='20',        default=20)
    parser.add_argument('-q',  type=int, required=False, help="minimum MAPQ for including read in coverage",    metavar='0',         default=0)
    parser.add_argument('-b',  type=str, required=False, help="bed file of targeted regions",                   metavar='',          default='')
    parser.add_argument('-r',  type=str, required=False, help="refname: t2t / hg38 / hg19",                     metavar='hg38',      default='hg38')
    parser.add_argument('-rt', type=str, required=False, help="read type: hifi / clr / ont",                    metavar='hifi',      default='hifi')
    parser.add_argument('--report-cnvs', required=False, help="[EXPERIMENTAL] report CNVs",                     action='store_true', default=False)
    args = parser.parse_args()

    IN_BAM = args.i

    READ_MODE = args.rt
    if READ_MODE not in ['hifi', 'clr', 'ont']:
        print('Error: -rt must be either hifi, clr, or ont')
        exit(1)

    OUT_DIR = args.o
    if OUT_DIR[-1] != '/':
        OUT_DIR += '/'
    makedir(OUT_DIR)
    PLOT_DIR = OUT_DIR + 'plots/'
    makedir(PLOT_DIR)

    MIN_MAPQ = max(0, args.q)
    WINDOW_SIZE = max(1, args.w)
    VAR_WINDOW = max(1, args.wv)
    CNV_WINDOW = max(1, args.wc)
    METHYL_WINDOW = max(1, args.wm)
    CNV_NUM_INDS = int((CNV_WINDOW / VAR_WINDOW) + 0.5)

    CNV_MINVAR = max(1, args.cv)
    METHYL_MINSITES = max(1, args.mv)

    HOM_VAF_THRESH = 0.900

    # don't plot coverage if we're <= this many windows away from unstable regions
    BUFFER_UNSTABLE_COV = max(int(1000000/VAR_WINDOW), 1)

    # don't call CNVs if we're <= this many bp away from unstable regions
    BUFFER_UNSTABLE_CNV = 2000000

    # minimum number of windows involved to report CNV event out
    MIN_CNV_WINDOW_TO_REPORT = 3

    REF_VERS  = args.r
    BED_FILE  = args.b
    BED_5MC   = args.m
    IN_VCF    = args.v
    SAMP_NAME = args.s

    VAR_FILT_WHITELIST = args.vw.split(',')
    VAR_FILT_BLACKLIST = args.vb.split(',')

    REPORT_COPYNUM = args.report_cnvs

    if REPORT_COPYNUM and not(IN_VCF):
        print('Error: --report-cnv requires an input vcf (-v)')
        exit(1)

    OUT_NPZ = f'{OUT_DIR}cov.npz'
    VAF_NPZ = f'{OUT_DIR}vaf.npz'
    CNV_BED = f'{OUT_DIR}cnv.bed'
    if SAMP_NAME:
        OUT_NPZ = f'{OUT_DIR}cov_{SAMP_NAME}.npz'
        VAF_NPZ = f'{OUT_DIR}vaf_{SAMP_NAME}.npz'
        CNV_BED = f'{OUT_DIR}cnv_{SAMP_NAME}.bed'

    #
    # targeted region bedfile for coverage computation, also will be plotted
    #
    bed_regions = {}
    if BED_FILE:
        with open(BED_FILE,'r') as f:
            for line in f:
                splt = line.strip().split('\t')
                if len(splt) < 3: # malformed or empty line
                    continue
                if len(splt) >= 4:
                    bed_annot = ','.join(splt[3:])
                else:
                    bed_annot = ''
                if bed_annot == '':
                    bed_annot = f'{splt[0]}:{splt[1]}-{splt[2]}'
                if splt[0] not in bed_regions:
                    bed_regions[splt[0]] = []
                (p1, p2) = sorted([int(splt[1]), int(splt[2])])
                bed_regions[splt[0]].append((p1, p2, bed_annot))

    sim_path = str(pathlib.Path(__file__).resolve().parent)
    resource_dir = sim_path + '/resources/'
    CYTOBAND_BED = resource_dir + f'{REF_VERS}-cytoband.bed'
    cyto_by_chr = {}
    unstable_by_chr = {}
    with open(CYTOBAND_BED,'r') as f:
        for line in f:
            splt = line.strip().split('\t')
            if splt[0] not in cyto_by_chr:
                cyto_by_chr[splt[0]] = []
                unstable_by_chr[splt[0]] = []
            cyto_by_chr[splt[0]].append((int(splt[1]), int(splt[2]), splt[3], splt[4]))
            if splt[4] in UNSTABLE_REGION:
                unstable_by_chr[splt[0]].append((int(splt[1]), int(splt[2])))

    SVM_SCALAR = joblib.load(resource_dir + 'svm_scaler.pkl')
    SVM_MODEL = joblib.load(resource_dir + 'svm_classifier.pkl')

    prev_ref = None
    rnm_dict = {}
    alns_by_zmw = []    # alignment start/end per zmw
    rlen_by_zmw = []    # max tlen observed for each zmw
    covdat_by_ref = {}  #
    all_bed_result = []
    tt = time.perf_counter()

    if IN_BAM[-4:].lower() == '.bam':
        #
        if REF_VERS not in REFFILE_NAMES:
            print('Error: -r must be one of the following:')
            print(sorted(REFFILE_NAMES.keys()))
            exit(1)
        CONTIG_SIZES = REFFILE_NAMES[REF_VERS]
        print(f'using reference: {REF_VERS}')
        #
        qc_metrics = {}
        qc_metrics['bases_q20'] = 0

        #
        samfile = pysam.AlignmentFile(IN_BAM, "rb")
        refseqs = samfile.references
        #
        for aln in samfile.fetch(until_eof=True):
            splt = str(aln).split('\t')
            my_ref_ind  = splt[2].replace('#','')
            # pysam weirdness
            if my_ref_ind.isdigit():
                splt[2] = refseqs[int(my_ref_ind)]
            elif my_ref_ind == '-1':
                splt[2] = refseqs[-1]
            else:
                splt[2] = my_ref_ind
            #
            ref   = splt[2]
            pos   = int(splt[3])
            mapq  = int(splt[4])
            cigar = splt[5]

            if ref == '*':      # skip unmapped reads
                continue
            if mapq < MIN_MAPQ:
                continue
            #if pos > 1000000:  # for debugging purposes
            #   continue

            if not(aln.is_supplementary) and not(aln.is_secondary):
                qc_metrics['bases_q20'] += sum(1 for n in aln.query_qualities if n >= 20)

            if READ_MODE == 'clr':
                rnm = strip_polymerase_coords(splt[0])
                template_len = splt[0].split('/')[-1].split('_')
                template_len = int(template_len[1]) - int(template_len[0])
            elif READ_MODE in ['hifi', 'ont']:
                rnm = splt[0]
                template_len = len(splt[9])

            if ref != prev_ref:
                # compute coverage on previous ref now that we're done
                if prev_ref is not None and len(alns_by_zmw) and prev_ref in CONTIG_SIZES:
                    (covdat_by_ref[prev_ref], bed_results) = reads_2_cov(prev_ref, alns_by_zmw, OUT_DIR, CONTIG_SIZES, WINDOW_SIZE, bed_regions)
                    all_bed_result.extend(bed_results)
                    sys.stdout.write(f' ({int(time.perf_counter() - tt)} sec)\n')
                    sys.stdout.flush()
                # reset for next ref
                if ref in CONTIG_SIZES:
                    sys.stdout.write(f'processing reads on {ref}...')
                    sys.stdout.flush()
                    tt = time.perf_counter()
                else:
                    print('skipping reads on '+ref+'...')
                alns_by_zmw = []
                rnm_dict = {}
                prev_ref = ref

            if ref not in CONTIG_SIZES:
                continue

            letters = re.split(r"\d+",cigar)[1:]
            numbers = [int(n) for n in re.findall(r"\d+",cigar)]
            adj     = 0
            for i in range(len(letters)):
                if letters[i] in REF_CHAR:
                    adj += numbers[i]

            if rnm in rnm_dict:
                my_rind = rnm_dict[rnm]
            else:
                rnm_dict[rnm] = len(rnm_dict)
                my_rind       = len(rnm_dict)-1
                alns_by_zmw.append([])
                rlen_by_zmw.append(0)

            alns_by_zmw[my_rind].append((pos, pos+adj))
            rlen_by_zmw[my_rind] = max([rlen_by_zmw[my_rind], template_len])
        samfile.close()

        # qc
        qc_metrics['total_reads'] = len(rlen_by_zmw)
        qc_metrics['total_yield'] = sum(rlen_by_zmw)
        qc_metrics['readlength_mean'] = int(np.mean(rlen_by_zmw))
        qc_metrics['readlength_median'] = int(np.median(rlen_by_zmw))
        qc_metrics['readlength_n50'] = compute_n50(rlen_by_zmw)
        qc_keys = ['total_reads', 'total_yield', 'bases_q20', 'readlength_mean', 'readlength_median', 'readlength_n50']
        with open(f'{OUT_DIR}qc.tsv', 'w') as f:
            f.write('\t'.join(qc_keys) + '\n')
            f.write('\t'.join([str(qc_metrics[n]) for n in qc_keys]) + '\n')

        # readlength histogram
        fig = mpl.figure(1,figsize=(10,3.5))
        logbins = np.geomspace(1000, 1000000, 60)
        mpl.hist(rlen_by_zmw, bins=logbins, color='gray')
        mpl.xscale('log')
        mpl.xlim(1000, 1000000)
        mpl.grid(which='both', linestyle='--', alpha=0.5)
        mpl.ylabel('read count')
        mpl.legend([f'{len(rlen_by_zmw)} total reads'])
        mpl.tight_layout()
        mpl.savefig(f'{PLOT_DIR}readlengths{IMAGE_SUFFIX}')
        mpl.close(fig)

        # we probably we need to process the final ref, assuming no contigs beyond chrM
        if ref not in covdat_by_ref and len(alns_by_zmw) and ref in CONTIG_SIZES:
            (covdat_by_ref[ref], bed_results) = reads_2_cov(ref, alns_by_zmw, OUT_DIR, CONTIG_SIZES, WINDOW_SIZE, bed_regions)
            all_bed_result.extend(bed_results)
        sys.stdout.write(f' ({int(time.perf_counter() - tt)} sec)\n')
        sys.stdout.flush()

        # save output
        sorted_chr = [n[1] for n in sorted([(LEXICO_2_IND[k],k) for k in covdat_by_ref.keys()])]
        np.savez_compressed(OUT_NPZ, ref_vers=REF_VERS, window_size=WINDOW_SIZE, sorted_chr=sorted_chr, **covdat_by_ref)
    #
    elif IN_BAM[-4:].lower() == '.npz':
        print('reading from an existing npz archive instead of bam...')
        in_npz = np.load(IN_BAM)
        REF_VERS = str(in_npz['ref_vers'])
        WINDOW_SIZE = int(in_npz['window_size'])
        sorted_chr = in_npz['sorted_chr'].tolist()
        covdat_by_ref = {k:in_npz[k] for k in sorted_chr}
        print(f' - ignoring -r and instead using: {REF_VERS}')
        print(f' - ignoring -w and instead using: {WINDOW_SIZE}')
        CONTIG_SIZES = REFFILE_NAMES[REF_VERS]
        if BED_FILE:
            print('Warning: a bed file was specified but will be ignored because input is .npz')
    #
    else:
        print('Error: -i must be .bam or .npz')
        exit(1)

    #
    # READ VCFs
    # we're assuming vcfs have GT and AF fields, and are sorted
    #
    var_kde_by_chr = {}
    var_het_by_chr = {}
    var_hom_by_chr = {}
    het_dens_by_chr = {}
    hom_dens_by_chr = {}
    USING_VAR_NPZ = False
    if IN_VCF:
        if IN_VCF[-4:].lower() == '.vcf' or IN_VCF[-7:].lower() == '.vcf.gz':
            sys.stdout.write('reading input VCF...')
            sys.stdout.flush()
            tt = time.perf_counter()
            is_gzipped = True
            with gzip.open(IN_VCF, 'r') as fh:
                try:
                    fh.read(1)
                except OSError:
                    is_gzipped = False
            if is_gzipped:
                f = gzip.open(IN_VCF,'rt')
            else:
                f = open(IN_VCF,'r')
            for line in f:
                if line[0] != '#':
                    splt = line.strip().split('\t')
                    my_chr = splt[0]
                    my_pos = int(splt[1])
                    my_filt = splt[6].split(';')
                    # filters of interest:
                    # -- Clair3 / ClairS-TO: PASS,  NonSomatic
                    # -- Mutect2: germline, haplotype, panel_of_normals, weak_evidence
                    if any(n in VAR_FILT_BLACKLIST for n in my_filt):
                        continue
                    if any(n in VAR_FILT_WHITELIST for n in my_filt):
                        fmt_split = splt[8].split(':')
                        dat_split = splt[9].split(':')
                        if 'GT' in fmt_split and 'AF' in fmt_split:
                            ind_gt = fmt_split.index('GT')
                            ind_af = fmt_split.index('AF')
                            my_gt = dat_split[ind_gt]
                            my_af = float(dat_split[ind_af].split(',')[0]) # if multiallelic, take first
                            if my_gt == '1/1' or my_gt == '1|1' or my_af >= HOM_VAF_THRESH:
                                if my_chr not in var_hom_by_chr:
                                    var_hom_by_chr[my_chr] = [[], []]
                                var_hom_by_chr[my_chr][0].append(my_pos)
                                var_hom_by_chr[my_chr][1].append(my_af)
                            else:
                                if my_chr not in var_het_by_chr:
                                    var_het_by_chr[my_chr] = [[], []]
                                var_het_by_chr[my_chr][0].append(my_pos)
                                var_het_by_chr[my_chr][1].append(my_af)
                                #print(splt[0], splt[1], splt[3], splt[4], my_filt, my_gt, my_af)
            f.close()
            sys.stdout.write(f' ({int(time.perf_counter() - tt)} sec)\n')
            sys.stdout.flush()
            #
            # keep the raw calls for CNV prediction
            #
            for my_chr in sorted_chr:
                if my_chr in var_het_by_chr:
                    var_het_by_chr[my_chr] = np.array(var_het_by_chr[my_chr])
                else:
                    var_het_by_chr[my_chr] = np.array([[],[]])
                if my_chr in var_hom_by_chr:
                    var_hom_by_chr[my_chr] = np.array(var_hom_by_chr[my_chr])
                else:
                    var_hom_by_chr[my_chr] = np.array([[],[]])
        #
        elif IN_VCF[-4:].lower() == '.npz':
            print('reading from an existing npz archive instead of vcf...')
            in_npz = np.load(IN_VCF)
            if in_npz['extra_covwin'] != WINDOW_SIZE:
                print('Error: coverage window size in variant npz does not match.')
                exit(1)
            VAR_WINDOW = int(in_npz['extra_varwin'])
            VAR_FILT_WHITELIST = str(in_npz['var_filt_whitelist']).split(',')
            VAR_FILT_BLACKLIST = str(in_npz['var_filt_blacklist']).split(',')
            print(f' - ignoring -vd and instead using: {VAR_WINDOW}')
            print(f' - ignoring -vw and instead using: {VAR_FILT_WHITELIST}')
            print(f' - ignoring -vb and instead using: {VAR_FILT_BLACKLIST}')
            var_kde_by_chr = {k:in_npz[f'kde_{k}'] for k in sorted_chr}
            var_het_by_chr = {k:in_npz[f'het_{k}'] for k in sorted_chr}
            var_hom_by_chr = {k:in_npz[f'hom_{k}'] for k in sorted_chr}
            USING_VAR_NPZ = True
        #
        else:
            print('Error: -v must be .vcf or .vcf.gz or .npz')
            exit(1)

        #
        # compute het/hom variant densities
        #
        for my_chr in sorted_chr:
            my_dens = np.zeros((int(CONTIG_SIZES[my_chr]/VAR_WINDOW)+1), dtype='float')
            for my_vpos in var_het_by_chr[my_chr][0,:]:
                my_dens[int(my_vpos)//VAR_WINDOW] += 1.0
            het_dens_by_chr[my_chr] = np.array(my_dens, copy=True)
            #
            my_dens = np.zeros((int(CONTIG_SIZES[my_chr]/VAR_WINDOW)+1), dtype='float')
            for my_vpos in var_hom_by_chr[my_chr][0,:]:
                my_dens[int(my_vpos)//VAR_WINDOW] += 1.0
            hom_dens_by_chr[my_chr] = np.array(my_dens, copy=True)

    #
    # read 5mc bed file (currently expects pacbio cpgtools format where modscore is column 4)
    #
    methylation_dens = {}
    if BED_5MC:
        bed_methylation = {}
        with open(BED_5MC,'r') as f:
            for line in f:
                splt = line.strip().split('\t')
                if len(splt) <= 3: # malformed or empty line
                    continue
                if len(splt) >= 4:
                    my_modscore = float(splt[3])
                my_chr = splt[0]
                if my_chr not in bed_methylation:
                    bed_methylation[splt[0]] = [[], []]
                my_pos = int(splt[1])
                bed_methylation[my_chr][0].append(my_pos)
                bed_methylation[my_chr][1].append(my_modscore)
        #
        for my_chr in sorted_chr:
            my_dens = np.zeros((int(CONTIG_SIZES[my_chr]/METHYL_WINDOW)+1), dtype='float')
            if my_chr in bed_methylation:
                for mi in range(0, CONTIG_SIZES[my_chr], METHYL_WINDOW):
                    start_coords = mi
                    end_coords = mi + METHYL_WINDOW
                    (m_lb, m_ub) = find_indices_in_range(bed_methylation[my_chr][0], start_coords, end_coords)
                    my_window_methyl = bed_methylation[my_chr][1][m_lb:m_ub]
                    if len(my_window_methyl) >= METHYL_MINSITES:
                        my_dens[start_coords//METHYL_WINDOW] = np.mean(my_window_methyl)
            methylation_dens[my_chr] = np.array(my_dens, copy=True)

    #
    # determine average coverage across whole genome
    # -- in normal samples this will correspond to 2 copies, but in tumors it might be 3+
    #
    all_win = []
    masked_covdat_by_ref = {}
    for my_chr in sorted_chr:
        if my_chr in UNSTABLE_CHR or my_chr in EXCLUDE_JUST_FROM_COV:
            masked_covdat_by_ref[my_chr] = covdat_by_ref[my_chr]
            continue
        cy = np.copy(covdat_by_ref[my_chr])
        if my_chr in unstable_by_chr:
            for ur in unstable_by_chr[my_chr]:
                w1 = max(math.floor(ur[0]/WINDOW_SIZE) - BUFFER_UNSTABLE_COV, 0)
                w2 = min(math.ceil(ur[1]/WINDOW_SIZE) + BUFFER_UNSTABLE_COV, len(cy)-1)
                cy[w1:w2+1] = -1.0
        masked_covdat_by_ref[my_chr] = np.copy(cy)
        all_win.extend(cy[cy >= 0.0].tolist())
    all_avg_cov = (np.mean(all_win), np.median(all_win), np.std(all_win))

    # use median as our first choice
    if all_avg_cov[1] > 0.0:
        avg_log2 = np.log2(all_avg_cov[1])

    # if median is zero, lets try mean instead
    elif all_avg_cov[0] > 0.0:
        avg_log2 = np.log2(all_avg_cov[0])

    # if both are zero, we received an empty or corrupt alignment
    else:
        avg_log2 = 0.0

    fig = mpl.figure(1, figsize=(10,5))
    with np.errstate(divide='ignore', invalid='ignore'):
        mpl.hist(np.log2(all_win) - avg_log2, bins=300, range=[COV_YT[0], COV_YT[-1]])
    mpl.xticks(COV_YT, COV_YL)
    mpl.xlim([COV_YT[0], COV_YT[-1]])
    mpl.grid(which='both', linestyle='--', alpha=0.6)
    mpl.xlabel('normalized log2 depth')
    mpl.ylabel('bin count')
    mpl.tight_layout()
    mpl.savefig(f'{PLOT_DIR}depth-hist{IMAGE_SUFFIX}')
    mpl.close(fig)
    del all_win

    #
    #
    #

    fig_width_scalar = 11.5 / CONTIG_SIZES['chr1']
    fig_width_buffer = 0.5
    fig_width_min    = 2.0
    fig_height       = 5.0
    #
    plotted_cx_cy = {}
    cnv_bed_out = []
    training_data_out = []

    for my_chr in sorted_chr:
        sys.stdout.write(f'making plots for {my_chr}...')
        sys.stdout.flush()
        tt = time.perf_counter()
        #
        xt = np.arange(0,CONTIG_SIZES[my_chr],10000000)
        xl = [f'{n*10}M' for n in range(len(xt))]
        with np.errstate(divide='ignore', invalid='ignore'):
            #cy = np.log2(covdat_by_ref[my_chr]) - avg_log2
            cy = np.log2(masked_covdat_by_ref[my_chr]) - avg_log2
        cx = np.array([n*WINDOW_SIZE + WINDOW_SIZE/2 for n in range(len(cy))])
        plotted_cx_cy[my_chr] = (np.copy(cx), np.copy(cy))
        #
        if my_chr in UNSTABLE_CHR:
            print(' skipped')
            Z = np.zeros((KDE_NUMPOINTS_VAF, int(CONTIG_SIZES[my_chr]/WINDOW_SIZE)+1), dtype='float')
            var_kde_by_chr[my_chr] = np.array(Z, copy=True)
            continue

        #
        # PLOTTING
        #
        my_width = max(CONTIG_SIZES[my_chr]*fig_width_scalar + fig_width_buffer, fig_width_min)
        fig = mpl.figure(1, figsize=(my_width,fig_height), dpi=200)
        gs = gridspec.GridSpec(3, 1, height_ratios=[4,4,1])
        ax1 = mpl.subplot(gs[0])
        mpl.scatter(cx, cy, s=1, c='black')
        mpl.xlim(0, CONTIG_SIZES[my_chr])
        mpl.ylim(COV_YT[0], COV_YT[-1])
        mpl.xticks(xt,xl)
        mpl.yticks(COV_YT, COV_YL)
        mpl.grid(which='both', linestyle='--', alpha=0.6)
        mpl.ylabel('log2 cov change')
        for tick in ax1.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        #
        ax2 = mpl.subplot(gs[1])
        if USING_VAR_NPZ:
            Z = var_kde_by_chr[my_chr]
        else:
            Z = np.zeros((KDE_NUMPOINTS_VAF, int(CONTIG_SIZES[my_chr]/WINDOW_SIZE)+1), dtype='float')
            if my_chr in var_het_by_chr:
                for vi in range(var_het_by_chr[my_chr].shape[1]):
                    my_vpos = int(var_het_by_chr[my_chr][0,vi]/WINDOW_SIZE)
                    my_vvaf = int(var_het_by_chr[my_chr][1,vi]*KDE_NUMPOINTS_VAF)
                    my_std_pos = KDE_STD_POS/WINDOW_SIZE
                    my_pos_buff = int(my_std_pos*3) # go out 3 stds on either side
                    my_vaf_buff = int(KDE_STD_VAF*3)
                    for zy in range(max(0,my_vpos-my_pos_buff), min(Z.shape[1],my_vpos+my_pos_buff)):
                        for zx in range(max(0,my_vvaf-my_vaf_buff), min(Z.shape[0],my_vvaf+my_vaf_buff)):
                            Z[zx,zy] += np.exp(log_px(zx, zy, my_vvaf, my_vpos, KDE_STD_VAF, my_std_pos))
                for zy in range(Z.shape[1]):
                    my_sum = np.sum(Z[:,zy])
                    if my_sum > 0.0:
                        Z[:,zy] /= my_sum
            var_kde_by_chr[my_chr] = np.array(Z, copy=True)
        #
        X, Y = np.meshgrid(range(0,len(Z[0])+1), range(0,len(Z)+1))
        mpl.pcolormesh(X, Y, Z, rasterized=True)
        mpl.axis([0,len(Z[0]),0,len(Z)])
        mpl.yticks(KDE_YT, KDE_YL)
        mpl.ylabel('BAF')
        for tick in ax2.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        #
        ax3 = mpl.subplot(gs[2])
        if my_chr in cyto_by_chr:
            polygons = []
            p_color  = []
            p_alpha  = []
            for cdat in cyto_by_chr[my_chr]:
                pq = cdat[2][0]
                my_type = cdat[3]
                xp = [cdat[0], cdat[1]]
                yp = [-1, 1]
                if my_type == 'acen':
                    if pq == 'p':
                        polygons.append(Polygon(np.array([[xp[0],yp[0]], [xp[0],yp[1]], [xp[1],0]]), closed=True))
                    else:
                        polygons.append(Polygon(np.array([[xp[0],0], [xp[1],yp[1]], [xp[1],yp[0]]]), closed=True))
                else:
                    polygons.append(Polygon(np.array([[xp[0],yp[0]], [xp[0],yp[1]], [xp[1],yp[1]], [xp[1],yp[0]]]), closed=True))
                p_color.append(CYTOBAND_COLORS[my_type])
                p_alpha.append(0.8)
            for j in range(len(polygons)):
                ax3.add_collection(PatchCollection([polygons[j]], color=p_color[j], alpha=p_alpha[j], linewidth=0))
        mpl.xticks(xt,xl,rotation=70)
        mpl.yticks([],[])
        mpl.xlim(0,CONTIG_SIZES[my_chr])
        mpl.ylim(-1,1)
        mpl.ylabel(my_chr)
        mpl.tight_layout()
        mpl.savefig(f'{PLOT_DIR}cov_{my_chr}{IMAGE_SUFFIX}')
        mpl.close(fig)

        #
        # EXPERIMENTAL FEATURE: bimodal gaussian fit --> CNV calling
        #
        if REPORT_COPYNUM:
            sorted_het_coords = [int(n) for n in var_het_by_chr[my_chr][0,:]]
            sorted_het_vafs = [float(n) for n in var_het_by_chr[my_chr][1,:]]
            sorted_hom_coords = [int(n) for n in var_hom_by_chr[my_chr][0,:]]
            sorted_hom_vafs = [float(n) for n in var_hom_by_chr[my_chr][1,:]]
            cnv_bed_dat = []
            for vi in range(0, CONTIG_SIZES[my_chr], CNV_NUM_INDS * WINDOW_SIZE):
                start_coords = vi
                end_coords = vi + CNV_NUM_INDS * WINDOW_SIZE
                #
                my_cov_vector = plotted_cx_cy[my_chr][1][start_coords//WINDOW_SIZE:end_coords//WINDOW_SIZE]
                my_avg_cov = np.median(my_cov_vector)
                #
                (v_lb, v_ub) = find_indices_in_range(sorted_het_coords, start_coords, end_coords)
                my_window_hets = sorted_het_vafs[v_lb:v_ub]
                (v_lb_hom, v_ub_hom) = find_indices_in_range(sorted_hom_coords, start_coords, end_coords)
                my_window_homs = sorted_hom_vafs[v_lb_hom:v_ub_hom]
                if len(my_window_hets) + len(my_window_homs) > 0:
                    my_hethom_frac = len(my_window_hets) / (len(my_window_hets) + len(my_window_homs))
                else:
                    my_hethom_frac = -1.0
                #
                is_unstable = False
                svm_prediction = None
                if len(my_window_hets) >= CNV_MINVAR:
                    bimodal_fit = fit_bimodal_gaussian(np.array(my_window_hets))
                    #print(bimodal_fit)
                    if bimodal_fit is None:
                        is_unstable = True
                    else:
                        norm_ll_component_ratio = abs((bimodal_fit['component1_log_likelihood'] - bimodal_fit['component2_log_likelihood']) / len(my_window_hets))
                        if my_chr in unstable_by_chr:
                            for ur in unstable_by_chr[my_chr]:
                                if start_coords <= ur[1] + BUFFER_UNSTABLE_CNV and ur[0] - BUFFER_UNSTABLE_CNV <= end_coords:
                                    is_unstable = True
                                    break
                    #
                    # classification stuff
                    #
                    if is_unstable is False and bool(np.isnan(my_avg_cov)) is False:
                        feature_cov = my_avg_cov
                        if np.isneginf(feature_cov): # assume 0 copy region
                            feature_cov = -2.0
                        my_featurevec = [feature_cov,                                                         # log2_cov
                                         my_hethom_frac,                                                      # het_frac
                                         np.mean(my_window_hets),                                             # unimodal_u
                                         np.var(my_window_hets),                                              # unimodal_o2
                                         bimodal_fit['single_gaussian_log_likelihood'] / len(my_window_hets), # unimodal_nll
                                         abs(bimodal_fit['A']),                                               # bimodal_u_offset
                                         bimodal_fit['B'],                                                    # bimodal_o2
                                         bimodal_fit['max_log_likelihood'] / len(my_window_hets),             # bimodal_nll
                                         abs(norm_ll_component_ratio),                                        # nll_component_ratio
                                         bimodal_fit['single_gaussian_p-value']]                              # single_gaussian_pvalue
                        training_data_out.append(['', my_chr, start_coords, end_coords] + my_featurevec)
                        scaled_featurevec = SVM_SCALAR.transform(np.array(my_featurevec).reshape(1,-1)) # reshape to make a 2d matrix with a single sample
                        svm_prediction = SVM_MODEL.predict(scaled_featurevec)[0]
                #
                if svm_prediction is not None:
                    if svm_prediction == 6: # special case for LoH
                        cnv_assignment = -1
                    else:
                        cnv_assignment = svm_prediction
                    cnv_bed_dat.append((my_chr, start_coords, end_coords, cnv_assignment))
            #
            # merge windows into larger CNV calls
            #
            if cnv_bed_dat:
                cnv_windows = [[0, 1, cnv_bed_dat[0][3]]]
                current_copynum = cnv_bed_dat[0][3]
                for i,cbd in enumerate(cnv_bed_dat):
                    if i == 0:
                        continue
                    current_copynum = cnv_bed_dat[i][3]
                    if current_copynum == cnv_windows[-1][2]:
                        cnv_windows[-1][1] = i+1
                    else:
                        cnv_windows.append([i, i+1, current_copynum])
                for cw in cnv_windows:
                    if cw[1] - cw[0] < MIN_CNV_WINDOW_TO_REPORT:
                        continue
                    #avg_cnv_likelihood = np.mean([cnv_bed_dat[n][4] for n in range(cw[0], cw[1])])
                    #avg_cnv_coverage = np.mean([cnv_bed_dat[n][5] for n in range(cw[0], cw[1])])
                    #avg_hethom_frac = np.mean([cnv_bed_dat[n][6] for n in range(cw[0], cw[1])])
                    out_cnv_assignment = cw[2]
                    out_cnv_start = cnv_bed_dat[cw[0]][1]
                    out_cnv_end = cnv_bed_dat[cw[1]-1][2]
                    cnv_bed_out.append((my_chr, out_cnv_start, out_cnv_end, out_cnv_assignment))

        sys.stdout.write(f' ({int(time.perf_counter() - tt)} sec)\n')
        sys.stdout.flush()

    #
    # save parsed vcf data
    #
    if IN_VCF and USING_VAR_NPZ is False:
        var_npz_outdict = {}
        var_npz_outdict['extra_covwin'] = WINDOW_SIZE
        var_npz_outdict['extra_varwin'] = VAR_WINDOW
        #
        for my_chr in var_kde_by_chr.keys():
            var_npz_outdict[f'kde_{my_chr}'] = var_kde_by_chr[my_chr]
        for my_chr in var_het_by_chr.keys():
            var_npz_outdict[f'het_{my_chr}'] = var_het_by_chr[my_chr]
        for my_chr in var_hom_by_chr.keys():
            var_npz_outdict[f'hom_{my_chr}'] = var_hom_by_chr[my_chr]
        # save it all together in a single npz
        np.savez_compressed(VAF_NPZ, var_filt_whitelist=','.join(VAR_FILT_WHITELIST), var_filt_blacklist=','.join(VAR_FILT_BLACKLIST), **var_npz_outdict)

    #
    # whole genome plot (concatenated coverage)
    #
    plot_fn = f'{PLOT_DIR}cov_wholegenome{IMAGE_SUFFIX}'
    if SAMP_NAME:
        plot_fn = f'{PLOT_DIR}cov_wholegenome_{SAMP_NAME}{IMAGE_SUFFIX}'
    fig = mpl.figure(1, figsize=(30,10), dpi=200)
    gs = gridspec.GridSpec(4, 1, height_ratios=[4,4,4,2])
    #
    ax1 = mpl.subplot(gs[0])
    current_x_offset = 0
    current_color_ind = 0
    concat_var_matrix = None
    concat_het_dens = None
    concat_hom_dens = None
    concat_5mc_dens = None
    bed_region_polygons = []
    cnv_region_coords = []
    chrom_xticks_major = [0]
    chrom_xlabels_major = ['']
    chrom_xticks_minor = []
    chrom_xlabels_minor = []
    for my_chr in sorted_chr:
        if my_chr in UNSTABLE_CHR:
            continue
        my_color = CHROM_COLOR_CYCLE[current_color_ind % len(CHROM_COLOR_CYCLE)]
        (cx, cy) = plotted_cx_cy[my_chr]
        #
        if my_chr in var_kde_by_chr:
            Zvar = var_kde_by_chr[my_chr]
        else:
            Zvar = np.zeros((KDE_NUMPOINTS_VAF, int(CONTIG_SIZES[my_chr]/WINDOW_SIZE)+1), dtype='float')
        if my_chr in het_dens_by_chr:
            next_het_dens = het_dens_by_chr[my_chr]
        else:
            next_het_dens = np.zeros((int(CONTIG_SIZES[my_chr]/VAR_WINDOW)+1), dtype='float')
        if my_chr in hom_dens_by_chr:
            next_hom_dens = hom_dens_by_chr[my_chr]
        else:
            next_hom_dens = np.zeros((int(CONTIG_SIZES[my_chr]/VAR_WINDOW)+1), dtype='float')
        if my_chr in methylation_dens:
            next_5mc_dens = methylation_dens[my_chr]
        else:
            next_5mc_dens = np.zeros((int(CONTIG_SIZES[my_chr]/METHYL_WINDOW)+1), dtype='float')
        #
        if my_chr == sorted_chr[0]:
            concat_var_matrix = Zvar
            concat_het_dens = next_het_dens
            concat_hom_dens = next_hom_dens
            concat_5mc_dens = next_5mc_dens
        else:
            concat_var_matrix = np.concatenate((concat_var_matrix, Zvar), axis=1)
            concat_het_dens = np.concatenate((concat_het_dens, next_het_dens), axis=0)
            concat_hom_dens = np.concatenate((concat_hom_dens, next_hom_dens), axis=0)
            concat_5mc_dens = np.concatenate((concat_5mc_dens, next_5mc_dens), axis=0)
        #
        if my_chr in bed_regions:
            for (bed_start, bed_end, bed_annot) in bed_regions[my_chr]:
                xp = [bed_start + current_x_offset, bed_end + current_x_offset]
                yp = [-1, 1]
                bed_region_polygons.append(Polygon(np.array([[xp[0],yp[0]], [xp[0],yp[1]], [xp[1],yp[1]], [xp[1],yp[0]]]), closed=True))
        #
        for cnv_tuple in cnv_bed_out:
            if cnv_tuple[0] == my_chr:
                cnv_region_coords.append([[cnv_tuple[1] + current_x_offset, cnv_tuple[2] + current_x_offset], [cnv_tuple[3], cnv_tuple[3]]])
        #
        if my_chr in unstable_by_chr:
            for ur in unstable_by_chr[my_chr]:
                w1 = max(math.floor(ur[0]/WINDOW_SIZE), 0)
                w2 = min(math.ceil(ur[1]/WINDOW_SIZE), len(cy)-1)
                cy[w1:w2+1] = COV_YT[0] - 1.0
        #
        mpl.scatter(cx + current_x_offset, cy, s=0.5, color=my_color)
        chrom_xticks_minor.append(current_x_offset + 0.5 * len(cx) * WINDOW_SIZE)
        chrom_xlabels_minor.append(my_chr)
        chrom_xticks_major.append(current_x_offset + len(cx) * WINDOW_SIZE)
        chrom_xlabels_major.append('')
        current_x_offset += len(cx) * WINDOW_SIZE
        current_color_ind += 1
    mpl.xticks(chrom_xticks_major, chrom_xlabels_major)
    mpl.xticks(chrom_xticks_minor, chrom_xlabels_minor, minor=True)
    mpl.xlim([0, current_x_offset])
    mpl.ylim(COV_YT[0], COV_YT[-1])
    mpl.grid(which='major', linestyle='--', alpha=0.6)
    mpl.ylabel('log2 cov change')
    mpl.title(f'{SAMP_NAME} - average coverage: {all_avg_cov[0]:0.3f}')
    for tick in ax1.xaxis.get_minor_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
    #
    ax2 = mpl.subplot(gs[1])
    Z = concat_var_matrix
    X, Y = np.meshgrid(range(0,len(Z[0])+1), range(0,len(Z)+1))
    mpl.pcolormesh(X, Y, Z, rasterized=True)
    mpl.axis([0,len(Z[0]),0,len(Z)])
    mpl.yticks(KDE_YT, KDE_YL)
    mpl.ylabel('BAF')
    for tick in ax2.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    #
    ax3 = mpl.subplot(gs[2])
    scale_factor = VAR_WINDOW / 1000000
    if np.sum(concat_het_dens) > 0.0 or np.sum(concat_hom_dens) > 0.0:
        mpl.semilogy([VAR_WINDOW*n for n in range(len(concat_het_dens))], concat_het_dens*scale_factor, color='blue', alpha=0.5)
        mpl.semilogy([VAR_WINDOW*n for n in range(len(concat_hom_dens))], concat_hom_dens*scale_factor, color='red', alpha=0.5)
    mpl.xticks(chrom_xticks_major, chrom_xlabels_major)
    #mpl.xticks(chrom_xticks_minor, chrom_xlabels_minor, minor=True)
    mpl.xlim(0,VAR_WINDOW*len(concat_het_dens))
    mpl.ylim(0.1, 100)
    mpl.grid(which='major', linestyle='--', alpha=0.6)
    mpl.ylabel('variants / Mb')
    for tick in ax3.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
    #
    ####ax4 = mpl.subplot(gs[3])
    ####mpl.scatter([METHYL_WINDOW*n for n in range(len(concat_5mc_dens))], concat_5mc_dens, s=0.5, color='red', alpha=0.7)
    ####mpl.xlim(0,METHYL_WINDOW*len(concat_5mc_dens))
    ####mpl.ylim(0,100)
    ####mpl.ylabel('methylation %')
    ####ax4.xaxis.get_major_formatter().set_scientific(False)
    ####for tick in ax4.xaxis.get_major_ticks():
    ####    tick.tick1line.set_visible(False)
    ####    tick.tick2line.set_visible(False)
    ####    tick.label1.set_visible(False)
    ####    tick.label2.set_visible(False)
    #
    ####ax4 = mpl.subplot(gs[3])
    ####for bed_poly in bed_region_polygons:
    ####    ax4.add_collection(PatchCollection([bed_poly], color='green', alpha=0.8, linewidth=0))
    ####mpl.yticks([],[])
    ####mpl.xlim([0,current_x_offset])
    ####mpl.ylim([-1,1])
    ####ax4.xaxis.get_major_formatter().set_scientific(False)
    ####for tick in ax4.xaxis.get_major_ticks():
    ####    tick.tick1line.set_visible(False)
    ####    tick.tick2line.set_visible(False)
    ####    tick.label1.set_visible(False)
    ####    tick.label2.set_visible(False)
    #
    ax4 = mpl.subplot(gs[3])
    for [cnv_x, cnv_y] in cnv_region_coords:
        my_color = 'black'
        if cnv_y[0] <= 1:
            my_color = 'red'
        if cnv_y[0] < 0:
            my_color = 'green'
        if cnv_y[0] > 2:
            my_color = 'blue'
        mpl.plot(cnv_x, cnv_y, color=my_color, linewidth=2)
    mpl.xticks(chrom_xticks_major, chrom_xlabels_major)
    mpl.yticks([-1,0,1,2,3,4,5], ['loh', '0', '1', '2', '3', '4', '≥5'])
    mpl.xlim([0, current_x_offset])
    mpl.ylim(-2, 6)
    mpl.grid(which='major', linestyle='--', alpha=0.6)
    mpl.ylabel('CNV')
    for tick in ax4.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
    #
    mpl.tight_layout()
    mpl.savefig(plot_fn)
    mpl.close(fig)

    #
    # whole genome plot (stacked annotations)
    #
    pass

    if cnv_bed_out:
        with open(CNV_BED, 'w') as f:
            for n in cnv_bed_out:
                if n[3] != 2: # don't report diploid
                    f.write(f'{n[0]}\t{n[1]}\t{n[2]}\t{n[3]}\n')

    #if training_data_out:
    #    with open(f'{OUT_DIR}training_data.tsv','w') as f:
    #        for tdat in training_data_out:
    #            f.write('\t'.join([str(n) for n in tdat]) + '\n')

    print(f'average coverage: {all_avg_cov[0]:0.3f}')
    if len(all_bed_result):
        print('region coverage:')
        for n in all_bed_result:
            print(f' - {n[0][2]}: {n[1]:0.3f}')
    with open(f'{OUT_DIR}region_coverage.tsv','w') as f:
        f.write('region\tmean_cov\tmedian_cov\tstd\n')
        f.write(f'whole_genome\t{all_avg_cov[0]:0.3f}\t{all_avg_cov[1]:0.3f}\t{all_avg_cov[2]:0.3f}\n')
        for n in all_bed_result:
            f.write(f'{n[0][2]}\t{n[1]:0.3f}\t{n[2]:0.3f}\t{n[3]:0.3f}\n')


if __name__ == '__main__':
    main()
