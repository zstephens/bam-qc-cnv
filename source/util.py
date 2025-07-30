import bisect
import os
import pysam
import numpy as np
import sys
import time

from collections import defaultdict


def exists_and_is_nonZero(fn):
    if os.path.isfile(fn):
        if os.path.getsize(fn) > 0:
            return True
    return False


def makedir(d):
    if not os.path.isdir(d):
        os.system('mkdir '+d)


def rm(fn):
    if os.path.isfile(fn):
        os.system('rm '+fn)


def strip_polymerase_coords(rn):
    return '/'.join(rn.split('/')[:-1])


def reads_2_cov(my_chr, readpos_list_all, out_dir, CONTIG_SIZES, WINDOW_SIZE, bed_regions):
    #
    if my_chr not in CONTIG_SIZES:
        print(' - skipping coverage computation for '+my_chr+'...')
        return None
    #
    cov = np.zeros(CONTIG_SIZES[my_chr], dtype='<i4')
    # collapse overlapping alignments
    for readpos_list in readpos_list_all:
        if len(readpos_list_all) == 0:
            continue
        found_overlaps = True
        while found_overlaps:
            found_overlaps = False
            for i in range(len(readpos_list)):
                for j in range(i+1,len(readpos_list)):
                    (x1, x2) = readpos_list[i]
                    (y1, y2) = readpos_list[j]
                    if x1 <= y2 and y1 <= x2:
                        found_overlaps = True
                        readpos_list[i] = (min([x1,y1]), max([x2,y2]))
                        del readpos_list[j]
                        break
                if found_overlaps:
                    break
        for rspan in readpos_list:
            cov[rspan[0]:rspan[1]] += 1
    # report coverage for specific bed regions
    bed_out = []
    if my_chr in bed_regions:
        for br in bed_regions[my_chr]:
            b1 = max(br[0],0)
            b2 = min(br[1],len(cov))
            if b2 - b1 <= 0 or len(cov[b1:b2]) == 0:
                bed_out.append([br, 0.0, 0.0, 0.0])
            else:
                bed_out.append([br, np.mean(cov[b1:b2]), np.median(cov[b1:b2]), np.std(cov[b1:b2])])
    # downsample
    if WINDOW_SIZE <= 1:
        return (cov, bed_out)
    out_cov = []
    for i in range(0,len(cov),WINDOW_SIZE):
        out_cov.append(np.mean(cov[i:i+WINDOW_SIZE]))
    cov = np.array(out_cov)
    return (cov, bed_out)


def compute_n50(readlens):
    sorted_lengths = sorted(readlens, reverse=True)
    total_length = sum(sorted_lengths)
    threshold = total_length * 0.5
    cumulative_sum = 0
    for length in sorted_lengths:
        cumulative_sum += length
        if cumulative_sum >= threshold:
            return length
    return None


def log_px(x, y, ux, uy, ox, oy):
    out = -0.5*np.log(2.0*np.pi*ox) - ((x-ux)*(x-ux))/(2*ox*ox) - 0.5*np.log(2.0*np.pi*oy) - ((y-uy)*(y-uy))/(2*oy*oy)
    return out


def find_indices_in_range(sorted_list, lb, ub):
    start_idx = bisect.bisect_left(sorted_list, lb)
    end_idx = bisect.bisect_right(sorted_list, ub)
    return (start_idx, end_idx)


def merge_intervals_fast(intervals):
    if not intervals:
        return []
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # overlap
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged


def reads_2_cov_faster(my_chr, readpos_list_all, CONTIG_SIZES, WINDOW_SIZE, bed_regions):
    all_intervals = []
    for readpos_list in readpos_list_all:
        if readpos_list:
            readpos_list.sort()
            merged = merge_intervals_fast(readpos_list)
            all_intervals.extend(merged)

    chr_length = CONTIG_SIZES[my_chr]
    cov = np.zeros(chr_length, dtype=np.int32)
    if all_intervals:
        all_intervals.sort()
        for start, end in all_intervals:
            start = max(0, start)
            end = min(chr_length, end)
            if start < end:
                cov[start:end] += 1

    bed_out = []
    if my_chr in bed_regions:
        for br in bed_regions[my_chr]:
            b1 = max(br[0], 0)
            b2 = min(br[1], len(cov))
            if b2 - b1 <= 0:
                bed_out.append([br, 0.0, 0.0, 0.0])
            else:
                region_cov = cov[b1:b2]
                bed_out.append([br, np.mean(region_cov), np.median(region_cov), np.std(region_cov)])

    if WINDOW_SIZE > 1:
        n_windows = len(cov) // WINDOW_SIZE
        if n_windows > 0:
            reshaped = cov[:n_windows * WINDOW_SIZE].reshape(n_windows, WINDOW_SIZE)
            cov = np.mean(reshaped, axis=1)

    return (cov, bed_out)


def process_bam_coverage(IN_BAM, CONTIG_SIZES, WINDOW_SIZE, bed_regions, MIN_MAPQ, READ_MODE):
    read_metrics = {}  # rnm -> {'q20_bases': int, 'length': int, 'counted': bool}
    qc_metrics = {'bases_q20': 0}
    covdat_by_ref = {}
    all_bed_result = []

    samfile = pysam.AlignmentFile(IN_BAM, "rb")

    for ref_name in CONTIG_SIZES:
        if ref_name not in samfile.references:
            continue

        sys.stdout.write(f'Processing reads on {ref_name}...')
        sys.stdout.flush()
        tt = time.perf_counter()

        read_spans = defaultdict(list)
        for aln in samfile.fetch(contig=ref_name):
            # Get read name efficiently
            if READ_MODE == 'clr':
                rnm = strip_polymerase_coords(aln.query_name)
                # extract template length from read name for CLR
                try:
                    template_coords = aln.query_name.split('/')[-1].split('_')
                    template_len = int(template_coords[1]) - int(template_coords[0])
                except:
                    template_len = len(aln.query_sequence) if aln.query_sequence else 0
            else:
                rnm = aln.query_name
                template_len = len(aln.query_sequence) if aln.query_sequence else 0

            # metrics for this specific read
            if rnm not in read_metrics:
                read_metrics[rnm] = {'q20_bases': 0, 'length': template_len, 'counted': False}
            read_metrics[rnm]['length'] = max(read_metrics[rnm]['length'], template_len)

            # count Q20 bases
            if (not read_metrics[rnm]['counted'] and not aln.is_supplementary and not aln.is_secondary and aln.query_qualities is not None):
                read_metrics[rnm]['q20_bases'] = sum(1 for q in aln.query_qualities if q >= 20)
                read_metrics[rnm]['counted'] = True

            if aln.mapping_quality < MIN_MAPQ:
                continue
            if aln.is_supplementary or aln.is_secondary:
                continue
            if aln.is_unmapped:
                continue

            start_pos = aln.reference_start
            end_pos = aln.reference_end
            if start_pos is not None and end_pos is not None:
                read_spans[rnm].append((start_pos, end_pos))

        alns_by_zmw = list(read_spans.values())
        if alns_by_zmw:
            (covdat_by_ref[ref_name], bed_results) = reads_2_cov_faster(ref_name, alns_by_zmw, CONTIG_SIZES, WINDOW_SIZE, bed_regions)
            all_bed_result.extend(bed_results)

        sys.stdout.write(f' ({int(time.perf_counter() - tt)} sec)\n')
        sys.stdout.flush()

    samfile.close()

    # Compute final QC metrics from collected read data
    rlen_by_zmw = [metrics['length'] for metrics in read_metrics.values()]
    qc_metrics['bases_q20'] = sum(metrics['q20_bases'] for metrics in read_metrics.values())
    qc_metrics['total_reads'] = len(rlen_by_zmw)
    qc_metrics['total_yield'] = sum(rlen_by_zmw)

    if rlen_by_zmw:
        qc_metrics['readlength_mean'] = int(np.mean(rlen_by_zmw))
        qc_metrics['readlength_median'] = int(np.median(rlen_by_zmw))
        qc_metrics['readlength_n50'] = compute_n50(rlen_by_zmw)
    else:
        qc_metrics['readlength_mean'] = 0
        qc_metrics['readlength_median'] = 0
        qc_metrics['readlength_n50'] = 0

    return covdat_by_ref, all_bed_result, qc_metrics, rlen_by_zmw
