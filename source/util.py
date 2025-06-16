import bisect
import os
import numpy as np


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
