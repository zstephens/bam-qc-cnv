
REF_CHAR = 'MX=D'

CYTOBAND_COLORS = {'gneg':(255,255,255),
                   'gpos25':(225,204,230),
                   'gpos50':(175,119,187),
                   'gpos75':(124,68,136),
                   'gpos100':(90,55,103),
                   'acen':(139,112,144),
                   'stalk':(139,112,144),
                   'gvar':(231,214,234),
                   'mappability_gneg':(230,230,230),
                   'mappability_gpos50':(170,170,170)}
CYTOBAND_COLORS = {k:(v[0]/255.,v[1]/255.,v[2]/255.) for k,v in CYTOBAND_COLORS.items()}

CHROM_COLOR_CYCLE = [(32, 119, 184),
                     (245, 126, 19),
                     (43, 159, 44),
                     (214, 37, 42),
                     (143, 103, 182),
                     (139, 85, 78),
                     (226, 120, 194),
                     (126, 126, 126),
                     (189, 188, 36),
                     (27, 189, 210)]
CHROM_COLOR_CYCLE = [(v[0]/255.,v[1]/255.,v[2]/255.) for v in CHROM_COLOR_CYCLE]

# mask these regions when computing average coverage
#  - 'mappability' was manually added to the cytoband file for regions short reads were struggling with
UNSTABLE_REGION = ['acen', 'gvar', 'stalk', 'mappability_gneg', 'mappability_gpos50']
UNSTABLE_CHR = ['chrY', 'chrM']
EXCLUDE_JUST_FROM_COV = ['chrX']

COV_YT = range(-3,3+1)
COV_YL = [str(n) for n in COV_YT]
KDE_NUMPOINTS_VAF = 50
KDE_STD_VAF = 0.020*KDE_NUMPOINTS_VAF
KDE_STD_POS = 20000
KDE_YT = [0.0, 0.25*KDE_NUMPOINTS_VAF, 0.50*KDE_NUMPOINTS_VAF, 0.75*KDE_NUMPOINTS_VAF, KDE_NUMPOINTS_VAF]
KDE_YL = ['0%', '25%', '50%', '75%', '100%']
KDE_YL = ['0', '.25', '.50', '.75', '1']

IMAGE_SUFFIX = '.png'
