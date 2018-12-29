import optparse
import sys
import os
import logging
from collections import defaultdict
from itertools import islice

optparser = optparse.OptionParser()

optparser.add_option("-d", "--datadir", dest="datadir", default="data",
                     help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards",
                     help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en",
                     help="suffix of English (target language) filename \
                     (default=en)")
optparser.add_option("-f", "--french", dest="french", default="fr",
                     help="suffix of French (source language) filename \
                     (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None,
                     help="filename for logging output")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5,
                     type="float", help="threshold for alignment (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents",
                     default=sys.maxsize, type="int",
                     help="Number of sentences to use for training and alignment")

(opts, _) = optparser.parse_args()

f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

def align():
    sys.stderr.write("Training IBM Model 1 (no nulls) with Expectation Maximization...\n")
    bitext = [[sentence.strip().split() for sentence in pair] for pair
              in islice(zip(open(f_data), open(e_data)), opts.num_sents)]

    f_count = defaultdict(int)
    e_count = defaultdict(int)
    fe_count = defaultdict(int)

    for (f, e) in bitext:
        for f_i in set(f):
            f_count[f_i] += 1
            for e_j in set(e):
                fe_count[(f_i, e_j)] += 1
        for e_j in set(e):
            e_count[e_j] += 1

    # Initializations
    num_iter = 5
    Vf_size = len(f_count)  # size of french vocabulary
    nulls = 10 # size of artificial nulls
    t = defaultdict(lambda: nulls / Vf_size)
    n = 100  # n smoothing parameter
    V = 100000 # V smoothing parameter

    for k in range(num_iter):
        f_count = defaultdict(int)
        e_count = defaultdict(int)
        fe_count = defaultdict(int)

        for (f, e) in bitext:
            for f_i in set(f):
                Z = 0
                for e_j in set(e):
                    Z += t[(f_i, e_j)]
                for e_j in set(e):
                    # Smoothing implementation
                    # c = (t[(f_i, e_j)] + n) / (Z + n*V)

                    # Regular implementation
                    c = t[(f_i, e_j)] / Z

                    fe_count[(f_i, e_j)] += c
                    e_count[e_j] += c

        for (f, e) in fe_count.keys():
            t[(f, e)] = fe_count[(f, e)] / e_count[e]
    # Best Alignment
    sys.stderr.write("Aligning...")
    for (f, e) in bitext:
        for (i, f_i) in enumerate(f):
            bestp = 0
            bestj = 0
            for (j, e_j) in enumerate(e):
                if t[(f_i, e_j)] > bestp:
                    bestp = t[(f_i, e_j)]
                    bestj = j
            if bestp > opts.threshold:
                sys.stdout.write("%i-%i " % (i, bestj))
        sys.stdout.write("\n")

if __name__ == "__main__":
    align()
