"""
Piboso classifier
Applies a pre-trained model

Marco Lui, February 2013
"""
from cPickle import load

import argparse, sys
import csv
from hydrat.store import Store

import numpy as np
import scipy.sparse as sp

from common import Timer

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("model", help="read model from")
  parser.add_argument("data", help="store containing pre-tokenized data")
  parser.add_argument("feat", help="store containing feature data")
  parser.add_argument("output", help="write output to PATH", metavar="PATH")
  args = parser.parse_args()

  features, L0_cl, L1_cl = load(open(args.model))
  fallback = Store(args.feat, 'r')
  store = Store(args.data, 'r', fallback=fallback)

  with Timer() as overall_timer:
    L0_preds = []
    for feat, cl in zip(features, L0_cl):
      fm = store.get_FeatureMap('NewDocuments', feat)
      # We need to trim the fv as the feature space may have grown when we tokenized more documents.
      # Hydrat's design is such that new features are appended to the end of a feature space, so
      # we can safely truncate the feature map.
      train_feat_count = cl.metadata['train_feat_count']
      assert(train_feat_count <= fm.raw.shape[1])
      fv = fm.raw[:,:train_feat_count]
      with Timer() as cl_timer:
        pred = cl(fv)
        print >>sys.stderr, "== L1 feat for {0} took {1:.2f}s ({2:.2f} inst/s) ==".format(feat, cl_timer.elapsed, cl_timer.rate(fv.shape[0]))
      L0_preds.append(pred)

    L0_preds = sp.csr_matrix(np.hstack(L0_preds))

    with Timer() as cl_timer:
      L1_preds = L1_cl(L0_preds)
      print >>sys.stderr, "== L1 classify took {0:.2f}s ({1:.2f} inst/s)==".format(cl_timer.elapsed, cl_timer.rate(L0_preds.shape[0]))
      
    print >>sys.stderr, "== classification took {0:.2f}s ({1:.2f} inst/s)==".format(overall_timer.elapsed, overall_timer.rate(L0_preds.shape[0]))

  cl_space = store.get_Space('ebmcat')
  instance_ids = store.get_Space('NewDocuments')

  with open(args.output,'w') as f:
    writer = csv.writer(f)
    for inst_id, cl_id in zip(instance_ids, L1_preds.argmax(axis=1)):
      cl_name = cl_space[cl_id]
      writer.writerow((inst_id, cl_name))
  
