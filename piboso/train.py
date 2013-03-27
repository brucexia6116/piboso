"""
Train the PIBOSO tagger using the ALTA2012 full dataset.
This consists of training two sets of classifiers: The L0 classifiers,
one per feature set, and the L1 classifier.

Marco Lui, March 2013
"""
import argparse, sys
from cPickle import dump

import numpy as np

from hydrat.store import Store
from hydrat.proxy import DataProxy
from hydrat.experiment import Experiment
import hydrat.classifier.liblinear as liblinear
import hydrat.classifier.meta.repeat as repeat 

import features
from corpora import ALTA2012Full
from common import Timer 

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--feats', default='all', help="feature group to process")
  parser.add_argument('feat_store', help='existing hydrat Store generated by features.py')
  parser.add_argument("output", help="produce output in PATH", metavar="PATH")
  args = parser.parse_args()

  class_space = 'ebmcat'

  try:
    features = features.feature_sets[args.feats]
  except KeyError:
    parser.error("unknown feature group: {0}".format(args.feats))

  l = repeat.RepeatLearner(liblinear.liblinearL(svm_type=0, output_probability=True))
  store = Store(args.feat_store, 'r') # TODO: Do we want this read-only?

  proxy = DataProxy(ALTA2012Full(), store=store)
  proxy.class_space = class_space
  
  L0_cl = []
  L1_fv = []
  L1_gs = None
  for feat in features: 
    proxy.feature_spaces = feat
    proxy.split_name = 'crossvalidation'

    with Timer() as L0_timer:
      L0_cl.append( l(proxy.featuremap.raw, proxy.classmap.raw) )
      print >>sys.stderr, "== training L0 for {0} took {1:.2f}s ==".format(feat, L0_timer.elapsed)

    with Timer() as L1_cv_timer:
      e = Experiment(proxy, l)
      if L1_gs is None:
        L1_gs = (e.overall_goldstandard().sum(axis=2) != 0)
      L1_fv.append( e.overall_classification().sum(axis=2) )
      print >>sys.stderr, "== training L1 feat for {0} took {1:.2f}s ==".format(feat, L1_cv_timer.elapsed)

  with Timer() as L1_timer:
    print >>sys.stderr, "== training L1 =="
    L1_fv = np.hstack(L1_fv)
    L1_cl = l(L1_fv, L1_gs)
    print >>sys.stderr, "== training L1 took {0:.2f}s ==".format(L1_timer.elapsed)


  with open(args.output, 'w') as out_f:
    dump((features, L0_cl, L1_cl), out_f)
    print >>sys.stderr, "== wrote model to {0} ==".format(args.output)