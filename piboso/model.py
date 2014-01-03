"""
Trained model loading
"""

import bz2, os
import pkgutil
import logging
import tempfile
from cPickle import load, loads
from collections import namedtuple

from hydrat import config
from hydrat.store import Store

# Disable hydrat's progressbar output
import hydrat.common.pb as pb
pb.ENABLED = False

from piboso.common import Timer
from piboso.tokenize import induce

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

def load_model(path):
  with bz2.BZ2File(path) as model_f:
    return load(model_f)

def load_default_model():
  data = pkgutil.get_data('piboso', 'models/default')
  return loads(bz2.decompress(data))

PIBOSOModel = namedtuple("PIBOSOModel", ['features','spaces','L0_cl','L1_cl'])

class PIBOSOTagger(object):
  """
  Implements a sentence-level tagger for the PIBOSO tagset
  """
  def __init__(self, path=None, tempdir=None):
    self.path = path 
    self.tempdir = tempdir
    self.model = None

  def __unpack(self):
    if self.model is None:
      with Timer() as t:
        if self.path is None:
          logger.info("unpacking default model")
          model = PIBOSOModel(*load_default_model())
        else:
          logger.info("unpacking model from: {}".format(self.path))
          model = PIBOSOModel(*load_model(self.path))

      logger.info("unpacking took {0:.2f}s".format(t.elapsed))

      # hydrat hardcodes the paths for the classifier, which need to be updated
      # if they are installed at a different location
      classifier = config.getpath('tools','liblinearclassifier') 
      # TODO: Check that the tool exists
      if model.L1_cl.classifier != classifier:
        logger.debug("updating classifier path from {0} to {1}".format(model.L1_cl.classifier, classifier))
        model.L1_cl.classifier = classifier
        for c in model.L0_cl:
          c.classifier = classifier
      self.model = model
    else:
      logger.debug("already unpacked!")

  def classify_batch(self, abstracts):
    """
    Only batch-mode classify is supported due to the underlying operations.
    Single-item classification is expected to be slow.
    @param abstracts a mapping from docid to a list of lines
    """
    self.__unpack()
    ts = {}
    for filename, lines in abstracts.iteritems():
      for i, line in enumerate(lines):
        docid = "{0}-{1}".format(filename, i+1)
        ts[docid] = line

    try:
      handle, store_path = tempfile.mkstemp(dir=self.tempdir)
      os.close(handle)
      logger.debug("temporary store at {}".format(store_path))

      with Timer() as feat_timer:
        induce(ts, store_path, self.model.features, self.model.spaces)
        logger.info("computing features took {0:.2f}s".format(feat_timer.elapsed))

      store = Store(store_path, 'r')

      with Timer() as cl_timer:
        L0_preds = []
        for feat, cl in zip(self.model.features, self.model.L0_cl):
          fm = store.get_FeatureMap('NewDocuments', feat)
          # We need to trim the fv as the feature space may have grown when we tokenized more documents.
          # Hydrat's design is such that new features are appended to the end of a feature space, so
          # we can safely truncate the feature map.
          train_feat_count = cl.metadata['train_feat_count']
          assert(train_feat_count <= fm.raw.shape[1])
          L0_preds.append( cl(fm.raw[:,:train_feat_count]) )

        L0_preds = sp.csr_matrix(np.hstack(L0_preds))
        L1_preds = self.model.L1_cl(L0_preds)
          
        logger.info("classification took {0:.2f}s ({1:.2f} inst/s)".format(cl_timer.elapsed, cl_timer.rate(L0_preds.shape[0])))

      cl_space = store.get_Space('ebmcat')
      instance_ids = store.get_Space('NewDocuments')
    finally:
      logger.debug("unlinking {}".format(store_path))
      os.unlink(store_path)

    return PIBOSOOutput(instance_ids, cl_space, L1_preds)

class PIBOSOOutput(object):
  def __init__(self, instance_ids, cl_space, preds):
    self.instance_ids = instance_ids
    self.cl_space = cl_space
    self.preds = preds

  def write_pred(self, writer):
    writer.writerow( ["inst_id"] + list(self.cl_space) )
    for inst_id, cl_id in zip(self.instance_ids, self.preds):
      writer.writerow([inst_id] + list(cl_id))

  def write_dist(self, writer):
    for inst_id, cl_id in zip(self.instance_ids, self.preds.argmax(axis=1)):
      cl_name = self.cl_space[cl_id]
      writer.writerow((inst_id, cl_name))
