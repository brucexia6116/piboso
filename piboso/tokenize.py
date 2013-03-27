"""
Tokenizer for feature sets used in PIBOSO sentence tagging.

Marco Lui, March 2013
"""

import os, sys, argparse
import tarfile
from contextlib import closing
from itertools import islice, groupby

import hydrat.common.extractors as ext
from hydrat.store import Store 
from hydrat.proxy import DataProxy

import piboso.features as features
from piboso.common import Timer, makedir
from piboso.corpora import NewDocuments

import multiprocessing as mp


def tokenize(ds, features, store_path):
  """
  Compute feature values and save them in a hydrat store.

  @param ds dataset to read from
  @param features names of features to read
  @param store_path path of store to write to
  """
  class_space = 'ebmcat'

  #print >>sys.stderr,  "=== opening store at {0} ===".format(store_path)
  with closing(Store(store_path, 'a')) as store:

    #print >>sys.stderr,  "=== inducing features ({0}) ===".format(features)
    # Induce all the features for the new test data
    proxy = DataProxy(ds, store=store)
    proxy.inducer.process(proxy.dataset, 
      fms=features,
      sqs=['abstract',],
    )

# This is split in two as some of the features were not declared
# in the dataset layer, and so need to be induced using an external
# tokenize call.
def tokenize_extra(ds, store_path):
  """
  Additional feature extraction for features that are not provided by the dataset
  implementation.

  @param ds dataset to read from
  @param store_path path of store to write to
  """
  class_space = 'ebmcat'

  #print >>sys.stderr,  "=== tokenize_extra for {0} ===".format(store_path)
  with closing(Store(store_path, 'a', recursive_close=False)) as store:
    proxy = DataProxy(ds, store=store)

    proxy.tokenstream_name = 'treetaggerlemmapos'
    proxy.tokenize(ext.bigram)

    proxy.tokenstream_name = 'treetaggerpos'
    proxy.tokenize(ext.bigram)
    proxy.tokenize(ext.trigram)
    
def induce(chunk, store_path, features, spaces):
  """
  Induce features for a list of abstracts.

  @param chunk list of abstracts (as open files) to process
  """
  ts = {}
  for f in chunk:
    for i, line in enumerate(f.readlines()):
      docid = "{0}-{1}".format(f.name, i+1)
      ts[docid] = line
  ds = NewDocuments(ts)

  # Merge feature spaces into the store
  with closing(Store(store_path, 'a')) as store:
    for space in spaces:
      md = {'name':space, 'type':'feature'}
      store.add_Space(spaces[space], md)

  # We do the feature induction in a subprocess to avoid Python holding on to memory.
  for feature in features:
    tokenize(ds, [feature], store_path)
    #p = mp.Process(target=tokenize, args=(ds, [feature], store_path))
    #p.start()
    #p.join()
    #p.terminate()

  tokenize_extra(ds, store_path)
  #p = mp.Process(target=tokenize_extra, args=(ds, store_path))
  #p.start()
  #p.join()
  #p.terminate()

def process_tarfile(data_path, feat_name, output_dir, fallback_path, parts=None, num_files=None):
  """
  Process medline abstracts stored in a tarfile. A two-layer directory
  structure is assumed, with numbered folders each containing files, and
  each file containing a separate abstract, sentence-tokenized into a 
  line-per-sentence format.

  @param data_path path to tar archive
  @param feat_name name of feature collection to process
  @param output_dir directory to save extracted features
  @param fallback_path path to fallback store (containing feature spaces)
  @param parts only process the first N parts in the archive
  @param num_files only process the first N files per archive part
  """

  try:
    features = features.feature_sets[feat_name]
  except KeyError:
    raise ValueError("unknown feature group: {0}".format(feat_name))

  with tarfile.open(data_path) as archive, Timer() as t:
    # The records in the tarfile are ordered by part. We only take the file records, 
    # extract the part ID from the filename and use that as a chunk.
    archive_iter = groupby((r for r in archive if r.isfile()), lambda r: r.name.split('/')[1])

    chunks_processed = 0
    files_processed = 0
    for chunk_id, chunk_tarinfos in archive_iter:
      if parts and chunks_processed >= parts:
        break

      if num_files:
        chunk_tarinfos = islice(chunk_tarinfos, num_files)

      chunk = [archive.extractfile(i) for i in chunk_tarinfos]
      store_path = os.path.join(output_dir, '{0}.features.{1}.h5'.format(chunk_id, feat_name)) 
      
      print >>sys.stderr, "==== processing Part {0} ({1} files) ====".format(chunk_id, len(chunk))
      raise NotImplementedError("need to update this to pass spaces rather than fallback path")
      induce(chunk, store_path, fallback_path, features)

      files_processed += len(chunk)
      chunks_processed += 1

      print >>sys.stderr,  "**** processed {0} files in {1}s ({2} f/s) ****".format(files_processed, t.elapsed, t.rate(files_processed))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-n','--number',type=int,help='number of files to process')
  parser.add_argument('-p','--parts',type=int,help='number of parts to process')
  parser.add_argument('--feats', default='all', help="feature group to process")
  parser.add_argument("data", metavar="PATH", help="read data from PATH (tgz format, filenames as docid)")
  parser.add_argument('feat_store', help='existing hydrat Store to read feature spaces from')
  parser.add_argument("outdir", help="produce output in DIR", metavar="DIR")
  args = parser.parse_args()

  makedir(args.outdir)

  print >>sys.stderr,  "=== producing output in {0} ===".format(args.outdir)
  process_tarfile(args.data, args.outdir, args.feat_store, args.parts, args.number)
