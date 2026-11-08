import unittest,tempfile,json,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
from clusterlab import data
from clusterlab.experiment import train,evaluate,predict,ROOT
from clusterlab.io import read,save
from clusterlab.kmeans import distances
from clusterlab.preprocess import transform
from clusterlab.metrics import ari,nmi

class ProtocolTests(unittest.TestCase):
 def test_data_integrity_and_no_duplicate_leakage(self):
  x,y=data.load('train');z,t=data.load('test');self.assertEqual(x.shape,(3823,64));self.assertEqual(z.shape,(1797,64))
  self.assertFalse(set(map(tuple,x))&set(map(tuple,z)))
  self.assertEqual(set(y),set(range(10)));self.assertEqual(set(t),set(range(10)))
 def test_full_refit_no_test_or_label_usage(self):
  loader=data.load
  class ForbiddenLabels:
   def __getattribute__(self,name):raise AssertionError('training touched labels')
  def only_fit(part):
   if part!='train':raise AssertionError('test accessed in training')
   x,_=loader(part);return x,ForbiddenLabels()
  with tempfile.TemporaryDirectory() as tmp,patch('clusterlab.data.load',side_effect=only_fit):
   m=train(tmp);gold=read(ROOT/'artifacts/models.json')
   self.assertEqual(set(m),set(gold))
   for arm in m:
    self.assertEqual(m[arm]['chosen'],gold[arm]['chosen'])
    np.testing.assert_allclose(m[arm]['centers'],gold[arm]['centers'],rtol=1e-8,atol=1e-10)
    for r,g in zip(m[arm]['runs'],gold[arm]['runs']):
     self.assertEqual(r['initial_rows'],g['initial_rows']);self.assertEqual(r['labels'],g['labels'])
     np.testing.assert_allclose([h['inertia'] for h in r['history']],[h['inertia'] for h in g['history']],rtol=1e-8)
 def test_full_heldout_reproduction(self):
  with tempfile.TemporaryDirectory() as tmp:
   for name in ['protocol','models','evaluation-plan']:save(Path(tmp)/(name+'.json'),read(ROOT/'artifacts'/(name+'.json')))
   r=evaluate(tmp);g=read(ROOT/'artifacts/report.json')
   self.assertEqual(r['rows'],1797)
   for arm in r['arms']:
    a,b=r['arms'][arm],g['arms'][arm]
    self.assertEqual(a['labels'],b['labels']);self.assertEqual(a['composition_cluster_by_digit'],b['composition_cluster_by_digit'])
    np.testing.assert_allclose(a['squared_distances'],b['squared_distances'],atol=1e-10)
    np.testing.assert_allclose(a['silhouettes'],b['silhouettes'],atol=1e-10)
    for key in ['ari','nmi','silhouette','mean_squared_distance']:self.assertAlmostEqual(a[key],b[key],places=10)
 def test_tamper_checked_before_loader(self):
  for artifact in ['models','protocol']:
   with tempfile.TemporaryDirectory() as tmp:
    for name in ['protocol','models','evaluation-plan']:save(Path(tmp)/(name+'.json'),read(ROOT/'artifacts'/(name+'.json')))
    p=Path(tmp)/(artifact+'.json');p.write_text(p.read_text()+' ')
    with patch('clusterlab.data.load',side_effect=AssertionError('test accessed')),self.assertRaisesRegex(ValueError,'frozen'):evaluate(tmp)
 def test_selected_restarts_are_fit_minima(self):
  m=read(ROOT/'artifacts/models.json');r=read(ROOT/'artifacts/report.json');x,_=data.load('train')
  for arm,model in m.items():
   self.assertEqual(model['chosen'],min(range(10),key=lambda i:model['runs'][i]['inertia']))
   self.assertEqual(len(model['runs']),10)
   for run in model['runs']:
    d=distances(transform(x,model['representation']),run['centers']);labels=np.argmin(d,1)
    self.assertEqual(labels.tolist(),run['labels']);self.assertAlmostEqual(d[np.arange(len(x)),labels].sum(),run['inertia'],places=7)
   self.assertEqual(sum(r['arms'][arm]['counts']),1797)
 def test_predict_all_examples_and_strict_shape(self):
  m=read(ROOT/'artifacts/models.json');r=read(ROOT/'artifacts/report.json')
  for arm,case in r['arms'].items():
   for e in case['examples']:
    p=predict(e['pixels'],m)[arm];self.assertEqual(p['cluster'],e['cluster']);np.testing.assert_allclose(p['squared_distances'],e['distances'],atol=1e-11)
  for bad in [[0]*63,[float('nan')]*64,[-1]*64,[2]*64]:
   with self.assertRaises(ValueError):predict(bad,m)
 def test_json_rejects_nonfinite_and_duplicate(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'bad.json'
   for text in ['{"x":NaN}','{"x":1,"x":2}']:
    p.write_text(text)
    with self.assertRaises(ValueError):read(p)
 def test_labels_permuted_does_not_change_agreement(self):
  r=read(ROOT/'artifacts/report.json');_,y=data.load('test');permutation=np.array([8,3,6,1,5,2,9,0,4,7])
  for a in r['arms'].values():
   self.assertAlmostEqual(ari(permutation[y],a['labels']),a['ari']);self.assertAlmostEqual(nmi(permutation[y],a['labels']),a['nmi'])

if __name__=='__main__':unittest.main()
