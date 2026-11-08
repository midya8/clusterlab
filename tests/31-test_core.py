import unittest,itertools,math
from fractions import Fraction
import numpy as np
from clusterlab.kmeans import distances,initialize,lloyd,fit
from clusterlab.preprocess import fit_transform,transform,inverse
from clusterlab.metrics import ari,nmi,silhouette,contingency

class KMeansTests(unittest.TestCase):
 def test_distances_scalar(self):
  x=np.random.default_rng(2).normal(size=(7,3));c=x[:3]
  expected=[[sum((float(a)-float(b))**2 for a,b in zip(row,center)) for center in c] for row in x]
  np.testing.assert_allclose(distances(x,c),expected,atol=1e-14)
 def test_lloyd_final_means_and_objective(self):
  x=np.random.default_rng(4).normal(size=(30,4));r=fit(x,3,(1,2,3))
  for run in r['runs']:
   centers=np.array(run['centers']);labels=np.array(run['labels'])
   self.assertTrue(all(a['inertia']>=b['inertia']-1e-12 for a,b in zip(run['history'],run['history'][1:])))
   np.testing.assert_array_equal(labels,distances(x,centers).argmin(axis=1))
   for j in range(3):np.testing.assert_allclose(centers[j],x[labels==j].mean(axis=0))
   self.assertAlmostEqual(run['inertia'],sum(math.dist(row,centers[j])**2 for row,j in zip(x,labels)))
  self.assertEqual(r['chosen'],min(range(3),key=lambda j:r['runs'][j]['inertia']))
 def test_empty_repair(self):
  x=np.array([[0.],[1.],[9.],[10.]])
  r=lloyd(x,[[0.],[0.],[0.]])
  self.assertGreater(len(r['repairs']),0);self.assertEqual(r['reason'],'stable_assignments')
  self.assertEqual(len(set(r['labels'])),3)
  self.assertTrue(all(a['inertia']>=b['inertia'] for a,b in zip(r['history'],r['history'][1:])))
 def test_tie_lowest_center(self):
  r=lloyd([[0.],[1.],[2.]],[[0.],[2.]],max_iter=1)
  self.assertEqual(np.argmin(distances([[1.]],[[0.],[2.]])),0)
  self.assertEqual(sum(r['history'][0]['counts']),3)
 def test_initialization_independent_d_squared_oracle(self):
  x=np.array([[0.],[1.],[4.],[8.],[12.]])
  rng=np.random.default_rng(9);chosen=[int(rng.integers(5))]
  while len(chosen)<4:
   weights=[min((float(x[i,0])-float(x[j,0]))**2 for j in chosen) for i in range(5)]
   threshold=float(rng.random())*sum(weights);cumulative=0
   for i,w in enumerate(weights):
    cumulative+=w
    if cumulative>threshold:chosen.append(i);break
  self.assertEqual(initialize(x,4,9)[1],chosen)
 def test_init_and_seed(self):
  x=np.arange(20.).reshape(10,2)
  for mode in ['plus','uniform']:
   a,ids=initialize(x,5,7,mode);b,j=initialize(x,5,7,mode)
   np.testing.assert_array_equal(a,b);self.assertEqual(ids,j);self.assertEqual(len(set(ids)),5)
  _,ids=initialize([[0.],[0.],[1.],[2.]],3,2)
  self.assertEqual(len(set(float(z[0]) for z in np.array([[0.],[0.],[1.],[2.]])[ids])),3)
 def test_distinct_and_input_reject(self):
  for x,k in [([[0.],[0.]],2),([[1.]],2),([[float('nan')]],1),([],1)]:
   with self.assertRaises(ValueError):initialize(x,k,1)
  for x,c in [([[1.]],[[1.,2.]]),([[float('inf')]],[[0.]]),([[1.]],[])]:
   with self.assertRaises(ValueError):lloyd(x,c)
  with self.assertRaises(ValueError):fit([[1.],[2.]],1,())
  with self.assertRaises(ValueError):fit([[1.],[2.]],1,(1.1,))
  with self.assertRaises(ValueError):distances([[1e308]],[[-1e308]])
 def test_iteration_budget_truth(self):
  r=lloyd([[0.],[1.],[4.],[5.],[9.],[10.]],[[0.],[1.]],max_iter=1)
  self.assertEqual(r['reason'],'iteration_limit');self.assertEqual(len(r['history']),2)
 def test_local_minimum_exact_counterexample(self):
  x=[0,1,4,5,9,10];objectives=[]
  # Exhaust every binary partition with point0 fixed in group0; no contiguity assumption.
  for tail in itertools.product([0,1],repeat=5):
   labels=(0,)+tail
   if 1 not in labels:continue
   total=Fraction(0)
   for j in [0,1]:
    row=[Fraction(v) for v,g in zip(x,labels) if g==j];mean=sum(row)/len(row)
    total+=sum((v-mean)**2 for v in row)
   objectives.append(total)
  self.assertEqual(min(objectives),Fraction(35,2))
  r=lloyd(np.array(x)[:,None],[[0.],[1.]])
  self.assertEqual(r['inertia'],26.5);self.assertGreater(r['inertia'],float(min(objectives)))

class GeometryTests(unittest.TestCase):
 def setUp(self):self.x=np.random.default_rng(21).normal(size=(30,5))*[1,2,3,4,5]
 def test_raw_identity(self):
  m=fit_transform(self.x,'raw',3);np.testing.assert_array_equal(transform(self.x,m),self.x)
 def test_pca_orthogonal_reconstruction(self):
  m=fit_transform(self.x,'pca',3);b=np.array(m['basis']);z=transform(self.x,m)
  np.testing.assert_allclose(b.T@b,np.eye(3),atol=1e-14)
  np.testing.assert_allclose(inverse(z,m),(self.x-self.x.mean(0))@b@b.T+self.x.mean(0),atol=1e-14)
  evals=np.linalg.eigvalsh(np.cov(self.x,rowvar=False))[::-1]
  np.testing.assert_allclose(np.var(z,axis=0,ddof=1),evals[:3],atol=1e-13)
 def test_whitening_changes_metric(self):
  m=fit_transform(self.x,'white',3);z=transform(self.x,m)
  np.testing.assert_allclose(np.cov(z,rowvar=False),np.eye(3),atol=1e-14)
  self.assertFalse(np.allclose(distances(z[:2],z[:2]),distances(self.x[:2],self.x[:2])))
 def test_pca_fit_only(self):
  m=fit_transform(self.x,'pca',3);before=repr(m);transform(self.x+100,m);self.assertEqual(before,repr(m))
  with self.assertRaises(ValueError):fit_transform(np.ones((4,3)),'white',2)
  with self.assertRaises(ValueError):transform([[1.,2.]],m)

class MetricTests(unittest.TestCase):
 def test_ari_independent_pairs_and_permutation_null(self):
  a=[0,0,0,1,1,2];b=[0,0,1,1,2,2];pairs=list(itertools.combinations(range(6),2))
  agreement=lambda v:sum((a[i]==a[j])==(v[i]==v[j]) for i,j in pairs)/len(pairs)
  expected=sum(agreement(v) for v in set(itertools.permutations(b)))/len(set(itertools.permutations(b)))
  self.assertAlmostEqual(ari(a,b),(agreement(b)-expected)/(1-expected))
  self.assertAlmostEqual(ari(a,b),ari(a,[8,8,4,4,7,7]))
 def test_nmi_entropy_reference(self):
  a=[0,0,0,1,1,2];b=[0,0,1,1,2,2];c=contingency(a,b)
  entropy=lambda probs:-sum(p*math.log(p) for p in probs if p)
  ha=entropy(c.sum(1)/6);hb=entropy(c.sum(0)/6);joint=entropy(c.flat/6 if False else np.array(c).flatten()/6)
  self.assertAlmostEqual(nmi(a,b),2*(ha+hb-joint)/(ha+hb))
  self.assertAlmostEqual(nmi(a,b),nmi(b,a));self.assertAlmostEqual(nmi(a,a),1)
 def test_metric_degenerate(self):
  self.assertEqual(ari([0],[8]),1);self.assertEqual(ari([0,0],[1,1]),1)
  self.assertEqual(nmi([0,0],[1,1]),1);self.assertEqual(nmi([0,0],[0,1]),0)
  self.assertAlmostEqual(ari([0,0,1,1],[0,1,0,1]),-.5)
  with self.assertRaises(ValueError):ari([],[])
  with self.assertRaises(ValueError):nmi([0],[0,1])
 def test_silhouette_independent_scalar(self):
  x=np.array([[0.,0.],[1.,0.],[3.,0.],[5.,0.],[7.,0.]]);labels=[0,0,1,1,2];ref=[]
  for i,p in enumerate(x):
   own=[j for j in range(5) if labels[j]==labels[i] and j!=i]
   if not own:ref.append(0);continue
   a=sum(math.dist(p,x[j]) for j in own)/len(own)
   b=min(sum(math.dist(p,x[j]) for j in range(5) if labels[j]==g)/labels.count(g) for g in set(labels)-{labels[i]})
   ref.append((b-a)/max(a,b))
  np.testing.assert_allclose(silhouette(x,labels,2),ref,atol=1e-14)
  np.testing.assert_allclose(silhouette(x,labels,5),ref,atol=1e-14)
 def test_silhouette_invalid_and_zero(self):
  for labels in [[0,0,0],[0,1,2]]:
   with self.assertRaises(ValueError):silhouette([[0.],[1.],[2.]],labels)
  np.testing.assert_array_equal(silhouette([[0.],[0.],[0.]],[0,0,1]),[0,0,0])

if __name__=='__main__':unittest.main()
