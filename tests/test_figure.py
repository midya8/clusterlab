import unittest,importlib.util,xml.etree.ElementTree as ET,base64,gzip,json
from pathlib import Path
from clusterlab.io import read
from clusterlab.experiment import ROOT
spec=importlib.util.spec_from_file_location('render_audit',Path(__file__).resolve().parents[1]/'examples/render_audit.py');renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
class FigureTests(unittest.TestCase):
 def test_exact_figure_and_all_values(self):
  text=renderer.render();self.assertEqual(text,(ROOT.parent/'assets/cluster-audit.svg').read_text());tree=ET.fromstring(text);r=read(ROOT/'artifacts/report.json');m=read(ROOT/'artifacts/models.json')
  pixels=[e for e in tree.iter() if 'data-pixel' in e.attrib];self.assertEqual(len(pixels),640)
  for e in pixels:
   j=int(e.attrib['data-cluster']);i=int(e.attrib['data-pixel']);self.assertEqual(float(e.attrib['data-value']),m['raw_plus']['pixel_centers'][j][i])
   v=round(max(0,min(1,float(e.attrib['data-value'])))*255);self.assertEqual(e.attrib['fill'],f'rgb({v},{v},{v})')
  cells=[e for e in tree.iter() if 'data-metric' in e.attrib];self.assertEqual(len(cells),16)
  for e in cells:self.assertEqual(float(e.attrib['data-value']),r['arms'][e.attrib['data-arm']][e.attrib['data-metric']])
  metadata=next(e for e in tree.iter() if e.attrib.get('id')=='report-gzip');packed=base64.b64decode(metadata.text);self.assertEqual(packed[9],255);self.assertEqual(json.loads(gzip.decompress(packed)),r)
 def test_deep_report_and_centroid_mutations_change_figure(self):
  r=read(ROOT/'artifacts/report.json');r['arms']['raw_plus']['labels'][-1]=99;self.assertNotEqual(renderer.render(r),renderer.render())
  m=read(ROOT/'artifacts/models.json');m['raw_plus']['pixel_centers'][0][0]=.99;self.assertNotEqual(renderer.render(models=m),renderer.render())
