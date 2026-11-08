"""Reproducible SVG from frozen learned means and actual holdout scores."""
from pathlib import Path
import sys,html,gzip,base64
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from clusterlab.io import read,dumps
from clusterlab.experiment import ROOT

def render(report=None,models=None):
 r=read(ROOT/'artifacts/report.json') if report is None else report;m=read(ROOT/'artifacts/models.json') if models is None else models
 packed=bytearray(gzip.compress(dumps(r).encode(),mtime=0));packed[9]=255
 out=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1040" viewBox="0 0 1200 1040" role="img" aria-labelledby="title desc">','<title id="title">ClusterLab: shapes first, labels later</title>','<desc id="desc">Ten learned raw-pixel k-means++ centers, all four held-out score rows and a local-minimum counterexample. Cluster IDs are not digit predictions.</desc>','<metadata id="report-gzip">'+base64.b64encode(packed).decode()+'</metadata>','<rect width="1200" height="1040" fill="#f7f4ed"/>']
 def text(x,y,s,size=18,color='#183d38',family='Arial'):
  out.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{color}">{html.escape(str(s))}</text>')
 text(38,36,'MIDYA / CLUSTERLAB / REAL-DATA UNSUPERVISED GEOMETRY',13,'#a4492e')
 text(38,88,'Shapes first. Labels later.',42,family='Georgia')
 text(38,123,'3,823 fitting digits. 1,797 untouched test digits. No labels used to fit or choose a restart.',18)
 text(38,155,'Ten learned raw-pixel means / k-means++ seed 107 / arbitrary cluster indices',16,'#596861')
 for j,p in enumerate(m['raw_plus']['pixel_centers']):
  col=j%5;row=j//5;x=38+col*230;y=180+row*216
  out.append(f'<rect x="{x}" y="{y}" width="212" height="204" fill="#fffdf8" stroke="#d4d8cc"/>')
  text(x+12,y+24,'CLUSTER '+str(j),13)
  for i,v in enumerate(p):
   color=round(max(0,min(1,v))*255)
   out.append(f'<rect data-cluster="{j}" data-pixel="{i}" data-value="{v:.17g}" x="{x+14+i%8*18}" y="{y+36+i//8*18}" width="18" height="18" fill="rgb({color},{color},{color})"/>')
  count=r['arms']['raw_plus']['counts'][j];text(x+14,y+198,str(count)+' held-out rows',12,'#596861')
 text(38,634,'Frozen official test audit / k = 10 / ten starts per arm',27,family='Georgia')
 headers=['Arm','Adjusted Rand','Arithmetic NMI','Silhouette*','Fit inertia*']
 xs=[38,420,610,805,990]
 for x,s in zip(xs,headers):text(x,673,s,15,'#a4492e')
 names={'raw_plus':'64 pixels / k-means++','raw_uniform':'64 pixels / uniform','pca16_plus':'16 PCA / k-means++','white16_plus':'16 whitened PCA / k-means++'}
 for i,arm in enumerate(names):
  a=r['arms'][arm];y=712+i*43
  text(38,y,names[arm],16)
  for x,key in zip(xs[1:],['ari','nmi','silhouette','fit_best_inertia']):
   v=a[key];display=f'{v:.4f}' if key!='fit_best_inertia' else f'{v:.2f}'
   out.append(f'<text data-arm="{arm}" data-metric="{key}" data-value="{v:.17g}" x="{x}" y="{y}" font-family="monospace" font-size="18" fill="#183d38">{display}</text>')
 text(38,884,'* PCA and whitening change the metric. Inertia/silhouette are not a cross-geometry ranking.',15,'#a4492e')
 text(38,912,'ARI/NMI use labels only after fitting. NMI is not chance-adjusted. Scores are not classifier accuracy.',15,'#596861')
 text(38,953,'Descent is not global optimality: [0, 1, 4, 5, 9, 10], k=2, starts [0, 1] -> 26.5.',17)
 text(38,980,'All 31 nonempty binary partitions give global optimum 17.5. No global certificate for real-data fits.',16)
 text(38,1015,'Alpaydin & Kaynak (1998), UCI Optical Digits, CC BY 4.0. Code MIT. Fixed-seed descriptive audit; no endorsement.',13,'#596861')
 out.append('</svg>');return '\n'.join(out)+'\n'

if __name__=='__main__':Path('assets/cluster-audit.svg').write_text(render())
