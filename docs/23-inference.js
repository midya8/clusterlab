'use strict';
(function(root){
 function infer(pixels,model){
  if(!Array.isArray(pixels)||pixels.length!==64||pixels.some(x=>!Number.isFinite(x)||x<0||x>1))throw Error('Expected 64 finite pixels in [0,1]');
  const p=model.representation,b=p.basis,scale=p.scale,z=scale.map((s,j)=>{let v=0;for(let i=0;i<64;i++)v+=(pixels[i]-p.mean[i])*b[i][j];return v/s;});
  const d=model.centers.map(c=>c.reduce((sum,v,j)=>sum+(z[j]-v)**2,0));
  let winner=0;for(let j=1;j<d.length;j++)if(d[j]<d[winner])winner=j;
  return {cluster:winner,squared_distances:d};
 }
 const api={infer};if(typeof module!=='undefined')module.exports=api;else root.ClusterInference=api;
})(globalThis);
