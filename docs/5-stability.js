'use strict';
(function(root){
 function verify(w){
  if(!Array.isArray(w.cost_units)||!w.cost_units.length||[w.scale,w.total_units,...w.row_dual,...w.column_dual,...w.cost_units.flat()].some(x=>typeof x!=='string'||!/^[-]?\d+$/.test(x)))throw Error('exact integer strings required');
  const c=w.cost_units.map(r=>r.map(BigInt)),u=w.row_dual.map(BigInt),v=w.column_dual.map(BigInt),a=w.assignment,n=c.length;
  if(a.length!==n||u.length!==n||v.length!==n||new Set(a).size!==n||a.some(x=>!Number.isInteger(x)||x<0||x>=n)||c.some(r=>r.length!==n)||BigInt(w.scale)<=0n)throw Error('invalid permutation, dimensions or scale');
  let total=0n,dual=0n;
  for(let i=0;i<n;i++){
   for(let j=0;j<n;j++)if(u[i]+v[j]>c[i][j])throw Error('dual infeasible');
   if(u[i]+v[a[i]]!==c[i][a[i]])throw Error('matched edge not tight');
   total+=c[i][a[i]];dual+=u[i]+v[i];
  }
  if(total!==BigInt(w.total_units)||dual!==total)throw Error('primal/dual mismatch');
  return true;
 }
 function pair(arm,left,right){
  const w=arm.pairs[Math.min(left,right)+'-'+Math.max(left,right)];
  if(left<=right)return w;
  const assignment=Array(w.assignment.length);w.assignment.forEach((j,i)=>assignment[j]=i);
  return {...w,left,right,assignment,cost_units:w.cost_units[0].map((_,j)=>w.cost_units.map(r=>r[j])),row_dual:w.column_dual,column_dual:w.row_dual};
 }
 const api={verify,pair};if(typeof module!=='undefined')module.exports=api;else root.ClusterStability=api;
 if(typeof document==='undefined')return;
 const D=CLUSTER_STABILITY,$=id=>document.getElementById(id);let pixels=[],custom=false;
 for(let i=0;i<10;i++){for(const id of ['left','right'])$(id).add(new Option('Seed '+(100+i),i));}
 D.examples.forEach(e=>$('row').add(new Option('Official row '+e.row,e.row)));
 for(let i=0;i<64;i++)$('pixel').add(new Option(i+' / row '+Math.floor(i/8)+', col '+i%8,i));
 function draw(canvas,p){const c=canvas.getContext('2d'),s=canvas.width/8;c.clearRect(0,0,canvas.width,canvas.height);p.forEach((v,i)=>{v=Math.round(Math.max(0,Math.min(1,v))*255);c.fillStyle=`rgb(${v},${v},${v})`;c.fillRect(i%8*s,Math.floor(i/8)*s,s,s);});}
 function model(a,i){return {...a.runs[i],representation:a.representation};}
 function matrix(a,l,r){
  const cv=$('matrix'),c=cv.getContext('2d'),s=48,start=70;c.clearRect(0,0,600,600);c.font='15px monospace';c.textAlign='center';c.fillStyle='#183d38';
  for(let i=0;i<10;i++){c.fillText(100+i,start+i*s+s/2,45);c.fillText(100+i,30,start+i*s+s/2+5);}
  for(let i=0;i<10;i++)for(let j=0;j<10;j++){const value=pair(a,i,j).ari;c.fillStyle=`rgb(${Math.round(238-185*value)},${Math.round(241-132*value)},${Math.round(233-138*value)})`;c.fillRect(start+j*s,start+i*s,s-1,s-1);if(i===l&&j===r){c.strokeStyle='#a4492e';c.lineWidth=4;c.strokeRect(start+j*s+2,start+i*s+2,s-5,s-5);}}
  $('range').textContent=`Across 45 distinct pairs: ARI min ${a.pairwise_ari_min.toFixed(6)}, mean ${a.pairwise_ari_mean.toFixed(6)}, max ${a.pairwise_ari_max.toFixed(6)}. Ten self-pairs equal 1 and are excluded from the summary.`;
  $('matrix-table').innerHTML='<caption>Fitting partition ARI; rows and columns are seeds</caption><thead><tr><th>Seed</th>'+a.runs.map(v=>'<th scope="col">'+v.seed+'</th>').join('')+'</tr></thead><tbody>'+a.runs.map((v,i)=>'<tr><th scope="row">'+v.seed+'</th>'+a.runs.map((_,j)=>'<td>'+pair(a,i,j).ari.toFixed(6)+'</td>').join('')+'</tr>').join('')+'</tbody>';
 }
 function update(){try{
  const arm=$('arm').value,a=D.arms[arm],l=Number($('left').value),r=Number($('right').value),w=pair(a,l,r);verify(w);
  const lr=a.runs[l],rr=a.runs[r],left=ClusterInference.infer(pixels,model(a,l)),right=ClusterInference.infer(pixels,model(a,r));
  $('selection').textContent=`Original fit-inertia winner: seed ${100+a.chosen}, unchanged. Viewing seeds ${lr.seed} / ${rr.seed}, fitting inertias ${lr.inertia.toFixed(4)} / ${rr.inertia.toFixed(4)}. Cluster IDs have no fixed meaning across runs.`;
  $('ari').textContent=w.ari.toFixed(6);$('movement').textContent=w.mean_squared_displacement.toFixed(6);$('certificate').textContent='EXACT CHECK PASSED';
  $('pairs').replaceChildren();w.assignment.forEach((j,i)=>{const card=document.createElement('article');card.innerHTML=`<h3>Left C${i} → Right C${j}</h3><div class="images"></div>`;for(const [run,index,side] of [[lr,i,'left'],[rr,j,'right']]){const cv=document.createElement('canvas');cv.width=cv.height=64;cv.setAttribute('aria-label',side+' centroid '+index);draw(cv,run.pixel_centers[index]);card.querySelector('.images').append(cv);}const p=document.createElement('p');p.textContent='Squared move '+(Number(w.cost_units[i][j])/Number(w.scale)).toFixed(5);card.append(p);$('pairs').append(card);});
  draw($('input'),pixels);const index=Number($('pixel').value);$('value').value=String(Math.round(pixels[index]*16));$('value-label').textContent=String(Math.round(pixels[index]*16));$('source').textContent=custom?'Custom pixels. No digit label.':'Original test row '+$('row').value+'. Digit labels omitted.';
  const gap=result=>{const x=result.squared_distances.slice().sort((a,b)=>a-b);return 'Nearest distance '+x[0].toFixed(6)+' / runner-up gap '+(x[1]-x[0]).toFixed(6);};
  $('left-result').textContent=`Left seed ${lr.seed}: C${left.cluster}`;$('right-result').textContent=`Right seed ${rr.seed}: C${right.cluster}`;$('left-gap').textContent=gap(left);$('right-gap').textContent=gap(right);
  $('aligned-result').textContent=`Left C${left.cluster} is matched to right C${w.assignment[left.cluster]}. The right model actually assigns this input to C${right.cluster}. ${w.assignment[left.cluster]===right.cluster?'These agree for this input.':'These disagree for this input.'}`;
  matrix(a,l,r);$('witness').textContent=JSON.stringify({schema:1,arm,models_sha256:D.models_sha256,left_seed:lr.seed,right_seed:rr.seed,chosen_seed:100+a.chosen,source_row:custom?null:Number($('row').value),pixels,left_inference:left,right_inference:right,matching:w,cost_semantics:D.cost_semantics},null,2);$('error').hidden=true;
 }catch(e){$('error').hidden=false;$('error').textContent=e.message;$('certificate').textContent='CHECK FAILED';}}
 function reset(){pixels=D.examples[Number($('row').value)].pixels.slice();custom=false;update();}
 function armChange(){const a=D.arms[$('arm').value];$('left').value=String(a.chosen);$('right').value=String((a.chosen+1)%10);update();}
 for(const id of ['left','right','pixel'])$(id).addEventListener('change',update);$('arm').addEventListener('change',armChange);$('row').addEventListener('change',reset);$('reset').addEventListener('click',reset);$('blank').addEventListener('click',()=>{pixels=Array(64).fill(0);custom=true;update();});$('value').addEventListener('input',()=>{pixels[Number($('pixel').value)]=Number($('value').value)/16;custom=true;update();});
 $('least').addEventListener('click',()=>{const a=D.arms[$('arm').value],w=Object.values(a.pairs).filter(w=>w.left!==w.right).reduce((x,y)=>x.ari<=y.ari?x:y);$('left').value=String(w.left);$('right').value=String(w.right);update();});
 $('copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('witness').textContent);$('copy-status').textContent='Copied exact JSON.';}catch{$('copy-status').textContent='Copy unavailable. Download JSON instead.';}});
 $('download').addEventListener('click',()=>{const u=URL.createObjectURL(new Blob([$('witness').textContent],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='clusterlab-stability.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);});pixels=D.examples[0].pixels.slice();armChange();
})(globalThis);
