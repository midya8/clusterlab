'use strict';
(function(root){
 const core=()=>typeof module!=='undefined'?require('./stability.js'):root.ClusterStability;
 function inverse(a){const b=Array(a.length);a.forEach((j,i)=>b[j]=i);return b;}
 function oriented(a,reversed){if(!reversed)return a;return {...a,exclusions:a.exclusions.map(e=>({...e,forbidden_row:e.forbidden_column,forbidden_column:e.forbidden_row,assignment:inverse(e.assignment),row_dual:e.column_dual,column_dual:e.row_dual}))};}
 function verify(w,a){
  core().verify(w);const n=w.assignment.length,c=w.cost_units.map(r=>r.map(BigInt));
  if(!Array.isArray(a.exclusions)||a.exclusions.length!==(n===1?0:n))throw Error('exclusion coverage');
  // Recount all-tight perfect matchings. A lone unused zero-cost edge need not
  // belong to an alternate perfect matching.
  const u=w.row_dual.map(BigInt),v=w.column_dual.map(BigInt);let dp=new Map([[0,1n]]);
  for(let i=0;i<n;i++){const next=new Map();for(const [mask,count] of dp)for(let j=0;j<n;j++)if(!(mask&(1<<j))&&c[i][j]===u[i]+v[j]){const key=mask|1<<j;next.set(key,(next.get(key)||0n)+count);}dp=next;}
  const count=dp.get((1<<n)-1)||0n;
  if(typeof a.optimal_count!=='string'||count!==BigInt(a.optimal_count)||a.unique!==(count===1n))throw Error('optimal-count mismatch');
  if(n===1){if(a.best_distinct!==null||a.gap_units!==null||a.strict_sufficient_cost_radius!==null)throw Error('singleton has no replacement');return true;}
  let low=c[0][0],high=low;for(const row of c)for(const x of row){if(x<low)low=x;if(x>high)high=x;}
  const penalty=BigInt(n)*high-BigInt(n-1)*low+1n,covered=new Set();let best=null;
  a.exclusions.forEach((e,k)=>{
   const i=e.forbidden_row,j=e.forbidden_column;
   if(!Number.isInteger(i)||i<0||i>=n||w.assignment[i]!==j||covered.has(i)||typeof e.penalty_units!=='string'||BigInt(e.penalty_units)!==penalty)throw Error('bad exclusion');covered.add(i);
   const costs=w.cost_units.map(row=>row.slice());costs[i][j]=e.penalty_units;
   core().verify({...e,cost_units:costs,scale:w.scale});
   if(e.assignment[i]===j)throw Error('forbidden edge used');
   let originalTotal=0n;e.assignment.forEach((col,row)=>originalTotal+=c[row][col]);if(originalTotal!==BigInt(e.total_units))throw Error('wrong original cost');
   if(best===null||BigInt(e.total_units)<BigInt(a.exclusions[best].total_units))best=k;
  });
  const gap=BigInt(a.exclusions[best].total_units)-BigInt(w.total_units);
  if(!Number.isInteger(a.best_distinct)||a.best_distinct<0||a.best_distinct>=n||BigInt(a.exclusions[a.best_distinct].total_units)!==BigInt(a.exclusions[best].total_units)||typeof a.gap_units!=='string'||gap!==BigInt(a.gap_units)||gap<0n||(gap>0n)!==a.unique)throw Error('replacement gap mismatch');
  const rho=a.strict_sufficient_cost_radius;
  if(typeof rho.numerator!=='string'||typeof rho.denominator!=='string'||BigInt(rho.denominator)<=0n||BigInt(rho.numerator)*BigInt(2*n)!==gap*BigInt(rho.denominator))throw Error('radius mismatch');
  return true;
 }
 const api={verify,oriented};if(typeof module!=='undefined')module.exports=api;else root.ClusterAmbiguity=api;
})(globalThis);
