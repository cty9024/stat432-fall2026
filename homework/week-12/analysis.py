"""Run: python3 analysis.py (or --help). Labels are accessed only after selection."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/golub-matplotlib')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse, base64, hashlib, html, io, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, pairwise_distances
from threadpoolctl import threadpool_limits

p=argparse.ArgumentParser()
p.add_argument('--data',default='data/golub.csv'); p.add_argument('--feature-prefix',default='gene_')
p.add_argument('--id',default='sample_id'); p.add_argument('--label',default='leukemia_type')
p.add_argument('--output',default='.'); p.add_argument('--skill',default='SKILL.md')
a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
inter=out/'intermediates'; inter.mkdir(exist_ok=True)
skill=Path(a.skill).read_bytes(); data_bytes=Path(a.data).read_bytes()
df=pd.read_csv(a.data)
features=[c for c in df if c.startswith(a.feature_prefix) and c not in [a.id,a.label]]
raw=df[features].apply(pd.to_numeric,errors='raise')
assert len(features)>0 and not np.isinf(raw.to_numpy()).any()
keep=raw.nunique(dropna=True)>1
X=StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(raw.loc[:,keep]))
pca=PCA(svd_solver='full').fit(X); scores=pca.transform(X)
spaces={'standardized':X}; dims={}
for v in [.8,.9,.95]:
    d=int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_),v)+1)
    spaces[f'PCA{int(v*100)}']=scores[:,:d]; dims[str(v)]=d

def model(c, seed=432):
    common={'n_clusters':c['k']}
    if c['algorithm']=='kmeans': return KMeans(**common,n_init=50,random_state=seed,max_iter=1000)
    if c['algorithm']=='hierarchical': return AgglomerativeClustering(**common,linkage=c['linkage'])
    kw=dict(**common,affinity=c['affinity'],random_state=seed,n_init=30,assign_labels='kmeans')
    kw.update({ 'gamma':c['gamma']} if c['affinity']=='rbf' else {'n_neighbors':c['neighbors']})
    return SpectralClustering(**kw)
rows=[]; assignments={}
with threadpool_limits(limits=1):
 for space,Z in spaces.items():
  dist=pairwise_distances(Z,squared=True); median=float(np.median(dist[np.triu_indices(len(Z),1)]))
  for k in range(2,min(6,len(X)-1)+1):
   configs=[dict(algorithm='kmeans',k=k)]+[dict(algorithm='hierarchical',k=k,linkage=l) for l in ['ward','complete','average']]
   configs += [dict(algorithm='spectral',k=k,affinity='rbf',gamma=g/median) for g in [.25,1,4]]
   configs += [dict(algorithm='spectral',k=k,affinity='nearest_neighbors',neighbors=n) for n in [5,10,15] if n<len(X)]
   for c in configs:
    c=dict(c,space=space); idx=len(rows)
    with warnings.catch_warnings(record=True) as ws:
     warnings.simplefilter('always')
     try:
      m=model(c); y=m.fit_predict(Z); sizes=np.bincount(y)
      valid=len(sizes)==k and sizes.min()>=2 and not any('converg' in str(w.message).lower() for w in ws)
      r=dict(c,id=idx,valid=bool(valid),min_size=int(sizes.min()),silhouette=float(silhouette_score(X,y)),CH=float(calinski_harabasz_score(X,y)),DB=float(davies_bouldin_score(X,y)),warnings=' | '.join(str(w.message) for w in ws))
      assignments[idx]=y
     except Exception as e: r=dict(c,id=idx,valid=False,error=str(e))
    rows.append(r)
 candidates=pd.DataFrame(rows); candidates.to_csv(inter/'all_candidates.csv',index=False)
 valid=candidates[candidates.valid].sort_values('silhouette',ascending=False)
 # Stability assessed for top candidate per algorithm/representation plus all within 0.01 of best.
 shortlist=set(valid.groupby(['algorithm','space']).head(1).id)|set(valid[valid.silhouette>=valid.silhouette.max()-.01].id)
 for idx in sorted(shortlist):
  c=rows[idx]; Z=spaces[c['space']]; y=assignments[idx]; seed_ari=[]; sub_ari=[]
  with warnings.catch_warnings():
   warnings.simplefilter('ignore')
   for seed in range(10): seed_ari.append(adjusted_rand_score(y,model(c,seed).fit_predict(Z)))
   for seed in range(30):
    ix=np.sort(np.random.default_rng(seed).choice(len(X),int(.8*len(X)),replace=False))
    # Fixed preprocessing isolates clustering sensitivity to omitted observations.
    ys=model(c,seed).fit_predict(Z[ix]); sub_ari.append(adjusted_rand_score(y[ix],ys))
  c.update(seed_ARI=float(np.mean(seed_ari)),subsample_ARI=float(np.mean(sub_ari)),subsample_ARI_sd=float(np.std(sub_ari)))
 stable=pd.DataFrame([rows[i] for i in shortlist])
 # Silhouette remains primary; stability/supporting indices break a 0.01 practical tie.
 near=stable[stable.silhouette>=valid.silhouette.max()-.01]
 chosen=near.sort_values(['subsample_ARI','seed_ARI','CH','DB'],ascending=[False,False,False,True]).iloc[0]
 idx=int(chosen.id); c=rows[idx]; y=assignments[idx]
 pd.DataFrame(rows).to_csv(inter/'all_candidates.csv',index=False)
 stable.to_csv(inter/'stability.csv',index=False)
 (inter/'selection.json').write_text(json.dumps(c,indent=2))
 # External labels first used here, after final selection is fixed.
 labels=df[a.label].astype(str); ari=adjusted_rand_score(labels,y)
 table=pd.crosstab(pd.Series(y+1,name='Cluster'),labels)
 pd.DataFrame({a.id:df[a.id],'cluster':y+1,a.label:labels}).to_csv(inter/'selected_assignments.csv',index=False)
 audit=dict(n=len(X),features=len(features),retained=int(keep.sum()),missing=int(raw.isna().sum().sum()),constant=int((~keep).sum()),sd_range=[float(raw.std().min()),float(raw.std().max())],max_abs_z=float(abs(X).max()),sample_norm_range=[float(np.linalg.norm(X,axis=1).min()),float(np.linalg.norm(X,axis=1).max())],PCA_dimensions=dims,ARI=float(ari),contingency=table.to_dict(),data_sha256=hashlib.sha256(data_bytes).hexdigest(),skill_sha256=hashlib.sha256(skill).hexdigest())
 (inter/'audit.json').write_text(json.dumps(audit,indent=2))
 print(json.dumps({'selected':c,'audit':audit},indent=2))
# Report: only the selected partition is displayed.
fig,ax=plt.subplots(figsize=(6.8,2.75),layout='constrained')
colors=['#2369a1','#d36b26','#328565','#9653a0','#b49c20','#666666']
for cluster in sorted(np.unique(y)):
 for label,marker in zip(sorted(labels.unique()),['o','^','s','D']):
  mask=(y==cluster)&(labels==label)
  if not mask.any(): continue
  ax.scatter(scores[mask,0],scores[mask,1],c=colors[cluster],marker=marker,s=40,edgecolors='white',linewidth=.5,label=f'Cluster {cluster+1} · {label}')
ax.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.15); ax.legend(fontsize=8,loc='upper right')
buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=180); plt.close(fig)
encoded=base64.b64encode(buf.getvalue()).decode()
explained=float(pca.explained_variance_ratio_[:2].sum())
name={'spectral':'Spectral clustering','kmeans':'K-means','hierarchical':'Hierarchical clustering'}[c['algorithm']]
params=(f"RBF affinity, γ = {c['gamma']:.7f}; 30 k-means assignment starts" if c['algorithm']=='spectral' and c['affinity']=='rbf' else str({k:v for k,v in c.items() if k in ['linkage','neighbors','k']}))
body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Golub gene-expression clustering</title>
<style>
*{{box-sizing:border-box}} body{{font:10.5pt/1.23 Arial,sans-serif;color:#18232c;max-width:7.4in;margin:24px auto}} h1{{font-size:21pt;margin:0 0 5px}} h2{{font-size:12pt;margin:11px 0 4px}} p{{margin:5px 0}} .subtitle{{color:#53616d;font-size:10pt}} img{{width:100%;height:2.65in;object-fit:contain}} table{{border-collapse:collapse;font-size:10pt;margin:5px 0;width:100%}} th,td{{padding:4px 9px;border-bottom:1px solid #cbd3da;text-align:center}} .metrics{{background:#edf3f7;padding:8px;font-size:10.5pt}} .appendix{{break-before:page;page-break-before:always}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:9pt/1.25 monospace}} @page{{size:letter;margin:.55in}} @media print{{body{{margin:0;max-width:none}} .report{{height:9.85in}} img{{height:2.2in}}}}
</style><main class="report"><h1>Do expression clusters recover leukemia types?</h1>
<p class="subtitle">Golub data · 38 biological samples · 3,051 genes · unsupervised analysis</p>
<h2>Design and selection</h2><p>All genes were retained: no missing values or constant features. Gene standard deviations ranged from {audit['sd_range'][0]:.2f} to {audit['sd_range'][1]:.2f}; each gene was standardized to mean 0 and variance 1. Extreme standardized values reached |z| = {audit['max_abs_z']:.2f}; samples were retained because unusual expression can be biological.</p>
<p>Compared standardized genes with PCA retaining 80%, 90%, and 95% variance (21, 28, and 32 components). Tuned k = 2–6 for K-means (50 starts), hierarchical clustering (Ward, complete, average), and spectral clustering (RBF and nearest-neighbor affinities). All partitions were scored using Euclidean distances in the same standardized gene space; singleton solutions were excluded.</p>
<p>Silhouette was primary. Within 0.01 of the best score, stability, then Calinski–Harabasz (CH) and Davies–Bouldin (DB), broke ties. Stability used 10 seeds and 30 random 80% subsamples with preprocessing fixed. Leukemia labels were withheld until selection.</p>
<h2>Selected model: {name}, k = {c['k']}</h2>
<p>PCA retaining 95% variance (32 components); {params}; seed 432. This solution had the highest silhouette. Other candidates were more stable but had weaker separation; absolute stability remained limited.</p>
<p class="metrics">Silhouette <b>{c['silhouette']:.3f}</b> · CH <b>{c['CH']:.3f}</b> · DB <b>{c['DB']:.3f}</b><br>Mean ARI across seeds <b>{c['seed_ARI']:.2f}</b> · subsample ARI <b>{c['subsample_ARI']:.2f}</b> (SD {c['subsample_ARI_sd']:.2f})</p>
<img alt="PCA projection of the selected clusters; colors denote clusters and shapes denote leukemia labels." src="data:image/png;base64,{encoded}">
<p class="subtitle">Two-PC projection ({explained:.1%} total variance); colors show selected clusters, shapes show labels.</p>
<table><thead><tr><th>Selected cluster</th><th>ALL</th><th>AML</th><th>Total</th></tr></thead><tbody>'''
for cluster in table.index:
 body+=f'<tr><td>{cluster}</td><td>{table.loc[cluster,"ALL"]}</td><td>{table.loc[cluster,"AML"]}</td><td>{table.loc[cluster].sum()}</td></tr>'
body+=f'''</tbody></table><p><b>Interpretation.</b> The 32- and 6-sample groups show weak full-space separation but appear separated mainly along PC1 in projection. The smaller group contains only ALL samples, but most ALL samples cluster with every AML sample. External ARI = <b>{ari:.3f}</b>: the partition does not recover ALL versus AML; it is an exploratory expression grouping, not established biological subtypes.</p>
<p><b>Limits.</b> Only 38 samples support 3,051 genes. Scaling, PCA truncation, and spectral affinity choices affect distances and graph structure. Subsampling shows instability; fixed preprocessing may overstate robustness. Two PCs omit most variance and cannot establish full-space separation. Independent biological validation is needed.</p>
</main><section class="appendix"><h1>Appendix · complete skill definition</h1><pre>'''+html.escape(skill.decode())+'</pre></section></html>'
(out/'homework-12.html').write_text(body)
# Preserve the original byte sequence, including its initial newline.
if (out/'SKILL.md').resolve()!=Path(a.skill).resolve(): (out/'SKILL.md').write_bytes(skill)
assert Path(a.data).read_bytes()==data_bytes and Path(a.skill).read_bytes()==skill
assert html.unescape(body.split('<pre>',1)[1].split('</pre>',1)[0]).encode()==skill
print('Report saved; original dataset and skill hashes verified.')
