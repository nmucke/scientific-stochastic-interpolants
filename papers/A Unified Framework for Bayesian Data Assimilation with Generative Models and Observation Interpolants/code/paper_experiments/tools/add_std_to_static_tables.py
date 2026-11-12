import re, sys, pandas as pd
APPLY = '--apply' in sys.argv
R='/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results/'
MAN='/home/ntmucke/scientific-stochastic-interpolants/iclr_da_interpolant_manuscript/sections_review/'
ns = pd.concat([pd.read_csv(R+'navier_stokes/aggregated/all.csv'), pd.read_csv(R+'navier_stokes/ablation/aggregated.csv')])
ur = pd.read_csv(R+'urban/aggregated/all.csv')
NAME={'FlowDAS + SURGE':'SURGE (FlowDAS)','SDA + SURGE':'SURGE (SDA)','EnKF reference ($E=1000$)':'EnKF (reference)'}
NSSC=['16^2->128^2','32^2->128^2','sparse 5%','sparse 1.5625%']
STD='{\\scriptsize\\,$\\pm$%s}'
cellre=re.compile(r'^(\s*)(\\textbf\{|\\underline\{)?(-?\d+\.\d+)(\})?(\s*)$')
mism=[]; nadd={}; used=set()
def lookup(df,method,variant,scen,metric,Ms):
    q=df[(df.method==method)&(df.scenario==scen)&(df.metric==metric)&(df.M.isin(Ms))]
    q=q[q.variant.isna()] if variant is None else q[q.variant==variant]
    return q
def proc_cell(cell, df, method, variant, scen, metric, scale, Ms, tab, rowname):
    m=cellre.match(cell)
    if not m: mism.append((tab,rowname,scen,metric,'unparsed',cell)); return cell
    pre,op,num,cl,post=m.groups(); dec=len(num.split('.')[1])
    q=lookup(df,method,variant,scen,metric,Ms)
    hits=[r for _,r in q.iterrows() if f'{r.value*scale:.{dec}f}'==num]
    if not hits:
        mism.append((tab,rowname,variant,scen,metric,num,[(int(r.M),f'{r.value*scale:.{dec+2}f}') for _,r in q.iterrows()])); return cell
    Mset={int(r.M) for r in hits}
    r=hits[0]; used.add((tab,method,int(r.M)))
    if len(Mset)>1: print('AMBIG',tab,rowname,scen,metric,Mset)
    if pd.isna(r['std']) or r.n_traj<2:
        mism.append((tab,rowname,scen,metric,'nostd',num)); return cell
    nadd[tab]=nadd.get(tab,0)+1
    s=f'{r["std"]*scale:.{dec}f}'
    return f'{pre}{op or ""}{num}{cl or ""}{STD%s}{post}'
def ns_table(fname,label,metrics,Ms):
    L=open(MAN+fname).read().split('\n')
    i=next(k for k,l in enumerate(L) if '\\label{%s}'%label in l)
    variant=None
    while '\\bottomrule' not in L[i]:
        l=L[i]
        if '\\multicolumn' in l and 'textit' in l:
            variant='shared_jac1' if 'k=1$' in l else 'shared_jac5' if 'k=5$' in l else 'shared_jac10' if 'k=10$' in l else 'jacfree' if 'Jacobian-free' in l else '??'
        elif '\\midrule' in l and variant=='jacfree': variant=None
        elif l.strip().endswith('\\\\') and '&' in l and '$16^2$' not in l and 'multicolumn' not in l:
            body=l.rstrip()[:-2]; cells=body.split('&'); name=cells[0].strip()
            method=NAME.get(name,name); v=variant if method.startswith('Ours') else None
            for c in range(1,len(cells)):
                mi=(c-1)//4; sc=NSSC[(c-1)%4]
                if metrics[mi] is None: continue
                cells[c]=proc_cell(cells[c],ns,method,v,sc,metrics[mi],10,Ms,label,name)
            L[i]='&'.join(cells)+' \\\\'
        i+=1
    if APPLY: open(MAN+fname,'w').write('\n'.join(L))
def urban():
    fname='results.tex'; label='tab:urban_accuracy_M50'
    L=open(MAN+fname).read().split('\n')
    i=next(k for k,l in enumerate(L) if '\\label{%s}'%label in l and not l.lstrip().startswith('%'))
    variant=None; name=None; part=0
    while '\\bottomrule' not in L[i]:
        l=L[i]; s=l.strip()
        if 'Jacobian-free' in l: variant='jacfree'
        elif '\\multicolumn{9}' in l and 'textit' in l: variant='shared_jac10'
        elif s=='\\midrule' and variant=='jacfree': variant=None
        elif s.startswith('Ours') or s in ('FlowDAS + SURGE','SDA + SURGE','D-Flow SGLD','Guided FM (FIG)'): name=s; part=0
        elif s.startswith('&') and name and '$' not in s:
            end=' \\\\' if s.endswith('\\\\') else ''
            body=l.rstrip(); body=body[:-2].rstrip() if end else body
            cells=body.split('&'); method=NAME.get(name,name); v=variant if method.startswith('Ours') else None
            mets=['rmse_velocity','rmse_temperature'] if part==0 else ['crps_velocity','crps_temperature']
            for c in range(1,5):
                sc=['sparse 1.5625%','sparse 0.78125%'][(c-1)%2]
                cells[c]=proc_cell(cells[c],ur,method,v,sc,mets[(c-1)//2],1,[50],label,name)
            L[i]='&'.join(cells)+end; part+=1
        i+=1
    if APPLY: open(MAN+fname,'w').write('\n'.join(L))
ns_table('results.tex','tab:ns_accuracy',['rmse','crps',None],[100,50])
ns_table('appendix_ns_case.tex','tab:ablation',['rmse','crps','spread_skill',None],[100,50])
urban()
print('added',nadd); print('M used',used)
for m in mism: print('MISMATCH',m)
