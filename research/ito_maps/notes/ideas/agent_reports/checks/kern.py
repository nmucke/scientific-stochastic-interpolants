import numpy as np
a=0.1
N=200000
r=(np.arange(N)+0.5)/N; dr=1/N
def kern(sig,quad):
    beta = r**2 if quad else r
    dbeta = 2*r if quad else np.ones_like(r)
    gam=a*(1-r)
    g=beta*sig**2/(beta**2*sig**2+gam**2*r)
    A=-1/(1-r)+(dbeta+beta/(1-r))*g
    cumA=np.cumsum(A)*dr
    k=gam*np.exp(cumA[-1]-cumA)   # k_1(r)
    return k,gam
def unexplained(k,phis):
    P=np.stack(phis,1)
    coef,*_=np.linalg.lstsq(P,k,rcond=None)
    res=k-P@coef
    return (res**2).sum()/(k**2).sum()
for quad in (False,True):
  for sig in (0.03,0.1,0.17,0.3,1.0):
    k,gam=kern(sig,quad)
    out=[f"beta={'t2' if quad else 't'} sig/a={sig/a:4.1f} var/sig2={(k**2).sum()*dr/sig**2:.3f}"]
    for K in (1,4,16):
        ind=[((r>=j/K)&(r<(j+1)/K)).astype(float) for j in range(K)]
        out.append(f"K{K}: M {100*unexplained(k,[gam*i for i in ind]):6.2f}%  W {100*unexplained(k,ind):6.2f}%")
    print(" | ".join(out))
    if quad and sig in (0.03,0.17,1.0):
        print("   k(r)/k(0) at r=0,.25,.5,.75,.9,.99:",np.round(k[[0,N//4,N//2,3*N//4,int(.9*N),int(.99*N)]]/k[0],3))
