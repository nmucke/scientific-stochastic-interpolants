import numpy as np
def run(sig,a,beta,dbeta,label):
    gam=lambda t:a*(1-t)
    def A(t):
        if t==0: 
            t=1e-9
        g=beta(t)*sig**2/(beta(t)**2*sig**2+gam(t)**2*t)
        return -1/(1-t)+(dbeta(t)+beta(t)/(1-t))*g
    V=lambda s,t:a*a*((1-s)**3-(1-t)**3)/3
    row=[]
    for N in (1,2,4,8,16):
        ts=np.linspace(0,1,N+1); v=0.0
        for s,t in zip(ts[:-1],ts[1:]): v=(1+(t-s)*A(s))**2*v+V(s,t)
        row.append(f"N={N}:{np.sqrt(v)/sig:5.2f}")
    print(f"{label:38s} "+" ".join(row))
sig=0.17
bq=(lambda t:t**2, lambda t:2*t); bm=(lambda t:t*np.sqrt(3-2*t), lambda t:(3-3*t)/np.sqrt(3-2*t))
print("endpoint std / sigma, Euler drift + exact noise, exact (Gaussian) drift, path-independent G")
for ratio in (0.1/0.17, 1.0, np.sqrt(3), 3.0):
    run(sig,ratio*sig,*bq,f"beta=t^2      a/sig={ratio:.2f}")
    run(sig,ratio*sig,*bm,f"beta=t*sqrt(3-2t) a/sig={ratio:.2f}")
