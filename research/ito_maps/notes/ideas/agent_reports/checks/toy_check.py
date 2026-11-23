import numpy as np
from scipy.integrate import quad
a=1.0
def al(t): return 1-t
def be(t): return t*t
def ga(t): return a*(1-t)
def v(t,sf): return be(t)**2*sf**2+ga(t)**2*t
def k(t,sf):
    return (be(t)*2*t*sf**2 + ga(t)*(-a)*t)/v(t,sf)
def e(s,sf,t=1.0): return np.exp(quad(lambda u:k(u,sf),s,t,limit=200)[0])
print("Gaussian target N(m,sf^2), quadratic beta, gamma=a(1-t), a=1")
for sf in [0.05,0.1,0.3]:
    w2=quad(lambda r:(ga(r)*e(r,sf))**2,0,1,limit=200)[0]
    print(f"sf={sf}: Var(X_1)={w2:.6f} (target {sf**2:.6f}); a^2/3={a*a/3:.3f}; explicit-noise/forecast var ratio={(a*a/3)/sf**2:.1f}")
    for s in [0.5,0.7,0.8,0.9,0.95]:
        es=e(s,sf); kap_sde=es*be(s); bayes=be(s)**2*sf**2/v(s,sf)
        post=quad(lambda r:(ga(r)*e(r,sf))**2,s,1,limit=200)[0]
        inv=es**2*v(s,sf)+post
        print(f"   s={s}: contraction kappa=e_s*beta_s={kap_sde:.3f}  bayes-shrink={bayes:.3f}  fresh-var-frac={post/sf**2:.3f}  invariance check Var={inv:.6f}  noise floor a(1-s)/s^1.5={a*(1-s)/s**1.5:.3f}")
# leftover of A_{s,t}=a int (t-r)dW given K local M increments
def left(s,t,K):
    tot=quad(lambda r:(t-r)**2,s,t)[0]; res=0
    for j in range(K):
        l=s+(t-s)*j/K; r_=s+(t-s)*(j+1)/K
        aa=quad(lambda r:(t-r)**2,l,r_)[0]; ab=quad(lambda r:(t-r)*(1-r),l,r_)[0]; bb=quad(lambda r:(1-r)**2,l,r_)[0]
        res+=aa-ab*ab/bb
    return res/tot
for (s,t) in [(0,1),(0,.5),(.5,1),(0,.125),(.25,.5)]:
    print((s,t),"unexplained frac of Var(A) by K=1,4,16 local M-increments:",[f"{left(s,t,K):.4f}" for K in (1,4,16)])
