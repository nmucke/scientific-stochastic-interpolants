import numpy as np
a=0.1
beta=lambda t:t**2; dbeta=lambda t:2*t
gam=lambda t:a*(1-t)
def g(t,sig): return beta(t)*sig**2/(beta(t)**2*sig**2+gam(t)**2*t) if t>0 else 0.0
def A(t,sig): return -1/(1-t)+(dbeta(t)+beta(t)/(1-t))*g(t,sig)
V=lambda s,t:a*a*((1-s)**3-(1-t)**3)/3
# scalar Gaussian target N(0,sig^2), c=0. variance recursions with EXACT drift
print("ratio of endpoint std to true sigma (exact drift, c=0, y~N(0,sig^2), beta=t^2, a=0.1)")
for sig in (0.05,0.17,0.5):
  row=[]
  for N in (1,2,4,8,16,64):
    ts=np.linspace(0,1,N+1); ve=vb=0.0
    for s,t in zip(ts[:-1],ts[1:]):
        ve=(1+(t-s)*A(s,sig))**2*ve+V(s,t)          # Euler drift + exact noise (G-form, frozen drift)
        lam=(1-t)/(1-s); c=beta(t)-beta(s)*lam
        vb=(lam+c*g(s,sig))**2*vb+gam(t)**2*(t-s)   # bridge form, frozen denoiser
    row.append(f"N={N}: G-form {np.sqrt(ve)/sig:5.2f} bridge {np.sqrt(vb)/sig:5.2f}")
  print(f"sig={sig}: "+" | ".join(row))
# pathwise check of identity X_t = lam X_s + c*Ybar + gam_t (W_t-W_s) using fine EM on nonlinear denoiser (bimodal target +-1)
rng=np.random.default_rng(0); n=400000; dt=1/n; t=np.arange(n)*dt
def yhat(x,tt,m=0.3):
    b=beta(tt); v=gam(tt)**2*tt
    return m*np.tanh(b*m*x/v) if v>0 else 0.0
dW=rng.normal(size=n)*np.sqrt(dt); X=np.zeros(n+1); Yh=np.zeros(n)
for i in range(n):
    tt=t[i]; Yh[i]=yhat(X[i],tt)
    b=-X[i]/(1-tt)+(dbeta(tt)+beta(tt)/(1-tt))*Yh[i]
    X[i+1]=X[i]+b*dt+gam(tt)*dW[i]
W=np.concatenate([[0],np.cumsum(dW)])
for s,tt in ((0.0,0.5),(0.25,0.75),(0.5,0.999)):
    i,j=int(s*n),int(tt*n); rho=lambda u:beta(u)/gam(u)
    w=(dbeta(t[i:j])*gam(t[i:j])+beta(t[i:j])*a)/gam(t[i:j])**2   # (beta/gamma)'
    Ybar=(w*Yh[i:j]).sum()/w.sum(); lam=(1-tt)/(1-s); c=beta(tt)-beta(s)*lam
    print(f"s={s},t={tt}: X_t={X[j]:.5f}  bridge-form={lam*X[i]+c*Ybar+gam(tt)*(W[j]-W[i]):.5f}  Ybar={Ybar:.4f}")
