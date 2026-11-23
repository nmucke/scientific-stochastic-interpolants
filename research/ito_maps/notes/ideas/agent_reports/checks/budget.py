import numpy as np, sys
d=np.load(sys.argv[1])["state"]
print(d.shape,d.dtype)
x=d[180:200].astype(np.float64)   # test trajectories [20,100,128,128] (or other layout)
print(x.shape)
n=x.shape[-1]
k=np.fft.fftfreq(n,1/n); kx,ky=np.meshgrid(k,k,indexing="ij"); k2=kx**2+ky**2
w=np.fft.fft2(x,norm="forward")
p=np.abs(w)**2
Z=0.5*p.sum((-1,-2)); P=0.5*(p*k2).sum((-1,-2))
nu,al=1e-3,0.1
print("mean Z",Z.mean(),"mean P",P.mean(),"k_rms",np.sqrt(P.mean()/Z.mean()))
print("2 nu P + 2 alpha Z =",2*nu*P.mean()+2*al*Z.mean()," (theory 2 eps^2 = 2 if eps=1)")
print("terms:",2*nu*P.mean(),2*al*Z.mean())
# mean vorticity (DC) and lag autocorrelation
print("DC rms",np.sqrt((x.mean((-1,-2))**2).mean()))
xc=x-x.mean((0,1),keepdims=True)
for lag in [1,2,5,10,20,40]:
    c=(xc[:,:-lag]*xc[:,lag:]).mean()/ (xc**2).mean()
    print("lag",lag,"autocorr",round(c,3))
# one-step increment rms in normalised units
s=3.09969
print("rms field/s",np.sqrt((x**2).mean())/s,"rms one-step increment/s",np.sqrt(((x[:,1:]-x[:,:-1])**2).mean())/s)
