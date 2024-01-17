
import numpy as np
import matplotlib.pyplot as plt
from numba import jit 
from matplotlib import animation 

# Longueur de la boîte et résolution 
Ny = 180
Nx = 500
dx = dy = 1

cs = 1/3 # vitesse du son adimensionnée (au carré)
Delta = 1/4 # magic parameteter

# # vitesse d'entrée imposée
# uinf = 0.04
# # Nombre de Reynolds
# Re = 10

# Pas de temps et temps de simulation
Nt = 15000

# taille de l'obstacle (diamètre)
D = Ny/4.5

# viscosité du fluide 
tau = 0.51

# Viscosité dynamique
#nu = uinf*D/Re

# taux de relaxation de l'opérateur de collision TRT

#wp = 1.0 / (3*nu+0.5)
wp = 1/tau
wm = 1.0 / ( 0.5 + Delta/(1/wp - 0.5) )
# Vecteurs vitesses (D2Q9)
c = np.zeros((9,2))

c[0] = np.array( [0,0] )
c[1] = np.array( [0,1] )
c[2] = np.array( [1,1] )
c[3] = np.array( [1,0] )
c[4] = np.array( [1,-1] )
c[5] = np.array( [0,-1] )
c[6] = np.array( [-1,-1] )
c[7] = np.array( [-1,0] )
c[8] = np.array( [-1,1] )

# poids D2Q9
w = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

# conditions initiales 
ux = np.zeros((Ny+2, Nx+2))
uy = np.zeros((Ny+2, Nx+2))
rho = np.zeros((Ny+2,Nx+2))

# distributions initiales
f = np.zeros((9, Ny+2, Nx+2))
for i in range(9) :
    f[i,:,:] = np.ones((Ny+2, Nx+2))*w[i]

#@jit(nopython=True)
def opp(i: int) :
    """ donne la direction opposée à i"""
    if i != 0 :
        return i+4 - ((i+4)//9)*8
    return 0

opposites = np.array([opp(i) for i in range(9)])

# fonction de distribution (equilibre)
#@jit(nopython=True)

# perturbation de vitesse
ux1 = np.ones((Ny+2,Nx+2))
for y in range(Ny+2) :
    ux1[y,:] = tau #*np.sin(y/(Ny+1) *2*np.pi)4

def feq(ux, uy, rho, c, boundaries) :
    f = np.zeros((9, Ny+2, Nx+2))
    # for x in range(Nx+2) :
    #     for y in range(Ny+2) :
                        
    #         Ux = np.zeros(ux.shape)
    #         Ux[y, x] = ux[y, x] + rho[y, x]*tau*1e-5
                        
    #         for i in range(9) :
    #              f[i,y, x] = rho[y, x]*w[i]*(1 + (c[i,0]*uy[y, x] + c[i,1]*Ux[y, x])/cs + - 1/(2*cs)*(Ux[y, x]**2 + uy[y, x]**2) 
    #  
    Ux = np.zeros(ux.shape)                                        # + 1/(2*cs**2) * ( c[i,0]*uy[y, x] + c[i,1]*Ux[y, x])**2)
    Ux = ux + ux1
    for i in range(9) :
        f[i,:,:] = w[i]*rho*(1 + (c[i,0]*uy + c[i,1]*Ux)/cs + - 1/(2*cs)*(Ux**2 + uy**2) 
                             + 1/(2*cs**2) * ( c[i,0]*uy + c[i,1]*Ux)**2 )
        
    return f

#@jit(nopython=True)
def cylindre(Ny, Nx, D, centre) :
    I = np.zeros((Ny+2, Nx+2))
    for i in range(Ny+2) :
        for j in range(Nx+2) :
            if (i-centre[0])**2 + (j-centre[1])**2 < D**2/4 :
                I[i,j] = 1
    return I

#@jit(nopython=True)
def no_obstacle(Ny, Nx) :
    I = np.zeros((Ny+2, Nx+2))
    I[:,0] = I[:, -1] = 1
    return I 
#@jit(nopython=True)
def TRT_collision(f, ux, uy, rho, c, boundaries) :
    """ calcule la collision après l'étape de propagagtion """
    
    fequilibre = feq(ux, uy, rho, c, boundaries)
    fstar = np.zeros(f.shape)
    #print(fequilibre[:,4,6], rho[4,6])
    # for x in range(Nx+2) :
    #     for y in range(Ny+2) :
    #             for i in range(9) : 
    #                 fp, fm = (f[i, y, x] + f[opp(i), y, x])/2, (f[i, y, x] - f[opp(i), y, x])/2
    #                 feqp = 0.5*(fequilibre[i, y, x] + fequilibre[opp(i), y, x])
    #                 feqm = 0.5*(fequilibre[i, y, x] - fequilibre[opp(i), y, x])
                    
    #                 # propage uniquement les noeuds fluides
    #                 fstar[i, y, x] = f[i, y, x] - wp*(fp-feqp) - wm*(fm - feqm)
    #for i in range(9) : 
    fp, fm = (f + f[opposites])/2, (f - f[opposites])/2
    feqp = 0.5*(fequilibre + fequilibre[opposites])
    feqm = 0.5*(fequilibre - fequilibre[opposites])
                    
    fstar = f - wp*(fp-feqp) - wm*(fm - feqm)
        
    return fstar

#@jit(nopython=True)    
def update_macro(f, ux, uy, rho, boundaries) :
    
    mask = (boundaries == 0)
    
    UX = np.zeros((Ny+2,Nx+2))
    UY = np.zeros((Ny+2,Nx+2))
    RHO = np.zeros((Ny+2,Nx+2)) 
    
    # for x in range(Nx+2) :
    #     for y in range(Ny+2) :
    #         RHO[y, x] = sum(f[:, y, x])
            
    #         UX[y, x] = sum([c[i,1]*f[i,y,x] for i in range(9)]) /RHO[y,x]
            
    #         UY[y, x] = sum( [c[i,0]*f[i,y,x] for i in range(9)] ) /RHO[y,x]
    RHO = np.sum(f, axis=0)
    UX = np.sum([c[i,1]*f[i,:] for i in range(9)], axis=0)/RHO
    UY = np.sum([c[i,0]*f[i,:] for i in range(9)], axis=0)/RHO

    #plt.spy(RHO)
    #UX[mask] = 1/rho[mask] * np.sum([c[i,1]*f[i][mask] for i in range(9)], axis=0)
    #UY[mask] = 1/rho[mask] * np.sum([c[i,0]*f[i][mask] for i in range(9)], axis=0) 
    
    return UX, UY, RHO

f = np.zeros((9, Ny+2, Nx+2))
for i in range(9) :
    f[i,:,:] = np.ones((Ny+2, Nx+2))*w[i]


# perturbation aléatoire sur l'une des directions 
f[1, : ,:] += np.abs(np.random.normal(loc=0, scale=1e-4))

# calcul des frontières du domaine
centre = [Ny//2, Nx//7]
#boundaries = cylindre(Ny, Nx, D, centre)
boundaries = cylindre(Ny, Nx, D, centre)
boundaries[0,:] = 1
boundaries[-1,:] = 1

ux, uy, rho = update_macro(f, ux, uy, rho, boundaries)
plt.imshow(ux)
plt.colorbar()
plt.show()

viscosite = cs*(tau - 0.5)
print(viscosite)

Re_liste = []
# main loop

plot_curl = True

for t in range(1, Nt) :
    
    # parois absorbanted pour réduire le bruit et les ondes
    f[4,:,-1] = f[4,:,-2]
    f[5,:,-1] = f[5,:,-2]
    f[6,:,-1] = f[6,:,-2]
    
    
    if t%10 == 0 :
        
        if plot_curl :
            # calcul vorticité
            dfydx= ux[2:, 1:-1] -  ux[0:-2, 1:-1]
            dfxdy = uy[1:-1, 2:] - uy[1:-1, 0:-2]
            curl = dfydx - dfxdy
            
            np.savetxt("lbm2_"+str(t)+"_curl.txt", curl)
            np.savetxt("lbm2_"+str(t)+"_usquared.txt", ux**2+uy**2)
            
            Re_liste.append(D*np.mean(ux)/viscosite)
            print(Re_liste[-1], t)
        else :
            Re_liste.append(D*np.mean(ux)/viscosite)
            print(Re_liste[-1], t)
            np.savetxt("lbm2_"+str(t)+"_usquared.txt", ux**2+uy**2)
            np.savetxt("lbm2_"+str(t)+"_rho.txt", rho)
    
    # mise à jour des variables macroscopiques 
    ux, uy, rho = update_macro(f, ux, uy, rho, boundaries)

    # collision
    fstar = TRT_collision(f, ux, uy, rho, c, boundaries)
    
    # streaming
    for i in range(9) :
        cy, cx = int(c[i,0]), int(c[i,1])
        for x in range(Nx+2) :
            for y in range(Ny+2) :

                if boundaries[y, x] == 0 and boundaries[(y-cy+Ny+2)%(Ny+2), (x-cx+Nx+2)%(Nx+2)] == 0 :
                    f[i, y, x] = fstar[i,(y-cy+Ny+2)%(Ny+2),(x-cx+Nx+2)%(Nx+2)]
                    
                elif boundaries[y, x] == 0 and boundaries[(y-cy+Ny+2)%(Ny+2), (x-cx+Nx+2)%(Nx+2)] == 1 :
                    f[i, y, x] = fstar[opposites[i], y, x]
    
                                      
ux, uy, rho = update_macro(f, ux, uy, rho, boundaries)
    ##print(rho[5,3])
plt.imshow(ux)
plt.show()
np.savetxt("Re_lbm.txt", np.array(Re_liste))

