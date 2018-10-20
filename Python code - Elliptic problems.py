##Resolution de l'équation de Laplace
##

# On montre que le problème est équivalent à résoudre le system (**): M(S_n)R_n = F_n(S_n) ,  M(R_n)S_n = G_n(R_n) pour n=1, ... Puis de calculer la somme des s_n*r_n = u_n. 
# Pour resoudre (**) on adopte une méthode iterative: M(S_n^m)R_n^m+1 = F_n(S_n^m) ,  M(R_n^m+1)S_n^m+1 = G_n(R_n^m+1)

# Il faut définir avant M,D et F[][]


from __future__ import print_function, division

from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from numpy import exp,arange
import numpy as np
from scipy import misc 
import scipy.integrate as integrate
import scipy.special as special
from six.moves import range
import numpy as np
import matplotlib.pyplot as plt 

import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


## Definition des paramètres
I = 30
h = 1/I
n_s = 20 #n
S0_s = np.ones(I+1) #S_0
m_s = 20 #m


## Definition des Phi_i

def Phi(x,i):
    return np.maximum(0,1-np.abs(x-i*h)/h )
        

def PhiT(x,i,j): #Phi_i*Phi_j
    return Phi(x,i)*Phi(x,j)
    
## Calcul relatifs à f/F

def f1(x):
    return (16*np.pi**2)*np.cos(4*np.pi*x)
    #return 1
def f2(y):
    return 1
    #return 2*np.cos(4*np.pi*y)
    
    
def prod1(x,i):
    return f1(x)*Phi(x,i)
def prod2(x,i):
    return f2(x)*Phi(x,i)
    
    


def CalculF():
    F = np.zeros((2,I+1))
    for i in range(I+1):
            F[0][i] = integrate.quad(prod1, 0, 1, args=(i,))[0]
    for i in range(I+1):
            F[1][i] = integrate.quad(prod2, 0, 1, args=(i,))[0]
    return F
Ff = CalculF()


## Calcul de M et D


def CalculM():
    M = np.zeros((I+1,I+1))
    for i in range(I+1):
        for j in range(I+1):
            M[i][j] = integrate.quad(PhiT, 0, 1, args=(i,j))[0]
    return M

M = CalculM()


def CalculD():
    D = np.zeros((I+1,I+1))
    D[0][0] = 1/h
    D[0][1] = -1/h
    D[I][I]=1/h
    D[I][I-1]=-1/h
    for i in range(1,I):
        D[i][i]=2/h
        D[i][i-1]=-1/h
        D[i][i+1]=-1/h
    return D

D = CalculD()



## p=1

## Clacul des fonctions F,G,M

def Fn(V,S,R,n):  # Calcul de F_n. S/R: liste des S_k/R_k / F: matrice de f.  #S[k]=S_k

    sum1 = 0         # .dot est la multiplivation matricielle / .T transposée
                      # Notre n est = n-1 de l'enoncée, tous les indices sont décalés de 1
    sum2 = 0
    sum1 =((V.T).dot(Ff[1]))*(Ff[0])
        
    for k in range(n):
        sum2 = sum2 + (((V.T).dot(D)).dot(S[k]))*M.dot(R[k]) + (((V.T).dot(M)).dot(S[k]))*D.dot(R[k]) + ( ((V.T).dot(M)).dot(S[k]) )*M.dot(R[k])

    return sum1 - sum2   
    

    
def Gn(V,S,R,n): # Calcul de G_n
    sum1 = 0
    sum2 = 0
    sum1 = ((V.T).dot(Ff[0]))*(Ff[1])
    for k in range(n):
        sum2 = sum2 + (((V.T).dot(D)).dot(R[k]))*M.dot(S[k]) + (((V.T).dot(M)).dot(R[k]))*D.dot(S[k]) + (((V.T).dot(M)).dot(R[k]))*M.dot(S[k])

    return sum1 - sum2 
    
    
def Mn(V): #Calcul de M()
    return( ((V.T).dot(D).dot(V))*(M)  + ((V.T).dot(M).dot(V))*(D) + ((V.T).dot(M).dot(V))*(M) )

## Implémentation de la méthode iterative

    
def MethodeIterative (S0,m,n,S,R): #S and R give previous S_k, R_k #m nombre d'iterrations
    Sm = S0

    for k in range(m):
        
        Rm = np.linalg.solve(Mn(Sm), Fn(Sm,S,R,n))  #Rm = R^m+1, Sm=S^m
        
        Sm = np.linalg.solve(Mn(Rm), Gn(Rm,S,R,n)) #Rm=R^m+1, Sm = S^m+1

    return (Rm,Sm)
    
## Resolution du probleme: formulation d'Euler

def Euler(m,n,S0):
    S = []
    R = []
    for k in range(n):
        (r,s)=MethodeIterative(S0,m,k,S,R)

        S.append(s)
        R.append(r)
    return (S,R)

## Calcul de du vecteur solution à partir de S et R

def tenso(a,b): #Produit tensoriel
    t = np.zeros((I+1,I+1))
    for i in range(I+1):
        for j in range(I+1):
            t[i][j] = a[i]*b[j]
    return t
    
def U(S,R,n): #U dans la base tensorielle
    U = np.zeros((I+1, I+1))
    for k in range(n):
        U = np.add(U,tenso(S[k],R[k]))

    return U

## Calcul des solutions



(S_s,R_s) = Euler(m_s ,n_s ,S0_s)
U_s = U(S_s,R_s,n_s)


## Tacer les solutions

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xt = yt = np.arange(0.0, 1.0, h)
X, Y = np.meshgrid(xt, yt)
zs = np.array([U_s[x*I][y*I] for x,y in zip(np.ravel(X), np.ravel(Y))])
#zs = np.array([Phi(x,I)*Phi(y,I) for x,y in zip(np.ravel(X), np.ravel(Y))])

Z = zs.reshape(X.shape)

#ax.scatter(X, Y, Z)
ax.plot_surface(X,Y,Z)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')
#ax.set_zlim(-1,2) #Utile dans le cas d'un resultat constant
plt.show()

