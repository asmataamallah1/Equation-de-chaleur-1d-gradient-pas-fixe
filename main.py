import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt


n=100
L=100
T0=30.0
Tl=50.0

def fill_A(n):

    A=np.zeros((n-1,n-1))
    A[0,0]= 2
    A[0,1]=-1
    A[1,0]=-1
    A[n-2,n-2]=2
    A[n-2,n-3]=-1
    A[n-3,n-2]=-1

    for i in range(1,n-2):
        A[i,i]=2
        A[i,i+1]=-1
        A[i,i-1]=-1

    return(A)


def fill_B(n,T0,Tl,L):
    B=np.zeros(n-1)
    h=L/n
    B[0]   =math.sin(h) + T0/h**2
    B[n-2] =math.sin(L-h) + Tl/h**2
    for i in range(1,n-2):
        B[i]=math.sin((i+1)*h)
    return(B)


def defSystem(n,T0,Tl,L):
    A=fill_A(n)
    B=fill_B(n,T0,Tl,L)
    X=np.ones(n-1)
    return(A,B,X)

pas=0.1
X0=np.random.rand(n-1)
tolerence=0.01
max_iteration=1001
def gradientPasFixe(A,B,X0,pas,tolerence,max_iteration):
    compteur=0
    X=X0
    while LA.norm(np.matmul(A,X0)-B)>tolerence:
        X=X-pas * (np.matmul(A,X0)-B)
        compteur +=1
        if compteur > max_iteration :
            print("l'algorithme a divergé")
            break
    return(compteur)

n= [10,100,100]
Iter=[]
for i in range(0,len(n)-1):
    h=L/n[i]
    A=fill_A(n[i])
    B=fill_B(n[i],T0,Tl,L)
    x=np.zeros(n[i])
    N_iters=n[i]
    m=gradientPasFixe(A,B,X0,pas,tolerence,max_iteration)
    Iter.append(m)



plt.plot(n,Iter)
