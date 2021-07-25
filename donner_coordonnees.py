# -*- coding: utf-8 -*-
"""
Created on Tue May 29 08:58:42 2018

@author: kerbr
"""

from math import sin,cos,pi
import matplotlib.pyplot as plt
import numpy as np


def rotation_d_angle_teta_autour_de_U(Vect,U,teta):
    U = normer(U)
    V = rechercher_V_orthogonal_à_(U)
    V = normer(V)
    W = produit_vectoriel(U,V)
    P = [[U[0],V[0],W[0]],[U[1],V[1],W[1]],[U[2],V[2],W[2]]]
    Pinv = [U,V,W]
    M = [[1,0,0],[0,cos(teta),-sin(teta)],[0,sin(teta),cos(teta)]]
    R = produit_matriciel(P,produit_matriciel(M,Pinv))
    return produit_matrice_vecteur(R,Vect)

def normer(U):
    s = 0
    for i in U:
        s+=i**2
    s = s**(1/2)
    return [U[0]/s,U[1]/s,U[2]/s]


def rechercher_V_orthogonal_à_(U):
    if U == [1,0,0] or U == [-1,0,0] or U == [0,1,0] or U == [0,-1,0]:
        V = [0,0,1]
    elif U == [0,0,1] or U == [0,0,-1]:
        V = [1,0,0]
    else:
        if U[0] == 0:
            V = [0,-U[2],U[1]]
        else:
            a = -U[2]
            c = U[0]
            V = [a,0,c]
    return V

def produit_vectoriel(U,V):
    [xu,yu,zu] = U
    [xv,yv,zv] = V
    X = yu*zv-yv*zu
    Y = zu*xv-zv*xu
    Z = xu*yv-xv*yu
    return [X,Y,Z]

def produit_matriciel(A,B):
    [[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]] = A
    [[B11,B12,B13],[B21,B22,B23],[B31,B32,B33]] = B
    C11 = A11*B11+A12*B21+A13*B31
    C12 = A11*B12+A12*B22+A13*B32
    C13 = A11*B13+A12*B23+A13*B33
    
    C21 = A21*B11+A22*B21+A23*B31
    C22 = A21*B12+A22*B22+A23*B32
    C23 = A21*B13+A22*B23+A23*B33
    
    C31 = A31*B11+A32*B21+A33*B31
    C32 = A31*B12+A32*B22+A33*B32
    C33 = A31*B13+A32*B23+A33*B33

    return [[C11,C12,C13],[C21,C22,C23],[C31,C32,C33]]

def produit_matrice_vecteur(A,B):
    [[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]] = A
    [x,y,z] = B
    X = A11*x+A12*y+A13*z
    Y = A21*x+A22*y+A23*z
    Z = A31*x+A32*y+A33*z
    return [X,Y,Z]

def fonction_donner_la_valeur(E,longueurs):
    
    [AB,AC,AF,FG,GG1,GG2,GG3,GG4,BD,DE,EE1,EE2,EE3,EE4,CH,HI,II1,II2,II3,II4,AL,AJ,LM,JK,KK1,KK2,KK3,KK4,MM1,MM2,MM3,MM4] = longueurs
    [teta1,teta2,teta3,teta4,teta5,teta6,teta7,teta8,teta9,teta10,teta11,teta12,teta13,teta14,teta15,teta16,teta17,teta18,teta19,teta20,teta21,teta22,teta23,teta24,teta25,teta26,teta27,teta28,teta29,teta30,teta31,teta32,teta33,teta34,teta35,teta36,teta37,teta38,teta39,teta40,teta41,teta42] = E
    
    A = [5,5,5]
    B = donner_les_coordonnees_de_B(A,AB,teta1,teta2,teta3)
    C = donner_les_coordonnees_de_C(A,AC,teta1,teta2,teta3)
    F = donner_les_coordonnees_de_F(A,AF,teta1,teta2,teta3)
    G = donner_les_coordonnees_de_G(F,FG,teta1,teta2,teta3,teta4,teta5,teta6)
    G1 = donner_les_coordonnees_de_G1(G,GG1,teta1,teta2,teta3,teta4,teta5,teta6)
    G2 = donner_les_coordonnees_de_G2(G,GG2,teta1,teta2,teta3,teta4,teta5,teta6)
    G3 = donner_les_coordonnees_de_G3(G,GG3,teta1,teta2,teta3,teta4,teta5,teta6)
    G4 = donner_les_coordonnees_de_G4(G,GG4,teta1,teta2,teta3,teta4,teta5,teta6)
    D = donner_les_coordonnees_de_D(B,BD,teta1,teta2,teta3,teta7,teta8,teta9)
    E = donner_les_coordonnees_de_E(D,DE,teta1,teta2,teta3,teta7,teta8,teta9,teta10,teta11,teta12)
    H = donner_les_coordonnees_de_H(C,CH,teta1,teta2,teta3,teta13,teta14,teta15)
    I = donner_les_coordonnees_de_I(H,HI,teta1,teta2,teta3,teta13,teta14,teta15,teta16,teta17,teta18)
    E1 = donner_les_coordonnees_de_E1(E,EE1,teta1,teta2,teta3,teta7,teta8,teta9,teta10,teta11,teta12,teta19,teta20,teta21)
    E2 = donner_les_coordonnees_de_E2(E,EE2,teta1,teta2,teta3,teta7,teta8,teta9,teta10,teta11,teta12,teta19,teta20,teta21)
    E3 = donner_les_coordonnees_de_E3(E,EE3,teta1,teta2,teta3,teta7,teta8,teta9,teta10,teta11,teta12,teta19,teta20,teta21)
    E4 = donner_les_coordonnees_de_E4(E,EE4,teta1,teta2,teta3,teta7,teta8,teta9,teta10,teta11,teta12,teta19,teta20,teta21)
    I1 = donner_les_coordonnees_de_I1(I,II1,teta1,teta2,teta3,teta13,teta14,teta15,teta16,teta17,teta18,teta22,teta23,teta24)
    I2 = donner_les_coordonnees_de_I2(I,II2,teta1,teta2,teta3,teta13,teta14,teta15,teta16,teta17,teta18,teta22,teta23,teta24)
    I3 = donner_les_coordonnees_de_I3(I,II3,teta1,teta2,teta3,teta13,teta14,teta15,teta16,teta17,teta18,teta22,teta23,teta24)
    I4 = donner_les_coordonnees_de_I4(I,II4,teta1,teta2,teta3,teta13,teta14,teta15,teta16,teta17,teta18,teta22,teta23,teta24)
    L = donner_les_coordonnees_de_L(A,AL,teta25,teta26,teta27)
    J = donner_les_coordonnees_de_J(A,AJ,teta28,teta29,teta30)
    M = donner_les_coordonnees_de_M(L,LM,teta25,teta26,teta27,teta31,teta32,teta33)
    K = donner_les_coordonnees_de_K(J,JK,teta28,teta29,teta30,teta34,teta35,teta36)
    K1 = donner_les_coordonnees_de_K1(K,KK1,teta28,teta29,teta30,teta34,teta35,teta36,teta40,teta41,teta42)
    K2 = donner_les_coordonnees_de_K2(K,KK2,teta28,teta29,teta30,teta34,teta35,teta36,teta40,teta41,teta42)
    K3 = donner_les_coordonnees_de_K3(K,KK3,teta28,teta29,teta30,teta34,teta35,teta36,teta40,teta41,teta42)
    K4 = donner_les_coordonnees_de_K4(K,KK4,teta28,teta29,teta30,teta34,teta35,teta36,teta40,teta41,teta42)
    M1 = donner_les_coordonnees_de_M1(M,MM1,teta25,teta26,teta27,teta31,teta32,teta33,teta37,teta38,teta39)
    M2 = donner_les_coordonnees_de_M2(M,MM2,teta25,teta26,teta27,teta31,teta32,teta33,teta37,teta38,teta39)
    M3 = donner_les_coordonnees_de_M3(M,MM3,teta25,teta26,teta27,teta31,teta32,teta33,teta37,teta38,teta39)
    M4 = donner_les_coordonnees_de_M4(M,MM4,teta25,teta26,teta27,teta31,teta32,teta33,teta37,teta38,teta39)

    return [A,B,C,F,G,G1,G2,G3,G4,D,E,H,I,E1,E2,E3,E4,I1,I2,I3,I4,L,J,M,K,K1,K2,K3,K4,M1,M2,M3,M4]

def donner_les_coordonnees_de_B(A,AB,teta1,teta2,teta3):
    
    #x0 = [1,0,0]
    #y0 = [0,1,0]
    #z0 = [0,0,1]
    
    #x11 = x0
    y11 = rotation_d_angle_teta_autour_de_U([0,1,0],[1,0,0],teta1)
    z11 = rotation_d_angle_teta_autour_de_U([0,0,1],[1,0,0],teta1)
    
    #x12 = rotation_d_angle_teta_autour_de_U(x11,y11,teta2)
    #y12 = y11
    z12 = rotation_d_angle_teta_autour_de_U(z11,y11,teta2)
    
    #x13 = rotation_d_angle_teta_autour_de_U(x12,z12,teta3)
    #y13 = rotation_d_angle_teta_autour_de_U(y12,z12,teta3)
    #z13 = z12
    
    AB = rotation_d_angle_teta_autour_de_U(AB,[1,0,0],teta1)
    AB = rotation_d_angle_teta_autour_de_U(AB,y11,teta2)
    AB = rotation_d_angle_teta_autour_de_U(AB,z12,teta3)
    
    return [A[0]+AB[0],A[1]+AB[1],A[2]+AB[2]]

def donner_les_coordonnees_de_C(A,AC,teta1,teta2,teta3):
    
    #x0 = [1,0,0]
    #y0 = [0,1,0]
    #z0 = [0,0,1]
    
    #x11 = x0
    y11 = rotation_d_angle_teta_autour_de_U([0,1,0],[1,0,0],teta1)
    z11 = rotation_d_angle_teta_autour_de_U([0,0,1],[1,0,0],teta1)
    
    #x12 = rotation_d_angle_teta_autour_de_U(x11,y11,teta2)
    #y12 = y11
    z12 = rotation_d_angle_teta_autour_de_U(z11,y11,teta2)
    
    #x13 = rotation_d_angle_teta_autour_de_U(x12,z12,teta3)
    #y13 = rotation_d_angle_teta_autour_de_U(y12,z12,teta3)
    #z13 = z12
    
    AC = rotation_d_angle_teta_autour_de_U(AC,[1,0,0],teta1)
    AC = rotation_d_angle_teta_autour_de_U(AC,y11,teta2)
    AC = rotation_d_angle_teta_autour_de_U(AC,z12,teta3)
    
    return [A[0]+AC[0],A[1]+AC[1],A[2]+AC[2]]

def donner_les_coordonnees_de_F(A,AF,teta1,teta2,teta3):
    
    #x0 = [1,0,0]
    #y0 = [0,1,0]
    #z0 = [0,0,1]
    
    #x11 = x0
    y11 = rotation_d_angle_teta_autour_de_U([0,1,0],[1,0,0],teta1)
    z11 = rotation_d_angle_teta_autour_de_U([0,0,1],[1,0,0],teta1)
    
    #x12 = rotation_d_angle_teta_autour_de_U(x11,y11,teta2)
    #y12 = y11
    z12 = rotation_d_angle_teta_autour_de_U(z11,y11,teta2)
    
    #x13 = rotation_d_angle_teta_autour_de_U(x12,z12,teta3)
    #y13 = rotation_d_angle_teta_autour_de_U(y12,z12,teta3)
    
    AF = rotation_d_angle_teta_autour_de_U(AF,[1,0,0],teta1)
    AF = rotation_d_angle_teta_autour_de_U(AF,y11,teta2)
    AF = rotation_d_angle_teta_autour_de_U(AF,z12,teta3)
    
    return [A[0]+AF[0],A[1]+AF[1],A[2]+AF[2]]

def donner_les_coordonnees_de_G(F,FG,teta1,teta2,teta3,teta4,teta5,teta6):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta1)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta1)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta2)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta2)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta3)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta3)
    Z13 = Z12
    
    #X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta4)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta4)
    
    #X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta5)
    #Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta5)
    
    #X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta6)
    #Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta6)
    #Z23 = Z22
    
    FG = rotation_d_angle_teta_autour_de_U(FG,X0,teta1)
    FG = rotation_d_angle_teta_autour_de_U(FG,Y11,teta2)
    FG = rotation_d_angle_teta_autour_de_U(FG,Z12,teta3)
    
    FG = rotation_d_angle_teta_autour_de_U(FG,X13,teta4)
    FG = rotation_d_angle_teta_autour_de_U(FG,Y21,teta5)
    FG = rotation_d_angle_teta_autour_de_U(FG,Z22,teta6)
    
    return [F[0]+FG[0],F[1]+FG[1],F[2]+FG[2]]
    
def donner_les_coordonnees_de_G1(G,GG1,teta1,teta2,teta3,teta4,teta5,teta6):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta1)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta1)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta2)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta2)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta3)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta3)
    Z13 = Z12
    
    #X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta4)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta4)
    
    #X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta5)
    #Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta5)
    
    #X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta6)
    #Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta6)
    #Z23 = Z22
    
    GG1 = rotation_d_angle_teta_autour_de_U(GG1,X0,teta1)
    GG1 = rotation_d_angle_teta_autour_de_U(GG1,Y11,teta2)
    GG1 = rotation_d_angle_teta_autour_de_U(GG1,Z12,teta3)
    GG1 = rotation_d_angle_teta_autour_de_U(GG1,X13,teta4)
    GG1 = rotation_d_angle_teta_autour_de_U(GG1,Y21,teta5)
    GG1 = rotation_d_angle_teta_autour_de_U(GG1,Z22,teta6)
    
    return [G[0]+GG1[0],G[1]+GG1[1],G[2]+GG1[2]]

def donner_les_coordonnees_de_G2(G,GG2,teta1,teta2,teta3,teta4,teta5,teta6):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta1)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta1)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta2)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta2)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta3)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta3)
    Z13 = Z12
    
    #X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta4)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta4)
    
    #X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta5)
    #Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta5)
    
    #X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta6)
    #Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta6)
    #Z23 = Z22
    
    GG2 = rotation_d_angle_teta_autour_de_U(GG2,X0,teta1)
    GG2 = rotation_d_angle_teta_autour_de_U(GG2,Y11,teta2)
    GG2 = rotation_d_angle_teta_autour_de_U(GG2,Z12,teta3)
    GG2 = rotation_d_angle_teta_autour_de_U(GG2,X13,teta4)
    GG2 = rotation_d_angle_teta_autour_de_U(GG2,Y21,teta5)
    GG2 = rotation_d_angle_teta_autour_de_U(GG2,Z22,teta6)
    
    return [G[0]+GG2[0],G[1]+GG2[1],G[2]+GG2[2]]

def donner_les_coordonnees_de_G3(G,GG3,teta1,teta2,teta3,teta4,teta5,teta6):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta1)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta1)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta2)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta2)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta3)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta3)
    Z13 = Z12
    
    #X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta4)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta4)
    
    #X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta5)
    #Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta5)
    
    #X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta6)
    #Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta6)
    #Z23 = Z22
    
    GG3 = rotation_d_angle_teta_autour_de_U(GG3,X0,teta1)
    GG3 = rotation_d_angle_teta_autour_de_U(GG3,Y11,teta2)
    GG3 = rotation_d_angle_teta_autour_de_U(GG3,Z12,teta3)
    GG3 = rotation_d_angle_teta_autour_de_U(GG3,X13,teta4)
    GG3 = rotation_d_angle_teta_autour_de_U(GG3,Y21,teta5)
    GG3 = rotation_d_angle_teta_autour_de_U(GG3,Z22,teta6)
    
    return [G[0]+GG3[0],G[1]+GG3[1],G[2]+GG3[2]]

def donner_les_coordonnees_de_G4(G,GG4,teta1,teta2,teta3,teta4,teta5,teta6):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta1)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta1)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta2)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta2)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta3)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta3)
    Z13 = Z12
    
    #X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta4)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta4)
    
    #X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta5)
    #Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta5)
    
    #X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta6)
    #Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta6)
    #Z23 = Z22
    
    GG4 = rotation_d_angle_teta_autour_de_U(GG4,X0,teta1)
    GG4 = rotation_d_angle_teta_autour_de_U(GG4,Y11,teta2)
    GG4 = rotation_d_angle_teta_autour_de_U(GG4,Z12,teta3)
    GG4 = rotation_d_angle_teta_autour_de_U(GG4,X13,teta4)
    GG4 = rotation_d_angle_teta_autour_de_U(GG4,Y21,teta5)
    GG4 = rotation_d_angle_teta_autour_de_U(GG4,Z22,teta6)
    
    return [G[0]+GG4[0],G[1]+GG4[1],G[2]+GG4[2]]

def donner_les_coordonnees_de_D(B,BD,teta1,teta2,teta3,teta7,teta8,teta9):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta1)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta1)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta2)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta2)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta3)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta3)
    Z13 = Z12
    
    #X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta7)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta7)
    
    #X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta8)
    #Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta8)
    
    #X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta9)
    #Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta9)
    #Z23 = Z22
    
    BD = rotation_d_angle_teta_autour_de_U(BD,X0,teta1)
    BD = rotation_d_angle_teta_autour_de_U(BD,Y11,teta2)
    BD = rotation_d_angle_teta_autour_de_U(BD,Z12,teta3)
    BD = rotation_d_angle_teta_autour_de_U(BD,X13,teta7)
    BD = rotation_d_angle_teta_autour_de_U(BD,Y21,teta8)
    BD = rotation_d_angle_teta_autour_de_U(BD,Z22,teta9)
    
    return [B[0]+BD[0],B[1]+BD[1],B[2]+BD[2]]

def donner_les_coordonnees_de_E(D,DE,teta1,teta2,teta3,teta7,teta8,teta9,teta10,teta11,teta12):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta1)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta1)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta2)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta2)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta3)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta3)
    Z13 = Z12
    
    X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta7)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta7)
    
    X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta8)
    Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta8)
    
    X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta9)
    Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta9)
    Z23 = Z22
    
    #X31 = X23
    Y31 = rotation_d_angle_teta_autour_de_U(Y23,X23,teta10)
    Z31 = rotation_d_angle_teta_autour_de_U(Z23,X23,teta10)
    
    #X32 = rotation_d_angle_teta_autour_de_U(X31,Y31,teta11)
    #Y32 = Y31
    Z32 = rotation_d_angle_teta_autour_de_U(Z31,Y31,teta11)
    
    #X33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta12)
    #Y33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta12)
    #Z33 = Z32
    
    DE = rotation_d_angle_teta_autour_de_U(DE,X0,teta1)
    DE = rotation_d_angle_teta_autour_de_U(DE,Y11,teta2)
    DE = rotation_d_angle_teta_autour_de_U(DE,Z12,teta3)
    
    DE = rotation_d_angle_teta_autour_de_U(DE,X13,teta7)
    DE = rotation_d_angle_teta_autour_de_U(DE,Y21,teta8)
    DE = rotation_d_angle_teta_autour_de_U(DE,Z22,teta9)
    
    DE = rotation_d_angle_teta_autour_de_U(DE,X23,teta10)
    DE = rotation_d_angle_teta_autour_de_U(DE,Y31,teta11)
    DE = rotation_d_angle_teta_autour_de_U(DE,Z32,teta12)
    
    return [D[0]+DE[0],D[1]+DE[1],D[2]+DE[2]]

def donner_les_coordonnees_de_E1(E,EE1,teta1,teta2,teta3,teta7,teta8,teta9,teta10,teta11,teta12,teta19,teta20,teta21):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta1)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta1)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta2)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta2)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta3)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta3)
    Z13 = Z12
    
    X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta7)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta7)
    
    X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta8)
    Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta8)
    
    X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta9)
    Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta9)
    Z23 = Z22
    
    X31 = X23
    Y31 = rotation_d_angle_teta_autour_de_U(Y23,X23,teta10)
    Z31 = rotation_d_angle_teta_autour_de_U(Z23,X23,teta10)
    
    X32 = rotation_d_angle_teta_autour_de_U(X31,Y31,teta11)
    Y32 = Y31
    Z32 = rotation_d_angle_teta_autour_de_U(Z31,Y31,teta11)
    
    X33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta12)
    Y33 = rotation_d_angle_teta_autour_de_U(Y32,Z32,teta12)
    Z33 = Z32
    
    #X41 = X33
    Y41 = rotation_d_angle_teta_autour_de_U(Y33,X33,teta19)
    Z41 = rotation_d_angle_teta_autour_de_U(Z33,X33,teta19)
    
    #X42 = rotation_d_angle_teta_autour_de_U(X41,Y41,teta20)
    #Y42 = Y41
    Z42 = rotation_d_angle_teta_autour_de_U(Z41,Y41,teta20)
    
    #X43 = rotation_d_angle_teta_autour_de_U(X42,Z42,teta21)
    #Y43 = rotation_d_angle_teta_autour_de_U(Y42,Z42,teta21)
    #Z43 = Z42
    
    EE1 = rotation_d_angle_teta_autour_de_U(EE1,X0,teta1)
    EE1 = rotation_d_angle_teta_autour_de_U(EE1,Y11,teta2)
    EE1 = rotation_d_angle_teta_autour_de_U(EE1,Z12,teta3)
    
    EE1 = rotation_d_angle_teta_autour_de_U(EE1,X13,teta7)
    EE1 = rotation_d_angle_teta_autour_de_U(EE1,Y21,teta8)
    EE1 = rotation_d_angle_teta_autour_de_U(EE1,Z22,teta9)
    
    EE1 = rotation_d_angle_teta_autour_de_U(EE1,X23,teta10)
    EE1 = rotation_d_angle_teta_autour_de_U(EE1,Y31,teta11)
    EE1 = rotation_d_angle_teta_autour_de_U(EE1,Z32,teta12)
    
    EE1 = rotation_d_angle_teta_autour_de_U(EE1,X33,teta19)
    EE1 = rotation_d_angle_teta_autour_de_U(EE1,Y41,teta20)
    EE1 = rotation_d_angle_teta_autour_de_U(EE1,Z42,teta21)
    
    return [E[0]+EE1[0],E[1]+EE1[1],E[2]+EE1[2]]

def donner_les_coordonnees_de_E2(E,EE2,teta1,teta2,teta3,teta7,teta8,teta9,teta10,teta11,teta12,teta19,teta20,teta21):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta1)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta1)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta2)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta2)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta3)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta3)
    Z13 = Z12
    
    X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta7)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta7)
    
    X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta8)
    Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta8)
    
    X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta9)
    Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta9)
    Z23 = Z22
    
    X31 = X23
    Y31 = rotation_d_angle_teta_autour_de_U(Y23,X23,teta10)
    Z31 = rotation_d_angle_teta_autour_de_U(Z23,X23,teta10)
    
    X32 = rotation_d_angle_teta_autour_de_U(X31,Y31,teta11)
    Y32 = Y31
    Z32 = rotation_d_angle_teta_autour_de_U(Z31,Y31,teta11)
    
    X33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta12)
    Y33 = rotation_d_angle_teta_autour_de_U(Y32,Z32,teta12)
    Z33 = Z32
    
    #X41 = X33
    Y41 = rotation_d_angle_teta_autour_de_U(Y33,X33,teta19)
    Z41 = rotation_d_angle_teta_autour_de_U(Z33,X33,teta19)
    
    #X42 = rotation_d_angle_teta_autour_de_U(X41,Y41,teta20)
    #Y42 = Y41
    Z42 = rotation_d_angle_teta_autour_de_U(Z41,Y41,teta20)
    
    #X43 = rotation_d_angle_teta_autour_de_U(X42,Z42,teta21)
    #Y43 = rotation_d_angle_teta_autour_de_U(Y42,Z42,teta21)
    #Z43 = Z42
    
    EE2 = rotation_d_angle_teta_autour_de_U(EE2,X0,teta1)
    EE2 = rotation_d_angle_teta_autour_de_U(EE2,Y11,teta2)
    EE2 = rotation_d_angle_teta_autour_de_U(EE2,Z12,teta3)
    
    EE2 = rotation_d_angle_teta_autour_de_U(EE2,X13,teta7)
    EE2 = rotation_d_angle_teta_autour_de_U(EE2,Y21,teta8)
    EE2 = rotation_d_angle_teta_autour_de_U(EE2,Z22,teta9)
    
    EE2 = rotation_d_angle_teta_autour_de_U(EE2,X23,teta10)
    EE2 = rotation_d_angle_teta_autour_de_U(EE2,Y31,teta11)
    EE2 = rotation_d_angle_teta_autour_de_U(EE2,Z32,teta12)
    
    EE2 = rotation_d_angle_teta_autour_de_U(EE2,X33,teta19)
    EE2 = rotation_d_angle_teta_autour_de_U(EE2,Y41,teta20)
    EE2 = rotation_d_angle_teta_autour_de_U(EE2,Z42,teta21)
    
    return [E[0]+EE2[0],E[1]+EE2[1],E[2]+EE2[2]]

def donner_les_coordonnees_de_E3(E,EE3,teta1,teta2,teta3,teta7,teta8,teta9,teta10,teta11,teta12,teta19,teta20,teta21):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta1)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta1)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta2)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta2)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta3)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta3)
    Z13 = Z12
    
    X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta7)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta7)
    
    X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta8)
    Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta8)
    
    X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta9)
    Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta9)
    Z23 = Z22
    
    X31 = X23
    Y31 = rotation_d_angle_teta_autour_de_U(Y23,X23,teta10)
    Z31 = rotation_d_angle_teta_autour_de_U(Z23,X23,teta10)
    
    X32 = rotation_d_angle_teta_autour_de_U(X31,Y31,teta11)
    Y32 = Y31
    Z32 = rotation_d_angle_teta_autour_de_U(Z31,Y31,teta11)
    
    X33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta12)
    Y33 = rotation_d_angle_teta_autour_de_U(Y32,Z32,teta12)
    Z33 = Z32
    
    #X41 = X33
    Y41 = rotation_d_angle_teta_autour_de_U(Y33,X33,teta19)
    Z41 = rotation_d_angle_teta_autour_de_U(Z33,X33,teta19)
    
    #X42 = rotation_d_angle_teta_autour_de_U(X41,Y41,teta20)
    #Y42 = Y41
    Z42 = rotation_d_angle_teta_autour_de_U(Z41,Y41,teta20)
    
    #X43 = rotation_d_angle_teta_autour_de_U(X42,Z42,teta21)
    #Y43 = rotation_d_angle_teta_autour_de_U(Y42,Z42,teta21)
    #Z43 = Z42
    
    EE3 = rotation_d_angle_teta_autour_de_U(EE3,X0,teta1)
    EE3 = rotation_d_angle_teta_autour_de_U(EE3,Y11,teta2)
    EE3 = rotation_d_angle_teta_autour_de_U(EE3,Z12,teta3)
    
    EE3 = rotation_d_angle_teta_autour_de_U(EE3,X13,teta7)
    EE3 = rotation_d_angle_teta_autour_de_U(EE3,Y21,teta8)
    EE3 = rotation_d_angle_teta_autour_de_U(EE3,Z22,teta9)
    
    EE3 = rotation_d_angle_teta_autour_de_U(EE3,X23,teta10)
    EE3 = rotation_d_angle_teta_autour_de_U(EE3,Y31,teta11)
    EE3 = rotation_d_angle_teta_autour_de_U(EE3,Z32,teta12)
    
    EE3 = rotation_d_angle_teta_autour_de_U(EE3,X33,teta19)
    EE3 = rotation_d_angle_teta_autour_de_U(EE3,Y41,teta20)
    EE3 = rotation_d_angle_teta_autour_de_U(EE3,Z42,teta21)
    
    return [E[0]+EE3[0],E[1]+EE3[1],E[2]+EE3[2]]

def donner_les_coordonnees_de_E4(E,EE4,teta1,teta2,teta3,teta7,teta8,teta9,teta10,teta11,teta12,teta19,teta20,teta21):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta1)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta1)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta2)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta2)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta3)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta3)
    Z13 = Z12
    
    X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta7)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta7)
    
    X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta8)
    Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta8)
    
    X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta9)
    Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta9)
    Z23 = Z22
    
    X31 = X23
    Y31 = rotation_d_angle_teta_autour_de_U(Y23,X23,teta10)
    Z31 = rotation_d_angle_teta_autour_de_U(Z23,X23,teta10)
    
    X32 = rotation_d_angle_teta_autour_de_U(X31,Y31,teta11)
    Y32 = Y31
    Z32 = rotation_d_angle_teta_autour_de_U(Z31,Y31,teta11)
    
    X33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta12)
    Y33 = rotation_d_angle_teta_autour_de_U(Y32,Z32,teta12)
    Z33 = Z32
    
    #X41 = X33
    Y41 = rotation_d_angle_teta_autour_de_U(Y33,X33,teta19)
    Z41 = rotation_d_angle_teta_autour_de_U(Z33,X33,teta19)
    
    #X42 = rotation_d_angle_teta_autour_de_U(X41,Y41,teta20)
    #Y42 = Y41
    Z42 = rotation_d_angle_teta_autour_de_U(Z41,Y41,teta20)
    
    #X43 = rotation_d_angle_teta_autour_de_U(X42,Z42,teta21)
    #Y43 = rotation_d_angle_teta_autour_de_U(Y42,Z42,teta21)
    #Z43 = Z42
    
    EE4 = rotation_d_angle_teta_autour_de_U(EE4,X0,teta1)
    EE4 = rotation_d_angle_teta_autour_de_U(EE4,Y11,teta2)
    EE4 = rotation_d_angle_teta_autour_de_U(EE4,Z12,teta3)
    
    EE4 = rotation_d_angle_teta_autour_de_U(EE4,X13,teta7)
    EE4 = rotation_d_angle_teta_autour_de_U(EE4,Y21,teta8)
    EE4 = rotation_d_angle_teta_autour_de_U(EE4,Z22,teta9)
    
    EE4 = rotation_d_angle_teta_autour_de_U(EE4,X23,teta10)
    EE4 = rotation_d_angle_teta_autour_de_U(EE4,Y31,teta11)
    EE4 = rotation_d_angle_teta_autour_de_U(EE4,Z32,teta12)
    
    EE4 = rotation_d_angle_teta_autour_de_U(EE4,X33,teta19)
    EE4 = rotation_d_angle_teta_autour_de_U(EE4,Y41,teta20)
    EE4 = rotation_d_angle_teta_autour_de_U(EE4,Z42,teta21)
    
    return [E[0]+EE4[0],E[1]+EE4[1],E[2]+EE4[2]]

def donner_les_coordonnees_de_H(C,CH,teta1,teta2,teta3,teta7,teta8,teta9):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta1)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta1)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta2)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta2)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta3)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta3)
    Z13 = Z12
    
    #X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta7)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta7)
    
    #X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta8)
    #Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta8)
    
    #X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta9)
    #Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta9)
    #Z23 = Z22
    
    CH = rotation_d_angle_teta_autour_de_U(CH,X0,teta1)
    CH = rotation_d_angle_teta_autour_de_U(CH,Y11,teta2)
    CH = rotation_d_angle_teta_autour_de_U(CH,Z12,teta3)
    CH = rotation_d_angle_teta_autour_de_U(CH,X13,teta7)
    CH = rotation_d_angle_teta_autour_de_U(CH,Y21,teta8)
    CH = rotation_d_angle_teta_autour_de_U(CH,Z22,teta9)
    
    return [C[0]+CH[0],C[1]+CH[1],C[2]+CH[2]]




def donner_les_coordonnees_de_I(H,HI,teta1,teta2,teta3,teta7,teta8,teta9,teta10,teta11,teta12):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta1)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta1)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta2)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta2)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta3)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta3)
    Z13 = Z12
    
    X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta7)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta7)
    
    X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta8)
    Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta8)
    
    X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta9)
    Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta9)
    Z23 = Z22
    
    #X31 = X23
    Y31 = rotation_d_angle_teta_autour_de_U(Y23,X23,teta10)
    Z31 = rotation_d_angle_teta_autour_de_U(Z23,X23,teta10)
    
    #X32 = rotation_d_angle_teta_autour_de_U(X31,Y31,teta11)
    #Y32 = Y31
    Z32 = rotation_d_angle_teta_autour_de_U(Z31,Y31,teta11)
    
    #X33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta12)
    #Y33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta12)
    #Z33 = Z32
    
    HI = rotation_d_angle_teta_autour_de_U(HI,X0,teta1)
    HI = rotation_d_angle_teta_autour_de_U(HI,Y11,teta2)
    HI = rotation_d_angle_teta_autour_de_U(HI,Z12,teta3)
    
    HI = rotation_d_angle_teta_autour_de_U(HI,X13,teta7)
    HI = rotation_d_angle_teta_autour_de_U(HI,Y21,teta8)
    HI = rotation_d_angle_teta_autour_de_U(HI,Z22,teta9)
    
    HI = rotation_d_angle_teta_autour_de_U(HI,X23,teta10)
    HI = rotation_d_angle_teta_autour_de_U(HI,Y31,teta11)
    HI = rotation_d_angle_teta_autour_de_U(HI,Z32,teta12)
    
    return [H[0]+HI[0],H[1]+HI[1],H[2]+HI[2]]



def donner_les_coordonnees_de_I1(I,II1,teta1,teta2,teta3,teta13,teta14,teta15,teta16,teta17,teta18,teta22,teta23,teta24):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta1)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta1)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta2)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta2)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta3)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta3)
    Z13 = Z12
    
    X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta13)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta13)
    
    X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta14)
    Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta14)
    
    X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta15)
    Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta15)
    Z23 = Z22
    
    X31 = X23
    Y31 = rotation_d_angle_teta_autour_de_U(Y23,X23,teta16)
    Z31 = rotation_d_angle_teta_autour_de_U(Z23,X23,teta16)
    
    X32 = rotation_d_angle_teta_autour_de_U(X31,Y31,teta17)
    Y32 = Y31
    Z32 = rotation_d_angle_teta_autour_de_U(Z31,Y31,teta17)
    
    X33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta18)
    Y33 = rotation_d_angle_teta_autour_de_U(Y32,Z32,teta18)
    Z33 = Z32
    
    #X41 = X33
    Y41 = rotation_d_angle_teta_autour_de_U(Y33,X33,teta22)
    Z41 = rotation_d_angle_teta_autour_de_U(Z33,X33,teta22)
    
    #X42 = rotation_d_angle_teta_autour_de_U(X41,Y41,teta23)
    #Y42 = Y41
    Z42 = rotation_d_angle_teta_autour_de_U(Z41,Y41,teta23)
    
    #X43 = rotation_d_angle_teta_autour_de_U(X42,Z42,teta24)
    #Y43 = rotation_d_angle_teta_autour_de_U(Y42,Z42,teta24)
    #Z43 = Z42
    
    II1 = rotation_d_angle_teta_autour_de_U(II1,X0,teta1)
    II1 = rotation_d_angle_teta_autour_de_U(II1,Y11,teta2)
    II1 = rotation_d_angle_teta_autour_de_U(II1,Z12,teta3)
    
    II1 = rotation_d_angle_teta_autour_de_U(II1,X13,teta13)
    II1 = rotation_d_angle_teta_autour_de_U(II1,Y21,teta14)
    II1 = rotation_d_angle_teta_autour_de_U(II1,Z22,teta15)
    
    II1 = rotation_d_angle_teta_autour_de_U(II1,X23,teta16)
    II1 = rotation_d_angle_teta_autour_de_U(II1,Y31,teta17)
    II1 = rotation_d_angle_teta_autour_de_U(II1,Z32,teta18)
    
    II1 = rotation_d_angle_teta_autour_de_U(II1,X33,teta22)
    II1 = rotation_d_angle_teta_autour_de_U(II1,Y41,teta23)
    II1 = rotation_d_angle_teta_autour_de_U(II1,Z42,teta24)
    
    return [I[0]+II1[0],I[1]+II1[1],I[2]+II1[2]]

def donner_les_coordonnees_de_I2(I,II2,teta1,teta2,teta3,teta13,teta14,teta15,teta16,teta17,teta18,teta22,teta23,teta24):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta1)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta1)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta2)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta2)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta3)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta3)
    Z13 = Z12
    
    X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta13)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta13)
    
    X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta14)
    Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta14)
    
    X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta15)
    Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta15)
    Z23 = Z22
    
    X31 = X23
    Y31 = rotation_d_angle_teta_autour_de_U(Y23,X23,teta16)
    Z31 = rotation_d_angle_teta_autour_de_U(Z23,X23,teta16)
    
    X32 = rotation_d_angle_teta_autour_de_U(X31,Y31,teta17)
    Y32 = Y31
    Z32 = rotation_d_angle_teta_autour_de_U(Z31,Y31,teta17)
    
    X33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta18)
    Y33 = rotation_d_angle_teta_autour_de_U(Y32,Z32,teta18)
    Z33 = Z32
    
    #X41 = X33
    Y41 = rotation_d_angle_teta_autour_de_U(Y33,X33,teta22)
    Z41 = rotation_d_angle_teta_autour_de_U(Z33,X33,teta22)
    
    #X42 = rotation_d_angle_teta_autour_de_U(X41,Y41,teta23)
    #Y42 = Y41
    Z42 = rotation_d_angle_teta_autour_de_U(Z41,Y41,teta23)
    
    #X43 = rotation_d_angle_teta_autour_de_U(X42,Z42,teta24)
    #Y43 = rotation_d_angle_teta_autour_de_U(Y42,Z42,teta24)
    #Z43 = Z42
    
    II2 = rotation_d_angle_teta_autour_de_U(II2,X0,teta1)
    II2 = rotation_d_angle_teta_autour_de_U(II2,Y11,teta2)
    II2 = rotation_d_angle_teta_autour_de_U(II2,Z12,teta3)
    
    II2 = rotation_d_angle_teta_autour_de_U(II2,X13,teta13)
    II2 = rotation_d_angle_teta_autour_de_U(II2,Y21,teta14)
    II2 = rotation_d_angle_teta_autour_de_U(II2,Z22,teta15)
    
    II2 = rotation_d_angle_teta_autour_de_U(II2,X23,teta16)
    II2 = rotation_d_angle_teta_autour_de_U(II2,Y31,teta17)
    II2 = rotation_d_angle_teta_autour_de_U(II2,Z32,teta18)
    
    II2 = rotation_d_angle_teta_autour_de_U(II2,X33,teta22)
    II2 = rotation_d_angle_teta_autour_de_U(II2,Y41,teta23)
    II2 = rotation_d_angle_teta_autour_de_U(II2,Z42,teta24)
    
    return [I[0]+II2[0],I[1]+II2[1],I[2]+II2[2]]

def donner_les_coordonnees_de_I3(I,II3,teta1,teta2,teta3,teta13,teta14,teta15,teta16,teta17,teta18,teta22,teta23,teta24):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta1)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta1)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta2)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta2)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta3)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta3)
    Z13 = Z12
    
    X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta13)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta13)
    
    X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta14)
    Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta14)
    
    X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta15)
    Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta15)
    Z23 = Z22
    
    X31 = X23
    Y31 = rotation_d_angle_teta_autour_de_U(Y23,X23,teta16)
    Z31 = rotation_d_angle_teta_autour_de_U(Z23,X23,teta16)
    
    X32 = rotation_d_angle_teta_autour_de_U(X31,Y31,teta17)
    Y32 = Y31
    Z32 = rotation_d_angle_teta_autour_de_U(Z31,Y31,teta17)
    
    X33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta18)
    Y33 = rotation_d_angle_teta_autour_de_U(Y32,Z32,teta18)
    Z33 = Z32
    
    #X41 = X33
    Y41 = rotation_d_angle_teta_autour_de_U(Y33,X33,teta22)
    Z41 = rotation_d_angle_teta_autour_de_U(Z33,X33,teta22)
    
    #X42 = rotation_d_angle_teta_autour_de_U(X41,Y41,teta23)
    #Y42 = Y41
    Z42 = rotation_d_angle_teta_autour_de_U(Z41,Y41,teta23)
    
    #X43 = rotation_d_angle_teta_autour_de_U(X42,Z42,teta24)
    #Y43 = rotation_d_angle_teta_autour_de_U(Y42,Z42,teta24)
    #Z43 = Z42
    
    II3 = rotation_d_angle_teta_autour_de_U(II3,X0,teta1)
    II3 = rotation_d_angle_teta_autour_de_U(II3,Y11,teta2)
    II3 = rotation_d_angle_teta_autour_de_U(II3,Z12,teta3)
    
    II3 = rotation_d_angle_teta_autour_de_U(II3,X13,teta13)
    II3 = rotation_d_angle_teta_autour_de_U(II3,Y21,teta14)
    II3 = rotation_d_angle_teta_autour_de_U(II3,Z22,teta15)
    
    II3 = rotation_d_angle_teta_autour_de_U(II3,X23,teta16)
    II3 = rotation_d_angle_teta_autour_de_U(II3,Y31,teta17)
    II3 = rotation_d_angle_teta_autour_de_U(II3,Z32,teta18)
    
    II3 = rotation_d_angle_teta_autour_de_U(II3,X33,teta22)
    II3 = rotation_d_angle_teta_autour_de_U(II3,Y41,teta23)
    II3 = rotation_d_angle_teta_autour_de_U(II3,Z42,teta24)
    
    return [I[0]+II3[0],I[1]+II3[1],I[2]+II3[2]]

def donner_les_coordonnees_de_I4(I,II4,teta1,teta2,teta3,teta13,teta14,teta15,teta16,teta17,teta18,teta22,teta23,teta24):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta1)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta1)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta2)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta2)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta3)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta3)
    Z13 = Z12
    
    X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta13)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta13)
    
    X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta14)
    Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta14)
    
    X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta15)
    Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta15)
    Z23 = Z22
    
    X31 = X23
    Y31 = rotation_d_angle_teta_autour_de_U(Y23,X23,teta16)
    Z31 = rotation_d_angle_teta_autour_de_U(Z23,X23,teta16)
    
    X32 = rotation_d_angle_teta_autour_de_U(X31,Y31,teta17)
    Y32 = Y31
    Z32 = rotation_d_angle_teta_autour_de_U(Z31,Y31,teta17)
    
    X33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta18)
    Y33 = rotation_d_angle_teta_autour_de_U(Y32,Z32,teta18)
    Z33 = Z32
    
    #X41 = X33
    Y41 = rotation_d_angle_teta_autour_de_U(Y33,X33,teta22)
    Z41 = rotation_d_angle_teta_autour_de_U(Z33,X33,teta22)
    
    #X42 = rotation_d_angle_teta_autour_de_U(X41,Y41,teta23)
    #Y42 = Y41
    Z42 = rotation_d_angle_teta_autour_de_U(Z41,Y41,teta23)
    
    #X43 = rotation_d_angle_teta_autour_de_U(X42,Z42,teta24)
    #Y43 = rotation_d_angle_teta_autour_de_U(Y42,Z42,teta24)
    #Z43 = Z42
    
    II4 = rotation_d_angle_teta_autour_de_U(II4,X0,teta1)
    II4 = rotation_d_angle_teta_autour_de_U(II4,Y11,teta2)
    II4 = rotation_d_angle_teta_autour_de_U(II4,Z12,teta3)
    
    II4 = rotation_d_angle_teta_autour_de_U(II4,X13,teta13)
    II4 = rotation_d_angle_teta_autour_de_U(II4,Y21,teta14)
    II4 = rotation_d_angle_teta_autour_de_U(II4,Z22,teta15)
    
    II4 = rotation_d_angle_teta_autour_de_U(II4,X23,teta16)
    II4 = rotation_d_angle_teta_autour_de_U(II4,Y31,teta17)
    II4 = rotation_d_angle_teta_autour_de_U(II4,Z32,teta18)
    
    II4 = rotation_d_angle_teta_autour_de_U(II4,X33,teta22)
    II4 = rotation_d_angle_teta_autour_de_U(II4,Y41,teta23)
    II4 = rotation_d_angle_teta_autour_de_U(II4,Z42,teta24)
    
    return [I[0]+II4[0],I[1]+II4[1],I[2]+II4[2]]

def donner_les_coordonnees_de_L(A,AL,teta25,teta26,teta27):
    
    #x0 = [1,0,0]
    #y0 = [0,1,0]
    #z0 = [0,0,1]
    
    #x11 = x0
    y11 = rotation_d_angle_teta_autour_de_U([0,1,0],[1,0,0],teta25)
    z11 = rotation_d_angle_teta_autour_de_U([0,0,1],[1,0,0],teta25)
    
    #x12 = rotation_d_angle_teta_autour_de_U(x11,y11,teta26)
    #y12 = y11
    z12 = rotation_d_angle_teta_autour_de_U(z11,y11,teta26)
    
    #x13 = rotation_d_angle_teta_autour_de_U(x12,z12,teta27)
    #y13 = rotation_d_angle_teta_autour_de_U(y12,z12,teta27)
    #z13 = z12
    
    AL = rotation_d_angle_teta_autour_de_U(AL,[1,0,0],teta25)
    AL = rotation_d_angle_teta_autour_de_U(AL,y11,teta26)
    AL = rotation_d_angle_teta_autour_de_U(AL,z12,teta27)
    
    return [A[0]+AL[0],A[1]+AL[1],A[2]+AL[2]]

def donner_les_coordonnees_de_J(A,AJ,teta28,teta29,teta30):
    
    #x0 = [1,0,0]
    #y0 = [0,1,0]
    #z0 = [0,0,1]
    
    #x11 = x0
    y11 = rotation_d_angle_teta_autour_de_U([0,1,0],[1,0,0],teta28)
    z11 = rotation_d_angle_teta_autour_de_U([0,0,1],[1,0,0],teta28)
    
    #x12 = rotation_d_angle_teta_autour_de_U(x11,y11,teta29)
    #y12 = y11
    z12 = rotation_d_angle_teta_autour_de_U(z11,y11,teta29)
    
    #x13 = rotation_d_angle_teta_autour_de_U(x12,z12,teta30)
    #y13 = rotation_d_angle_teta_autour_de_U(y12,z12,teta30)
    #z13 = z12
    
    AJ = rotation_d_angle_teta_autour_de_U(AJ,[1,0,0],teta28)
    AJ = rotation_d_angle_teta_autour_de_U(AJ,y11,teta29)
    AJ = rotation_d_angle_teta_autour_de_U(AJ,z12,teta30)
    
    return [A[0]+AJ[0],A[1]+AJ[1],A[2]+AJ[2]]

def donner_les_coordonnees_de_M(L,LM,teta25,teta26,teta27,teta31,teta32,teta33):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta25)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta25)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta26)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta26)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta27)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta27)
    Z13 = Z12
    
    #X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta31)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta31)
    
    #X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta32)
    #Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta32)
    
    #X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta33)
    #Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta33)
    #Z23 = Z22
    
    LM = rotation_d_angle_teta_autour_de_U(LM,X0,teta25)
    LM = rotation_d_angle_teta_autour_de_U(LM,Y11,teta26)
    LM = rotation_d_angle_teta_autour_de_U(LM,Z12,teta27)
    LM = rotation_d_angle_teta_autour_de_U(LM,X13,teta31)
    LM = rotation_d_angle_teta_autour_de_U(LM,Y21,teta32)
    LM = rotation_d_angle_teta_autour_de_U(LM,Z22,teta33)
    
    return [L[0]+LM[0],L[1]+LM[1],L[2]+LM[2]]

def donner_les_coordonnees_de_K(J,JK,teta28,teta29,teta30,teta34,teta35,teta36):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta28)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta28)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta29)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta29)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta30)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta30)
    Z13 = Z12
    
    #X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta34)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta34)
    
    #X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta35)
    #Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta35)
    
    #X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta36)
    #Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta36)
    #Z23 = Z22
    
    JK = rotation_d_angle_teta_autour_de_U(JK,X0,teta28)
    JK = rotation_d_angle_teta_autour_de_U(JK,Y11,teta29)
    JK = rotation_d_angle_teta_autour_de_U(JK,Z12,teta30)
    JK = rotation_d_angle_teta_autour_de_U(JK,X13,teta34)
    JK = rotation_d_angle_teta_autour_de_U(JK,Y21,teta35)
    JK = rotation_d_angle_teta_autour_de_U(JK,Z22,teta36)
    
    return [J[0]+JK[0],J[1]+JK[1],J[2]+JK[2]]

def donner_les_coordonnees_de_K1(K,KK1,teta28,teta29,teta30,teta34,teta35,teta36,teta40,teta41,teta42):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta28)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta28)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta29)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta29)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta30)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta30)
    Z13 = Z12
    
    X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta34)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta34)
    
    X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta35)
    Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta35)
    
    X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta36)
    Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta36)
    Z23 = Z22
    
    #X31 = X23
    Y31 = rotation_d_angle_teta_autour_de_U(Y23,X23,teta40)
    Z31 = rotation_d_angle_teta_autour_de_U(Z23,X23,teta40)
    
    #X32 = rotation_d_angle_teta_autour_de_U(X31,Y31,teta41)
    #Y32 = Y31
    Z32 = rotation_d_angle_teta_autour_de_U(Z31,Y31,teta41)
    
    #X33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta42)
    #Y33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta42)
    #Z33 = Z32
    
    KK1 = rotation_d_angle_teta_autour_de_U(KK1,X0,teta28)
    KK1 = rotation_d_angle_teta_autour_de_U(KK1,Y11,teta29)
    KK1 = rotation_d_angle_teta_autour_de_U(KK1,Z12,teta30)
    
    KK1 = rotation_d_angle_teta_autour_de_U(KK1,X13,teta34)
    KK1 = rotation_d_angle_teta_autour_de_U(KK1,Y21,teta35)
    KK1 = rotation_d_angle_teta_autour_de_U(KK1,Z22,teta36)

    KK1 = rotation_d_angle_teta_autour_de_U(KK1,X23,teta40)
    KK1 = rotation_d_angle_teta_autour_de_U(KK1,Y31,teta41)
    KK1 = rotation_d_angle_teta_autour_de_U(KK1,Z32,teta42)
    
    return [K[0]+KK1[0],K[1]+KK1[1],K[2]+KK1[2]]

def donner_les_coordonnees_de_K2(K,KK2,teta28,teta29,teta30,teta34,teta35,teta36,teta40,teta41,teta42):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta28)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta28)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta29)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta29)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta30)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta30)
    Z13 = Z12
    
    X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta34)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta34)
    
    X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta35)
    Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta35)
    
    X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta36)
    Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta36)
    Z23 = Z22
    
    #X31 = X23
    Y31 = rotation_d_angle_teta_autour_de_U(Y23,X23,teta40)
    Z31 = rotation_d_angle_teta_autour_de_U(Z23,X23,teta40)
    
    #X32 = rotation_d_angle_teta_autour_de_U(X31,Y31,teta41)
    #Y32 = Y31
    Z32 = rotation_d_angle_teta_autour_de_U(Z31,Y31,teta41)
    
    #X33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta42)
    #Y33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta42)
    #Z33 = Z32
    
    KK2 = rotation_d_angle_teta_autour_de_U(KK2,X0,teta28)
    KK2 = rotation_d_angle_teta_autour_de_U(KK2,Y11,teta29)
    KK2 = rotation_d_angle_teta_autour_de_U(KK2,Z12,teta30)
    
    KK2 = rotation_d_angle_teta_autour_de_U(KK2,X13,teta34)
    KK2 = rotation_d_angle_teta_autour_de_U(KK2,Y21,teta35)
    KK2 = rotation_d_angle_teta_autour_de_U(KK2,Z22,teta36)

    KK2 = rotation_d_angle_teta_autour_de_U(KK2,X23,teta40)
    KK2 = rotation_d_angle_teta_autour_de_U(KK2,Y31,teta41)
    KK2 = rotation_d_angle_teta_autour_de_U(KK2,Z32,teta42)
    
    return [K[0]+KK2[0],K[1]+KK2[1],K[2]+KK2[2]]

def donner_les_coordonnees_de_K3(K,KK3,teta28,teta29,teta30,teta34,teta35,teta36,teta40,teta41,teta42):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta28)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta28)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta29)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta29)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta30)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta30)
    Z13 = Z12
    
    X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta34)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta34)
    
    X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta35)
    Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta35)
    
    X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta36)
    Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta36)
    Z23 = Z22
    
    #X31 = X23
    Y31 = rotation_d_angle_teta_autour_de_U(Y23,X23,teta40)
    Z31 = rotation_d_angle_teta_autour_de_U(Z23,X23,teta40)
    
    #X32 = rotation_d_angle_teta_autour_de_U(X31,Y31,teta41)
    #Y32 = Y31
    Z32 = rotation_d_angle_teta_autour_de_U(Z31,Y31,teta41)
    
    #X33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta42)
    #Y33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta42)
    #Z33 = Z32
    
    KK3 = rotation_d_angle_teta_autour_de_U(KK3,X0,teta28)
    KK3 = rotation_d_angle_teta_autour_de_U(KK3,Y11,teta29)
    KK3 = rotation_d_angle_teta_autour_de_U(KK3,Z12,teta30)
    
    KK3 = rotation_d_angle_teta_autour_de_U(KK3,X13,teta34)
    KK3 = rotation_d_angle_teta_autour_de_U(KK3,Y21,teta35)
    KK3 = rotation_d_angle_teta_autour_de_U(KK3,Z22,teta36)

    KK3 = rotation_d_angle_teta_autour_de_U(KK3,X23,teta40)
    KK3 = rotation_d_angle_teta_autour_de_U(KK3,Y31,teta41)
    KK3 = rotation_d_angle_teta_autour_de_U(KK3,Z32,teta42)
    
    return [K[0]+KK3[0],K[1]+KK3[1],K[2]+KK3[2]]

def donner_les_coordonnees_de_K4(K,KK4,teta28,teta29,teta30,teta34,teta35,teta36,teta40,teta41,teta42):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta28)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta28)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta29)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta29)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta30)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta30)
    Z13 = Z12
    
    X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta34)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta34)
    
    X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta35)
    Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta35)
    
    X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta36)
    Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta36)
    Z23 = Z22
    
    #X31 = X23
    Y31 = rotation_d_angle_teta_autour_de_U(Y23,X23,teta40)
    Z31 = rotation_d_angle_teta_autour_de_U(Z23,X23,teta40)
    
    #X32 = rotation_d_angle_teta_autour_de_U(X31,Y31,teta41)
    #Y32 = Y31
    Z32 = rotation_d_angle_teta_autour_de_U(Z31,Y31,teta41)
    
    #X33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta42)
    #Y33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta42)
    #Z33 = Z32
    
    KK4 = rotation_d_angle_teta_autour_de_U(KK4,X0,teta28)
    KK4 = rotation_d_angle_teta_autour_de_U(KK4,Y11,teta29)
    KK4 = rotation_d_angle_teta_autour_de_U(KK4,Z12,teta30)
    
    KK4 = rotation_d_angle_teta_autour_de_U(KK4,X13,teta34)
    KK4 = rotation_d_angle_teta_autour_de_U(KK4,Y21,teta35)
    KK4 = rotation_d_angle_teta_autour_de_U(KK4,Z22,teta36)

    KK4 = rotation_d_angle_teta_autour_de_U(KK4,X23,teta40)
    KK4 = rotation_d_angle_teta_autour_de_U(KK4,Y31,teta41)
    KK4 = rotation_d_angle_teta_autour_de_U(KK4,Z32,teta42)
    
    return [K[0]+KK4[0],K[1]+KK4[1],K[2]+KK4[2]]

def donner_les_coordonnees_de_M1(M,MM1,teta25,teta26,teta27,teta31,teta32,teta33,teta37,teta38,teta39):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta25)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta25)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta26)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta26)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta27)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta27)
    Z13 = Z12
    
    X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta31)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta31)
    
    X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta32)
    Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta32)
    
    X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta33)
    Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta33)
    Z23 = Z22
    
    #X31 = X23
    Y31 = rotation_d_angle_teta_autour_de_U(Y23,X23,teta37)
    Z31 = rotation_d_angle_teta_autour_de_U(Z23,X23,teta37)
    
    #X32 = rotation_d_angle_teta_autour_de_U(X31,Y31,teta38)
    #Y32 = Y31
    Z32 = rotation_d_angle_teta_autour_de_U(Z31,Y31,teta38)
    
    #X33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta39)
    #Y33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta39)
    #Z33 = Z32
    
    MM1 = rotation_d_angle_teta_autour_de_U(MM1,X0,teta25)
    MM1 = rotation_d_angle_teta_autour_de_U(MM1,Y11,teta26)
    MM1 = rotation_d_angle_teta_autour_de_U(MM1,Z12,teta27)
    
    MM1 = rotation_d_angle_teta_autour_de_U(MM1,X13,teta31)
    MM1 = rotation_d_angle_teta_autour_de_U(MM1,Y21,teta32)
    MM1 = rotation_d_angle_teta_autour_de_U(MM1,Z22,teta33)

    MM1 = rotation_d_angle_teta_autour_de_U(MM1,X23,teta37)
    MM1 = rotation_d_angle_teta_autour_de_U(MM1,Y31,teta38)
    MM1 = rotation_d_angle_teta_autour_de_U(MM1,Z32,teta39)
    
    return [M[0]+MM1[0],M[1]+MM1[1],M[2]+MM1[2]]

def donner_les_coordonnees_de_M2(M,MM2,teta25,teta26,teta27,teta31,teta32,teta33,teta37,teta38,teta39):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta25)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta25)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta26)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta26)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta27)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta27)
    Z13 = Z12
    
    X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta31)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta31)
    
    X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta32)
    Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta32)
    
    X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta33)
    Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta33)
    Z23 = Z22
    
    #X31 = X23
    Y31 = rotation_d_angle_teta_autour_de_U(Y23,X23,teta37)
    Z31 = rotation_d_angle_teta_autour_de_U(Z23,X23,teta37)
    
    #X32 = rotation_d_angle_teta_autour_de_U(X31,Y31,teta38)
    #Y32 = Y31
    Z32 = rotation_d_angle_teta_autour_de_U(Z31,Y31,teta38)
    
    #X33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta39)
    #Y33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta39)
    #Z33 = Z32
    
    MM2 = rotation_d_angle_teta_autour_de_U(MM2,X0,teta25)
    MM2 = rotation_d_angle_teta_autour_de_U(MM2,Y11,teta26)
    MM2 = rotation_d_angle_teta_autour_de_U(MM2,Z12,teta27)
    
    MM2 = rotation_d_angle_teta_autour_de_U(MM2,X13,teta31)
    MM2 = rotation_d_angle_teta_autour_de_U(MM2,Y21,teta32)
    MM2 = rotation_d_angle_teta_autour_de_U(MM2,Z22,teta33)

    MM2 = rotation_d_angle_teta_autour_de_U(MM2,X23,teta37)
    MM2 = rotation_d_angle_teta_autour_de_U(MM2,Y31,teta38)
    MM2 = rotation_d_angle_teta_autour_de_U(MM2,Z32,teta39)
    
    return [M[0]+MM2[0],M[1]+MM2[1],M[2]+MM2[2]]

def donner_les_coordonnees_de_M3(M,MM3,teta25,teta26,teta27,teta31,teta32,teta33,teta37,teta38,teta39):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta25)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta25)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta26)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta26)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta27)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta27)
    Z13 = Z12
    
    X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta31)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta31)
    
    X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta32)
    Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta32)
    
    X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta33)
    Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta33)
    Z23 = Z22
    
    #X31 = X23
    Y31 = rotation_d_angle_teta_autour_de_U(Y23,X23,teta37)
    Z31 = rotation_d_angle_teta_autour_de_U(Z23,X23,teta37)
    
    #X32 = rotation_d_angle_teta_autour_de_U(X31,Y31,teta38)
    #Y32 = Y31
    Z32 = rotation_d_angle_teta_autour_de_U(Z31,Y31,teta38)
    
    #X33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta39)
    #Y33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta39)
    #Z33 = Z32
    
    MM3 = rotation_d_angle_teta_autour_de_U(MM3,X0,teta25)
    MM3 = rotation_d_angle_teta_autour_de_U(MM3,Y11,teta26)
    MM3 = rotation_d_angle_teta_autour_de_U(MM3,Z12,teta27)
    
    MM3 = rotation_d_angle_teta_autour_de_U(MM3,X13,teta31)
    MM3 = rotation_d_angle_teta_autour_de_U(MM3,Y21,teta32)
    MM3 = rotation_d_angle_teta_autour_de_U(MM3,Z22,teta33)

    MM3 = rotation_d_angle_teta_autour_de_U(MM3,X23,teta37)
    MM3 = rotation_d_angle_teta_autour_de_U(MM3,Y31,teta38)
    MM3 = rotation_d_angle_teta_autour_de_U(MM3,Z32,teta39)
    
    return [M[0]+MM3[0],M[1]+MM3[1],M[2]+MM3[2]]

def donner_les_coordonnees_de_M4(M,MM4,teta25,teta26,teta27,teta31,teta32,teta33,teta37,teta38,teta39):
    
    X0 = [1,0,0]
    Y0 = [0,1,0]
    Z0 = [0,0,1]
    
    X11 = X0
    Y11 = rotation_d_angle_teta_autour_de_U(Y0,X0,teta25)
    Z11 = rotation_d_angle_teta_autour_de_U(Z0,X0,teta25)
    
    X12 = rotation_d_angle_teta_autour_de_U(X11,Y11,teta26)
    Y12 = Y11
    Z12 = rotation_d_angle_teta_autour_de_U(Z11,Y11,teta26)
    
    X13 = rotation_d_angle_teta_autour_de_U(X12,Z12,teta27)
    Y13 = rotation_d_angle_teta_autour_de_U(Y12,Z12,teta27)
    Z13 = Z12
    
    X21 = X13
    Y21 = rotation_d_angle_teta_autour_de_U(Y13,X13,teta31)
    Z21 = rotation_d_angle_teta_autour_de_U(Z13,X13,teta31)
    
    X22 = rotation_d_angle_teta_autour_de_U(X21,Y21,teta32)
    Y22 = Y21
    Z22 = rotation_d_angle_teta_autour_de_U(Z21,Y21,teta32)
    
    X23 = rotation_d_angle_teta_autour_de_U(X22,Z22,teta33)
    Y23 = rotation_d_angle_teta_autour_de_U(Y22,Z22,teta33)
    Z23 = Z22
    
    #X31 = X23
    Y31 = rotation_d_angle_teta_autour_de_U(Y23,X23,teta37)
    Z31 = rotation_d_angle_teta_autour_de_U(Z23,X23,teta37)
    
    #X32 = rotation_d_angle_teta_autour_de_U(X31,Y31,teta38)
    #Y32 = Y31
    Z32 = rotation_d_angle_teta_autour_de_U(Z31,Y31,teta38)
    
    #X33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta39)
    #Y33 = rotation_d_angle_teta_autour_de_U(X32,Z32,teta39)
    #Z33 = Z32
    
    MM4 = rotation_d_angle_teta_autour_de_U(MM4,X0,teta25)
    MM4 = rotation_d_angle_teta_autour_de_U(MM4,Y11,teta26)
    MM4 = rotation_d_angle_teta_autour_de_U(MM4,Z12,teta27)
    
    MM4 = rotation_d_angle_teta_autour_de_U(MM4,X13,teta31)
    MM4 = rotation_d_angle_teta_autour_de_U(MM4,Y21,teta32)
    MM4 = rotation_d_angle_teta_autour_de_U(MM4,Z22,teta33)

    MM4 = rotation_d_angle_teta_autour_de_U(MM4,X23,teta37)
    MM4 = rotation_d_angle_teta_autour_de_U(MM4,Y31,teta38)
    MM4 = rotation_d_angle_teta_autour_de_U(MM4,Z32,teta39)
    
    return [M[0]+MM4[0],M[1]+MM4[1],M[2]+MM4[2]]

def tracer(A,B,couleur):
    plt.plot([A[0],B[0]],[A[2],B[2]],couleur)

def tracer_epaisseur(A,B,couleur,n):
    plt.plot([A[0],B[0]],[A[2],B[2]],couleur,linewidth = n)

def projection_sur_Oxz(E):
    [x,y,z] = E
    S = [x+(1/2)*y,0,(1/3)*y+z]
    return S

def translation(V,T):
    return([V[0]+T[0],V[1]+T[1],V[2]+T[2]])

def afficher_le_cube():
    
    A1,B1,C1,D1 = [0,0,0],[10,0,0],[10,10,0],[0,10,0]
    E1,F1,G1,H1 = [0,0,10],[10,0,10],[10,10,10],[0,10,10]
    
    A1 = projection_sur_Oxz(A1)
    B1 = projection_sur_Oxz(B1)
    C1 = projection_sur_Oxz(C1)
    D1 = projection_sur_Oxz(D1)
    E1 = projection_sur_Oxz(E1)
    F1 = projection_sur_Oxz(F1)
    G1 = projection_sur_Oxz(G1)
    H1 = projection_sur_Oxz(H1)
    
    tracer_epaisseur(A1,B1,"k",1/3)
    tracer_epaisseur(B1,C1,"k",1/3)
    tracer_epaisseur(C1,D1,"k",1/3)
    tracer_epaisseur(D1,A1,"k",1/3)
    
    tracer_epaisseur(E1,F1,"k",1/3)
    tracer_epaisseur(F1,G1,"k",1/3)
    tracer_epaisseur(G1,H1,"k",1/3)
    tracer_epaisseur(H1,E1,"k",1/3)
    
    tracer_epaisseur(A1,E1,"k",1/3)
    tracer_epaisseur(B1,F1,"k",1/3)
    tracer_epaisseur(C1,G1,"k",1/3)
    tracer_epaisseur(D1,H1,"k",1/3)




















