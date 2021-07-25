# -*- coding: utf-8 -*-
"""
Created on Tue May 29 07:04:41 2018

@author: kerbr
"""




from donner_coordonnees import *



def affichage(E):

    
    
    plt.axis([0,15,0,15])
    afficher_le_cube()
    
    AB = [3/2,0,3**(3/2)/2]
    AC = [-3/2,0,3**(3/2)/2]
    AF = [0,0,3**(3/2)/2]
    FG = [0,0,1]
    GG1 = [1/2,0,0]
    GG2 = [1/2,0,1/2]
    GG3 = [-1/2,0,1/2]
    GG4 = [-1/2,0,0]
    BD = [3/2,0,0]
    DE = [3/2,0,0]
    EE1 = [0,0,-1/2]
    EE2 = [1/2,0,-1/2]
    EE3 = [1/2,0,1/2]
    EE4 = [0,0,1/2]
    CH = [-3/2,0,0]
    HI = [-3/2,0,0]
    II1 = [0,0,1/2]
    II2 = [-1/2,0,1/2]
    II3 = [-1/2,0,-1/2]
    II4 = [0,0,-1/2]
    AL = [0,0,-3]
    AJ = [0,0,-3]
    LM = [0,0,-3/2]
    JK = [0,0,-3/2]
    KK1 = [-1/2,0,0]
    KK2 = [-1/2,0,-1/2]
    KK3 = [1/2,0,-1/2]
    KK4 = [1/2,0,0]
    MM1 = [-1/2,0,0]
    MM2 = [-1/2,0,-1/2]
    MM3 = [1/2,0,-1/2]
    MM4 = [+1/2,0,0]
    
    
    longueurs = [AB,AC,AF,FG,GG1,GG2,GG3,GG4,BD,DE,EE1,EE2,EE3,EE4,CH,HI,II1,II2,II3,II4,AL,AJ,LM,JK,KK1,KK2,KK3,KK4,MM1,MM2,MM3,MM4] 
    
    [A,B,C,F,G,G1,G2,G3,G4,D,E,H,I,E1,E2,E3,E4,I1,I2,I3,I4,L,J,M,K,K1,K2,K3,K4,M1,M2,M3,M4] = fonction_donner_la_valeur(E,longueurs)
    
    [A,B,C,F,G,G1,G2,G3,G4,D,E,H,I,E1,E2,E3,E4,I1,I2,I3,I4,L,J,M,K,K1,K2,K3,K4,M1,M2,M3,M4] = [projection_sur_Oxz(A),projection_sur_Oxz(B),projection_sur_Oxz(C),projection_sur_Oxz(F),projection_sur_Oxz(G),projection_sur_Oxz(G1),projection_sur_Oxz(G2),projection_sur_Oxz(G3),projection_sur_Oxz(G4),projection_sur_Oxz(D),projection_sur_Oxz(E),projection_sur_Oxz(H),projection_sur_Oxz(I),projection_sur_Oxz(E1),projection_sur_Oxz(E2),projection_sur_Oxz(E3),projection_sur_Oxz(E4),projection_sur_Oxz(I1),projection_sur_Oxz(I2),projection_sur_Oxz(I3),projection_sur_Oxz(I4),projection_sur_Oxz(L),projection_sur_Oxz(J),projection_sur_Oxz(M),projection_sur_Oxz(K),projection_sur_Oxz(K1),projection_sur_Oxz(K2),projection_sur_Oxz(K3),projection_sur_Oxz(K4),projection_sur_Oxz(M1),projection_sur_Oxz(M2),projection_sur_Oxz(M3),projection_sur_Oxz(M4)]
    
    
    A_tracer = [[A,B],[A,C],[B,C],[F,G],[G1,G2],[G2,G3],[G3,G4],[G4,G1],[B,D],[D,E],[C,H],[H,I],[E1,E2],[E2,E3],[E3,E4],[E4,E1],[I1,I2],[I2,I3],[I3,I4],[I4,I1],[A,L],[A,J],[L,M],[J,K],[K1,K2],[K2,K3],[K3,K4],[K4,K1],[M1,M2],[M2,M3],[M3,M4],[M4,M1]]
    
    
    for i in A_tracer:
        tracer(i[0],i[1],"b")
    



"""

valeurs = []
for i in np.linspace(0,1,10):
    valeurs+=[i]

for i in np.linspace(1,-1,10):
    valeurs+=[i]
for i in np.linspace(-1,0,10):
    valeurs+=[i]
""""""

compteur = 0
i,j,k = 0,0,0
for i in valeurs:
    
    torse =                             [i*4*pi/5,0,0]                     #1,2,3
    tete =                              [0,0,0]                     #4,5,6
    bras_gauche =                       [0,0,-pi/4]                     #7,8,9
    avant_bras_gauche =                 [0,pi/6,0]                     #10,11,12
    bras_droit =                        [0,0,pi/4]                     #13,14,15
    avant_bras_droit =                  [0,-pi/6,0]                     #16,17,18
    main_gauche =                       [0,0,0]                     #19,20,21
    main_droite =                       [0,0,0]                     #22,23,24
    cuisse_gauche =                     [0,pi/4,0]                     #25,26,27
    cuisse_droite =                     [0,-pi/4,0]                     #28,29,30
    tibia_gauche =                      [0,-pi/4,0]                     #31,32,33
    tibia_droit =                       [0,pi/4,0]                     #34,35,36
    pied_gauche =                       [0,0,0]                     #37,38,39
    pied_droit =                        [0,0,0]                     #40,41,42   

    E = torse + tete + bras_gauche + avant_bras_gauche + bras_droit + avant_bras_droit + main_gauche + main_droite + cuisse_gauche + cuisse_droite + tibia_gauche + tibia_droit + pied_gauche + pied_droit
    
    affichage(E)

    plt.savefig("figur"+str(compteur)+".png")
    plt.show()
    compteur+=1

for j in valeurs:
    
    torse =                             [0,i*4*pi/5,0]                    #1,2,3
    tete =                              [0,0,0]                     #4,5,6
    bras_gauche =                       [0,0,-pi/4]                     #7,8,9
    avant_bras_gauche =                 [0,pi/6,0]                     #10,11,12
    bras_droit =                        [0,0,pi/4]                     #13,14,15
    avant_bras_droit =                  [0,-pi/6,0]                     #16,17,18
    main_gauche =                       [0,0,0]                     #19,20,21
    main_droite =                       [0,0,0]                     #22,23,24
    cuisse_gauche =                     [0,pi/4,0]                     #25,26,27
    cuisse_droite =                     [0,-pi/4,0]                     #28,29,30
    tibia_gauche =                      [0,-pi/4,0]                     #31,32,33
    tibia_droit =                       [0,pi/4,0]                     #34,35,36
    pied_gauche =                       [0,0,0]                     #37,38,39
    pied_droit =                        [0,0,0]                     #40,41,42

    E = torse + tete + bras_gauche + avant_bras_gauche + bras_droit + avant_bras_droit + main_gauche + main_droite + cuisse_gauche + cuisse_droite + tibia_gauche + tibia_droit + pied_gauche + pied_droit
    
    affichage(E)

    plt.savefig("figur"+str(compteur)+".png")
    plt.show()
    compteur+=1


for k in valeurs:
    
    torse =                             [0,0,i*4*pi/5]                  #1,2,3
    tete =                              [0,0,0]                     #4,5,6
    bras_gauche =                       [0,0,-pi/4]                     #7,8,9
    avant_bras_gauche =                 [0,pi/6,0]                     #10,11,12
    bras_droit =                        [0,0,pi/4]                     #13,14,15
    avant_bras_droit =                  [0,-pi/6,0]                     #16,17,18
    main_gauche =                       [0,0,0]                     #19,20,21
    main_droite =                       [0,0,0]                     #22,23,24
    cuisse_gauche =                     [0,pi/4,0]                     #25,26,27
    cuisse_droite =                     [0,-pi/4,0]                     #28,29,30
    tibia_gauche =                      [0,-pi/4,0]                     #31,32,33
    tibia_droit =                       [0,pi/4,0]                     #34,35,36
    pied_gauche =                       [0,0,0]                     #37,38,39
    pied_droit =                        [0,0,0]                     #40,41,42

    E = torse + tete + bras_gauche + avant_bras_gauche + bras_droit + avant_bras_droit + main_gauche + main_droite + cuisse_gauche + cuisse_droite + tibia_gauche + tibia_droit + pied_gauche + pied_droit
    
    affichage(E)
    print(os.getcwd())
    plt.savefig("figur"+str(compteur)+".png")
    plt.show()
    compteur+=1

valeurs.reverse()
"""
"""
for k in valeurs:
    
    torse =                             [i*4*pi/5,j*4*pi/5,k*4*pi/5]                  #1,2,3
    tete =                              [0,0,0]                     #4,5,6
    bras_gauche =                       [0,0,-pi/4]                     #7,8,9
    avant_bras_gauche =                 [0,pi/6,0]                     #10,11,12
    bras_droit =                        [0,0,pi/4]                     #13,14,15
    avant_bras_droit =                  [0,-pi/6,0]                     #16,17,18
    main_gauche =                       [0,0,0]                     #19,20,21
    main_droite =                       [0,0,0]                     #22,23,24
    cuisse_gauche =                     [0,pi/4,0]                     #25,26,27
    cuisse_droite =                     [0,-pi/4,0]                     #28,29,30
    tibia_gauche =                      [0,-pi/4,0]                     #31,32,33
    tibia_droit =                       [0,pi/4,0]                     #34,35,36
    pied_gauche =                       [0,0,0]                     #37,38,39
    pied_droit =                        [0,0,0]                     #40,41,42

    E = torse + tete + bras_gauche + avant_bras_gauche + bras_droit + avant_bras_droit + main_gauche + main_droite + cuisse_gauche + cuisse_droite + tibia_gauche + tibia_droit + pied_gauche + pied_droit
    
    affichage(E)
    print(os.getcwd())
    plt.savefig("figur"+str(compteur)+".png")
    plt.show()
    compteur+=1




for j in valeurs:
    
    torse =                             [i*4*pi/5,j*4*pi/5,k*4*pi/5]                     #1,2,3
    tete =                              [0,0,0]                     #4,5,6
    bras_gauche =                       [0,0,-pi/4]                     #7,8,9
    avant_bras_gauche =                 [0,pi/6,0]                     #10,11,12
    bras_droit =                        [0,0,pi/4]                     #13,14,15
    avant_bras_droit =                  [0,-pi/6,0]                     #16,17,18
    main_gauche =                       [0,0,0]                     #19,20,21
    main_droite =                       [0,0,0]                     #22,23,24
    cuisse_gauche =                     [0,pi/4,0]                     #25,26,27
    cuisse_droite =                     [0,-pi/4,0]                     #28,29,30
    tibia_gauche =                      [0,-pi/4,0]                     #31,32,33
    tibia_droit =                       [0,pi/4,0]                     #34,35,36
    pied_gauche =                       [0,0,0]                     #37,38,39
    pied_droit =                        [0,0,0]                     #40,41,42

    E = torse + tete + bras_gauche + avant_bras_gauche + bras_droit + avant_bras_droit + main_gauche + main_droite + cuisse_gauche + cuisse_droite + tibia_gauche + tibia_droit + pied_gauche + pied_droit
    
    affichage(E)

    plt.savefig("figur"+str(compteur)+".png")
    plt.show()
    compteur+=1

"""
#for i in valeurs:
 
if True:
    i,j,k = -0.6,0,0
    torse =                             [i*4*pi/5,j*4*pi/5,k*4*pi/5]                     #1,2,3
    tete =                              [0,0,0]                     #4,5,6
    bras_gauche =                       [0,0,-pi/4]                     #7,8,9
    avant_bras_gauche =                 [0,pi/6,0]                     #10,11,12
    bras_droit =                        [0,0,pi/4]                     #13,14,15
    avant_bras_droit =                  [0,-pi/6,0]                     #16,17,18
    main_gauche =                       [0,0,0]                     #19,20,21
    main_droite =                       [0,0,0]                     #22,23,24
    cuisse_gauche =                     [0,pi/4,0]                     #25,26,27
    cuisse_droite =                     [0,-pi/4,0]                     #28,29,30
    tibia_gauche =                      [0,-pi/4,0]                     #31,32,33
    tibia_droit =                       [0,pi/4,0]                     #34,35,36
    pied_gauche =                       [0,0,0]                     #37,38,39
    pied_droit =                        [0,0,0]                     #40,41,42   

    E = torse + tete + bras_gauche + avant_bras_gauche + bras_droit + avant_bras_droit + main_gauche + main_droite + cuisse_gauche + cuisse_droite + tibia_gauche + tibia_droit + pied_gauche + pied_droit
    
    affichage(E)

    plt.savefig("figur"+str(compteur)+".png")
    plt.show()
    compteur+=1








"""
compteur = 0


valeurs = []
for i in np.linspace(0,1,5):
    valeurs+=[i]
for i in np.linspace(1,0,5):
    valeurs+=[i]
for i in np.linspace(-1,0,5):
    valeurs+=[i]

for i in valeurs:
    
    torse =                             [0,0,0]                     #1,2,3
    tete =                              [0,0,0]                     #4,5,6
    bras_gauche =                       [0,0,0]                     #7,8,9
    avant_bras_gauche =                 [0,0,0]                     #10,11,12
    bras_droit =                        [0,0,0]                     #13,14,15
    avant_bras_droit =                  [0,0,0]                     #16,17,18
    main_gauche =                       [0,0,0]                     #19,20,21
    main_droite =                       [0,0,0]                     #22,23,24
    cuisse_gauche =                     [0,0,0]                     #25,26,27
    cuisse_droite =                     [0,0,0]                     #28,29,30
    tibia_gauche =                      [0,0,0]                     #31,32,33
    tibia_droit =                       [0,0,0]                     #34,35,36
    pied_gauche =                       [0,0,0]                     #37,38,39
    pied_droit =                        [0,0,0]                     #40,41,42

    E = torse + tete + bras_gauche + avant_bras_gauche + bras_droit + avant_bras_droit + main_gauche + main_droite + cuisse_gauche + cuisse_droite + tibia_gauche + tibia_droit + pied_gauche + pied_droit
    
    affichage(E)

    plt.savefig("figure"+str(compteur)+".png")
    plt.show()
    compteur+=1
"""

