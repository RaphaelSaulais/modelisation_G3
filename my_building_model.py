# Write your code here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dm4bem

hauteur = 3 #hauteur appartement
L1 = 5 #largeur maison
L2 =9 #longueur maison
L3 = 2 #largeur petite pièce
L4 = 4 #longueur petite pièce
Lf = 1 #longeur fenêtre
            
Sg = Lf*hauteur*1           # m² surface area of the glass 
Sc = Si = 2*L1*hauteur + 2*L2*hauteur - Sg   # m² surface area of concrete & insulation of the extern walls
Scp = Sip = L3*hauteur + L4*hauteur #m² surface area of concrete & insulation of the little room

Φo = 800 # W/m2 average solar irradiation on the exterior wall
Φi = 100 # W/m2 surfacic power inside the building due to wall temperature
Qa = 2*83 + 50 + 110 + 10 # W Human activity in the bulding that brings power to the building
Φa = Φo


air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
pd.DataFrame(air, index=['Air'])

concrete = {'Conductivity': 1.400,          # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880,           # J/(kg⋅K)
            'Width': 0.2,                   # m
            'Surfaceext': Sc, #m² surface exterieure 
            'Surfacemurpiece' : Scp}            # m² surface des murs internes de la petite pièce

insulation = {'Conductivity': 0.027,        # W/(m·K)
              'Density': 55.0,              # kg/m³
              'Specific heat': 1210,        # J/(kg⋅K)
              'Width': 0.08,                # m
              'Surfaceext': Si,             #m² surface exterieure
              'Surfacemurpiece':  Sip}          # m² surface des murs internes de la petite pièce

glass = {'Conductivity': 1.4,               # W/(m·K)
         'Density': 2500,                   # kg/m³
         'Specific heat': 1210,             # J/(kg⋅K)
         'Width': 0.04,                     # m
         'Surfaceext': Sg}                   # m²

wall = pd.DataFrame.from_dict({'Layer_out': concrete,
                               'Layer_in': insulation,
                               'Glass': glass},
                              orient='index')
wall


# radiative properties
ε_wLW = 0.85    # long wave emmisivity: wall surface (concrete)
ε_gLW = 0.90    # long wave emmisivity: glass pyrex
α_wSW = 0.25    # short wave absortivity: white smooth surface
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass

σ = 5.67e-8     # W/(m²⋅K⁴) Stefan-Bolzmann constant
print(f'σ = {σ} W/(m²⋅K⁴)')

h = pd.DataFrame([{'in': 8., 'out': 25}], index=['h'])  # W/(m²⋅K)
h

# conduction
G_cd_ext = wall['Conductivity'] / wall['Width'] * wall['Surfaceext']
G_cd_in =  wall['Conductivity'] / wall['Width'] * wall['Surfacemurpiece']

G_cd = pd.DataFrame({
    'Surface ext': G_cd_ext,
    'Mur petite piece': G_cd_in
})

# convection
Gw = h * wall['Surface'].iloc[0]     # wall
Gg = h * wall['Surface'].iloc[2]     # glass 

#conductance fenêtre

Gf = Gg + 

# ventilation flow rate
Va = L1*L2*hauteur - L3*L4*hauteur                 # m³, volume of air grande pièce
ACH = 1                     # 1/h, air changes per hour
Va_dot = ACH / 3600 * Va    # m³/s, air infiltration

# ventilation & advection
Gv = air['Density'] * air['Specific heat'] * Va_dot

# P-controler gain
# Kp = 1e4            # almost perfect controller Kp -> ∞
# Kp = 1e-3           # no controller Kp -> 0
Kp = 0

# glass: convection outdoor & conduction
Ggs = float(1 / (1 / Gg.loc['h', 'out'] + 1 / (2 * G_cd['Glass'])))

Cext = wall['Density'] * wall['Specific heat'] * wall['Surfaceext'] * wall['Width']
Cint = wall['Density'] * wall['Specific heat'] * wall['Surfacemurpiece'] * wall['Width']
pd.DataFrame(Cext, columns=['Capacity'])
pd.DataFrame(Cint, columns=['Capacity'])

Cext['Air'] = air['Density'] * air['Specific heat'] * Va
Cint['Air'] = air['Density'] * air['Specific heat'] * Va
pd.DataFrame(Cext, columns=['Capacity'])
pd.DataFrame(Cint, columns=['Capacity'])

# temperature nodes
θ = ['θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5', 'θ6', 'θ7','θ8', 'θ9', 'θ10', 'θ11', 'θ12', 'θ13', 'θ14', 'θ15','θ16', 'θ17', 'θ18']

# flow-rate branches
q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11','q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20', 'q21', 'q22', 'q23','q24']

G = np.array(np.hstack([Gw['out'],2 * G_cd['Layer_out'], 2 * G_cd['Layer_out'], 2 * G_cd['Layer_in'], 2 * G_cd['Layer_in'],GLW, Gw['in'],Gg['in'],
 Ggs, 2 * G_cd['Glass'],Gv, Kp]))

# np.set_printoptions(precision=3, threshold=16, suppress=True)
# pd.set_option("display.precision", 1)
pd.DataFrame(G, index=q)

A = np.zeros([24, 18])       # n° of branches X n° of nodes
A[0, 0] = 1                 # branch 0: -> node 0
A[1, 0], A[1, 1] = -1, 1    # branch 1: node 0 -> node 1
A[2, 1], A[2, 2] = -1, 1    # branch 2: node 1 -> node 2
A[3, 2], A[3, 3] = -1, 1    # branch 3: node 2 -> node 3
A[4, 3], A[4, 4] = -1, 1    # branch 4: node 3 -> node 4
A[5, 4], A[5, 6] = -1, 1    # branch 5: node 4 -> node 6
#pas de branch 6
A[7, 5]=  1                 # branch 7: node 5 
A[8, 6] = 1                 # branch 8: -> node 6
A[9, 5], A[9, 6] = -1, 1    # branch 9: node 5 -> node 6
A[10, 6] = 1                # branch 10: -> node 6
A[11, 6],A[11, 18] = 1,-1   # branch 11: node 18-> node 6
A[12, 8],A[12, 18] = -1,1   # branch 12: node 8-> node 18
A[13, 9],A[13, 8] = -1,1   # branch 13: node 9-> node 8
A[14, 10],A[13, 9] = -1,1   # branch 14: node 10-> node 9
A[15, 11],A[15, 10] = -1,1   # branch 15: node 11-> node 10
A[16, 12],A[16, 11] = -1,1   # branch 16: node 12-> node 11
A[17, 13],A[17, 12] = -1,1   # branch 17: node 13-> node 12
A[18, 14],A[18, 13] = -1,1   # branch 18: node 14-> node 13
A[19, 15],A[19, 14] = -1,1   # branch 19: node 15-> node 14
A[20, 16],A[20, 15] = -1,1   # branch 20: node 16-> node 15
A[21, 17],A[21, 16] = -1,1   # branch 21: node 17-> node 16
A[22, 17]=1   # branch 22: node 17
A[23, 7],A[23, 12] = -1,1   # branch 23: node 7-> node 12
A[24, 7]=1   # branch 24: node 7




 C = np.array([0, Cext['Layer_out'], 0, Cext['Layer_in'], 0, 0,
                Cext['Air'], 0,Cint['Layer_in'],0,Cint['Layer_in'],0,Cint['Air'],0,Cext['Layer_in'],0,Cext['Layer_out'],0,0])

f = pd.Series(['Φo', 0, 0, 0, 'Φi', 0, 'Qa',0,0,0,0,0,'Qa',0,0,0,0, 'Φa',0],
              index=θ)
pd.DataFrame(f, index=θ)

y = np.zeros(19)         # nodes
y[[6,12]] = 1              # nodes (temperatures) of interest
pd.DataFrame(y, index=θ)


# thermal circuit
A = pd.DataFrame(A, index=q, columns=θ)
G = pd.Series(G, index=q)
C = pd.Series(C, index=θ)
b = pd.Series(b, index=q)
f = pd.Series(f, index=θ)
y = pd.Series(y, index=θ)

TC = {"A": A,
      "G": G,
      "C": C,
      "b": b,
      "f": f,
      "y": y}

# TC['G']['q11'] = 1e3  # Kp -> ∞, almost perfect controller
TC['G']['q10'] = 0      # Kp -> 0, no controller (free-floating)


#state space rpzation
[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)
us
                