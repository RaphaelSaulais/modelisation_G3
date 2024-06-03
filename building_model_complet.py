# Write your code here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dm4bem

hauteur = 3 #hauteur appartement
L1 = 5 #largeur maison
L2 =9 #longueur maison
L3 = 2 #largeur petite pièce
L4 = 4 #longueur petite pièce
Lf = 1 #longeur fenêtre
            
Sg = Lf*1           # m² surface area of the glass 
Sc = Si = 2*L1*hauteur + 2*L2*hauteur - 2*Sg   # m² surface area of concrete & insulation of the extern walls
Scp = Sip = L3*hauteur + L4*hauteur #m² surface area of concrete & insulation of the little room

Φo = 800 # W/m2 average solar irradiation on the exterior wall
Φi = 100 # W/m2 surfacic power inside the building due to wall temperature
Qa = 2*83 + 50 + 110 + 10 # W Human activity and electronic devices in the bulding that brings power to the building
Φa = Φo


air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
pd.DataFrame(air, index=['Air'])

concrete = {'Conductivity': 1.400,          # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880,           # J/(kg⋅K)
            'Width': 0.2}                # m
           

insulation = {'Conductivity': 0.027,        # W/(m·K)
              'Density': 55.0,              # kg/m³
              'Specific heat': 1210,        # J/(kg⋅K)
              'Width': 0.08}                # m
              
glass = {'Conductivity': 1.4,               # W/(m·K)
         'Density': 2500,                   # kg/m³
         'Specific heat': 1210,             # J/(kg⋅K)
         'Width': 0.04,                     # m
         'Surface': Sg}                   # m²

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
G_cd = wall['Conductivity'] / wall['Width'] 
G_cd=pd.DataFrame(G_cd, columns=['Conductance/S'])
G_cd = G_cd.transpose()   
    
#conductance fenêtre
Gg = h * wall['Surface'].iloc[2]

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
Ggs = float(1 / (1 / Gg.loc['h', 'out'] + 1 / (2 * G_cd['Glass']*Sg)))

Cext = wall['Density'] * wall['Specific heat'] * Sc * wall['Width']
Cint = wall['Density'] * wall['Specific heat'] *Scp * wall['Width']
pd.DataFrame(Cext, columns=['Capacity'])
pd.DataFrame(Cint, columns=['Capacity'])

Cext['Air'] = air['Density'] * air['Specific heat'] * Va
Cint['Air'] = air['Density'] * air['Specific heat'] * Va
pd.DataFrame(Cext, columns=['Capacity'])
pd.DataFrame(Cint, columns=['Capacity'])

# temperature nodes
θ = ['θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5', 'θ6', 'θ7','θ8', 'θ9', 'θ10', 'θ11', 'θ12', 'θ13', 'θ14', 'θ15','θ16', 'θ17', 'θ18']

# flow-rate branches
q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11','q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20', 'q21', 'q22', 'q23']

G = np.array(np.hstack([h['out']*Sc,2 * G_cd['Layer_out']*Sc, 2 * G_cd['Layer_out']*Sc, 2 * G_cd['Layer_in']*Sc, 2 * G_cd['Layer_in']*Sc, h['in']*Sc,2*Ggs, Gv, Gg['in'],
Kp, h['in']*Scp, 2 * G_cd['Layer_in']*Scp, 2 * G_cd['Layer_in']*Scp,2 * G_cd['Layer_out']*Scp,2 * G_cd['Layer_out']*Scp, h['in']*Scp, h['in']*Scp,  2 * G_cd['Layer_in']*Scp,
2 * G_cd['Layer_in']*Scp, 2 * G_cd['Layer_out']*Scp,2 * G_cd['Layer_out']*Scp, h['out']*(Scp-Sg), Gg['in'], Ggs]))


# np.set_printoptions(precision=3, threshold=16, suppress=True)
# pd.set_option("display.precision", 1)
pd.DataFrame(G, index=q)

A = np.zeros([24, 19])       # n° of branches X n° of nodes
A[0, 0] = 1                 # branch 0: -> node 0
A[1, 0], A[1, 1] = -1, 1    # branch 1: node 0 -> node 1
A[2, 1], A[2, 2] = -1, 1    # branch 2: node 1 -> node 2
A[3, 2], A[3, 3] = -1, 1    # branch 3: node 2 -> node 3
A[4, 3], A[4, 4] = -1, 1    # branch 4: node 3 -> node 4
A[5, 4], A[5, 6] = -1, 1    # branch 5: node 4 -> node 6
A[6, 5]=  1                 # branch 6: node 5 
A[7, 6] = 1                 # branch 7: -> node 6
A[8, 5], A[8, 6] = -1, 1    # branch 8: node 5 -> node 6
A[9, 6] = 1                # branch 9: -> node 6
A[10, 6],A[10, 18] = 1,-1   # branch 10: node 18-> node 6
A[11, 8],A[11, 18] = -1,1   # branch 11: node 8-> node 18
A[12, 9],A[12, 8] = -1,1   # branch 12: node 9-> node 8
A[13, 10],A[13, 9] = -1,1   # branch 13: node 10-> node 9
A[14, 11],A[14, 10] = -1,1   # branch 14: node 11-> node 10
A[15, 12],A[15, 11] = -1,1   # branch 15: node 12-> node 11
A[16, 13],A[16, 12] = -1,1   # branch 16: node 13-> node 12
A[17, 14],A[17, 13] = -1,1   # branch 17: node 14-> node 13
A[18, 15],A[18, 14] = -1,1   # branch 18: node 15-> node 14
A[19, 16],A[19, 15] = -1,1   # branch 29: node 16-> node 15
A[20, 17],A[20, 16] = -1,1   # branch 20: node 17-> node 16
A[21, 17]=1   # branch 21: node 17
A[22, 7],A[22, 12] = -1,1   # branch 22: node 7-> node 12
A[23, 7]=1   # branch 23: node 7




C = np.array([0, Cext['Layer_out'], 0, Cext['Layer_in'], 0, 0,
                Cext['Air'], 0,Cint['Layer_in'],0,Cint['Layer_in'],0,Cint['Air'],0,Cext['Layer_in'],0,Cext['Layer_out'],0,0])

f = pd.Series(['Φo', 0, 0, 0, 'Φi', 0, 'Qa',0,0,0,0,0,'Qa',0,0,0,0, 'Φa',0],
              index=θ)
pd.DataFrame(f, index=θ)

b = pd.Series(['To', 0, 0, 0, 0, 0, 'To', 'To', 0, 'Ti_sp', 0,0,0,0,0,0,0,0,0,0,0,'To',0,'To'],
              index=q)
              
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

# TC['G']['q7'] = 1e3  # Kp -> ∞, almost perfect controller
TC['G']['q7'] = 0      # Kp -> 0, no controller (free-floating)


#state space rpzation
[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)
us


 ################## suite code steady state test



 
 # by default TC['G']['q11'] = 0, i.e. Kp -> 0, no controller (free-floating)
if controller:
    TC['G']['q11'] = 1e3        # Kp -> ∞, almost perfect controller

if neglect_air_glass_capacity:
    TC['C']['θ6'] = TC['C']['θ7'] = 0
    # or
    TC['C'].update({'θ6': 0, 'θ7': 0})
    
# State-space

[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)

controller = False
neglect_air_glass_capacity = False
imposed_time_step = False               
Δt = 500
bss = np.zeros(24)        # temperature sources b for steady state
bss[[0, 6, 7, 21, 23]] = 10      # outdoor temperature
bss[[9]] = 20            # indoor set-point temperature

fss = np.zeros(19)         # flow-rate sources f for steady state

A = TC['A']
G = TC['G']
diag_G = pd.DataFrame(np.diag(G), index=G.index, columns=G.index)

θss = np.linalg.inv(A.T @ diag_G @ A) @ (A.T @ diag_G @ bss + fss)
print(f'θss = {np.around(θss, 2)} °C')

bss = np.zeros(24)        # temperature sources b for steady state

fss = np.zeros(19)         # flow-rate sources f for steady state
fss[[6]] = 1000

θssQ = np.linalg.inv(A.T @ diag_G @ A) @ (A.T @ diag_G @ bss + fss)
print(f'θssQ = {np.around(θssQ, 2)} °C')

bT = np.array([10, 10, 10, 20, 10,10])     # [To, To, To, Tisp]
fQ = np.array([0, 0, 0, 0, 0])         # [Φo, Φi, Qa, Φa]
uss = np.hstack([bT, fQ])           # input vector for state space
print(f'uss = {uss}')

inv_As = pd.DataFrame(np.linalg.inv(As),
                      columns=As.index, index=As.index)
yss = (-Cs @ inv_As @ Bs + Ds) @ uss

yss = float(yss.values[0])
print(f'yss = {yss:.2f} °C')

print(f'Error between DAE and state-space: {abs(θss[6] - yss):.2e} °C')

bT = np.array([0, 0, 0 ,0 ,0,0])         # [To, To, To, Tisp]
fQ = np.array([0, 0, 1000, 0, 0])      # [Φo, Φi, Qa, Φa]
uss = np.hstack([bT, fQ])

inv_As = pd.DataFrame(np.linalg.inv(As),
                      columns=As.index, index=As.index)
yssQ = (-Cs @ inv_As @ Bs + Ds) @ uss

yssQ = float(yssQ.values[0])
print(f'yssQ = {yssQ:.2f} °C')

print(f'Error between DAE and state-space: {abs(θssQ[6] - yssQ):.2e} °C')

# Eigenvalues analysis
λ = np.linalg.eig(As)[0]        # eigenvalues of matrix As

# time step
dtmax = 2 * min(-1. / λ)    # max time step for Euler explicit stability
dm4bem.print_rounded_time('dtmax', dtmax)

if imposed_time_step:
    dt = Δt
else:
    dt = dm4bem.round_time(dtmax)
dm4bem.print_rounded_time('dt', dt)

if dt < 10:
    raise ValueError("Time step is too small. Stopping the script.")
    
# settling time
t_settle = 4 * max(-1 / λ)
dm4bem.print_rounded_time('t_settle', t_settle)

# duration: next multiple of 3600 s that is larger than t_settle
duration = np.ceil(t_settle / 3600) * 3600
dm4bem.print_rounded_time('duration', duration)

# Create input_data_set
# ---------------------
# time vector
n = int(np.floor(duration / dt))    # number of time steps

# DateTimeIndex starting at "00:00:00" with a time step of dt
time = pd.date_range(start="2000-01-01 00:00:00",
                           periods=n, freq=f"{int(dt)}S")

To = 10 * np.ones(n)        # outdoor temperature
Ti_sp = 20 * np.ones(n)     # indoor temperature set point
Φa = 0 * np.ones(n)         # solar radiation absorbed by the glass
Qa = Φo = Φi = Φa           # auxiliary heat sources and solar radiation

data = {'To': To, 'Ti_sp': Ti_sp, 'Φo': Φo, 'Φi': Φi, 'Qa': Qa, 'Φa': Φa}
input_data_set = pd.DataFrame(data, index=time)

# inputs in time from input_data_set
u = dm4bem.inputs_in_time(us, input_data_set)

# Initial conditions
θ_exp = pd.DataFrame(index=u.index)     # empty df with index for explicit Euler
θ_imp = pd.DataFrame(index=u.index)     # empty df with index for implicit Euler

θ0 = 0.0                    # initial temperatures
θ_exp[As.columns] = θ0      # fill θ for Euler explicit with initial values θ0
θ_imp[As.columns] = θ0      # fill θ for Euler implicit with initial values θ0

I = np.eye(As.shape[0])     # identity matrix
for k in range(u.shape[0] - 1):
    θ_exp.iloc[k + 1] = (I + dt * As)\
        @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k]
    θ_imp.iloc[k + 1] = np.linalg.inv(I - dt * As)\
        @ (θ_imp.iloc[k] + dt * Bs @ u.iloc[k])
        
# outputs
y_exp = (Cs @ θ_exp.T + Ds @  u.T).T
y_imp = (Cs @ θ_imp.T + Ds @  u.T).T

# plot results
y = pd.concat([y_exp, y_imp], axis=1, keys=['Explicit', 'Implicit'])
# Flatten the two-level column labels into a single level
y.columns = y.columns.get_level_values(0)

ax = y.plot()
ax.set_xlabel('Time')
ax.set_ylabel('Indoor temperature, $\\theta_i$ / °C')
ax.set_title(f'Time step: $dt$ = {dt:.0f} s; $dt_{{max}}$ = {dtmax:.0f} s')
plt.show()

print('Steady-state indoor temperature obtained with:')
print(f'- DAE model: {float(θss[6]):.4f} °C')
print(f'- state-space model: {float(yss):.4f} °C')
print(f'- steady-state response to step input: \
{y_exp["θ6"].tail(1).values[0]:.4f} °C')

# Create input_data_set
# ---------------------
# time vector
n = int(np.floor(duration / dt))    # number of time steps

# Create a DateTimeIndex starting at "00:00:00" with a time step of dt
time = pd.date_range(start="2000-01-01 00:00:00",
                           periods=n, freq=f"{int(dt)}S")
# Create input_data_set
To = 0 * np.ones(n)         # outdoor temperature
Ti_sp =  20 * np.ones(n)     # indoor temperature set point
Φa = 0 * np.ones(n)         # solar radiation absorbed by the glass
Φo = Φi = Φa                # solar radiation
Qa = 1000 * np.ones(n)      # auxiliary heat sources
data = {'To': To, 'Ti_sp': Ti_sp, 'Φo': Φo, 'Φi': Φi, 'Qa': Qa, 'Φa': Φa}
input_data_set = pd.DataFrame(data, index=time)

# Get inputs in time from input_data_set
u = dm4bem.inputs_in_time(us, input_data_set)

# Initial conditions
θ_exp[As.columns] = θ0      # fill θ for Euler explicit with initial values θ0
θ_imp[As.columns] = θ0      # fill θ for Euler implicit with initial values θ0

I = np.eye(As.shape[0])     # identity matrix
for k in range(u.shape[0] - 1):
    θ_exp.iloc[k + 1] = (I + dt * As)\
        @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k]
    θ_imp.iloc[k + 1] = np.linalg.inv(I - dt * As)\
        @ (θ_imp.iloc[k] + dt * Bs @ u.iloc[k])
        
# outputs
y_exp = (Cs @ θ_exp.T + Ds @  u.T).T
y_imp = (Cs @ θ_imp.T + Ds @  u.T).T

# plot results
y = pd.concat([y_exp, y_imp], axis=1, keys=['Explicit', 'Implicit'])
# Flatten the two-level column labels into a single level
y.columns = y.columns.get_level_values(0)
ax = y.plot()
ax.set_xlabel('Time')
ax.set_ylabel('Indoor temperature, $\\theta_i$ / °C')
ax.set_title(f'Time step: $dt$ = {dt:.0f} s; $dt_{{max}}$ = {dtmax:.0f} s')
plt.show()

print('Steady-state indoor temperature obtained with:')
print(f'- DAE model: {float(θssQ[6]):.4f} °C')
print(f'- state-space model: {float(yssQ):.4f} °C')
print(f'- steady-state response to step input: \
{y_exp["θ6"].tail(1).values[0]:.4f} °C')