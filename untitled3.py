# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import math



def calc_alfa_n(V_0):
  #return 0.01*(V_0-10)/((math.e)**(((V_0-10)/10)-1))
  return 0.01*(V_0-10)/((math.e**((V_0-10)/10))-1)

def calc_beta_n(V_0):
  return 0.125*math.e**(V_0/80)

def calc_alfa_m(V_0):
  #return 0.1*(V_0-25)/(math.e**(((V_0-25)/10)-1))
  return 0.1*(V_0-25)/((math.e**((V_0-25)/10))-1)

def calc_beta_m(V_0):
  return 4*math.e**(V_0/18)

def calc_alfa_h(V_0):
  return 0.07*math.e**(V_0/20)

def calc_beta_h(V_0):
  #return 1/(math.e**(((V_0-30)/10)+1))
  return 1/((math.e**((V_0-30)/10))+1)

def calc_coef(alfa, beta, dt, coef_0): ##calcula m, n e h quando eles são passados por parâmetro
  return coef_0 + dt*(alfa*(1-coef_0)-beta*coef_0)

def calc_Gna(Gna_0, m_0, h_0):
  print(m_0,h_0,Gna_0)
  print(Gna_0*(m_0**3)*h_0)
  return Gna_0*(m_0**3)*h_0

def calc_Gk(Gk_0, n_0):
  return Gk_0*(n_0**4)

def calc_In(Gn_0, V_0, Vn):
  return Gn_0*(V_0-Vn)

def calc_V(V_0, dt, Ik, Ina, Il, Iapp):
  return V_0 + dt*(Iapp-Ik-Ina-Il)

##essas variáveis dependenm de V
#http://sisne.org/Disciplinas/Grad/IntrodNeurocComput/Modelo_de_Hodgkin_Huxley.pdf   pag32
V = [0]

alfa_m = [calc_alfa_m(V[-1])]
beta_m = [calc_beta_m(V[-1])]

alfa_n = [calc_alfa_n(V[-1])]
beta_n = [calc_beta_n(V[-1])]

alfa_h = [calc_alfa_h(V[-1])]
beta_h = [calc_beta_h(V[-1])]

dt = 0.01

Gna_0 = 120
Gk_0 = 36
Gl = 0.3
Cm = 1
Vna = 115
Vk = -12
Vl = 10.6

#m = [calc_coef(alfa_m[-1], beta_m[-1], dt, 0)] #0.05
#n = [calc_coef(alfa_n[-1], beta_n[-1], dt, 0)] #0.32
#h = [calc_coef(alfa_h[-1], beta_h[-1], dt, 0)] #0.6

m = [0.05] #0.05
n = [0.32] #0.32
h = [0.6] #0.6

T = [0]

Gna = [calc_Gna(Gna_0, m[-1], h[-1])]
Gk = [calc_Gk(Gk_0, n[-1])]

print(Gna,V)
Ina = [calc_In(Gna[-1], V[-1], Vna)]
Ik = [calc_In(Gk[-1], V[-1], Vk)]
Il = [calc_In(Gl, V[-1], Vl)]

Iapp = [0]
# or (T[-1]>=50 and T[-1]<=60)
# or (T[-1]>=35 and T[-1]<=40)

while T[-1]<200:
  if (T[-1]>=20 and T[-1]<=30) or (T[-1]>=35 and T[-1]<=40) or (T[-1]>=65 and T[-1]<=75):
    Iapp.append(150)
  else:
    Iapp.append(0)

  alfa_m.append(calc_alfa_m(V[-1]))
  beta_m.append(calc_beta_m(V[-1]))

  alfa_n.append(calc_alfa_n(V[-1]))
  beta_n.append(calc_beta_n(V[-1]))

  alfa_h.append(calc_alfa_h(V[-1]))
  beta_h.append(calc_beta_h(V[-1]))

  m.append(calc_coef(alfa_m[-1], beta_m[-1], dt, m[-1]))
  n.append(calc_coef(alfa_n[-1], beta_n[-1], dt, n[-1]))
  h.append(calc_coef(alfa_h[-1], beta_h[-1], dt, h[-1]))

  Gna.append(calc_Gna(Gna_0, m[-1], h[-1]))
  Gk.append(calc_Gk(Gk_0, n[-1]))

  Ina.append(calc_In(Gna[-1], V[-1], Vna))
  Ik.append(calc_In(Gk[-1], V[-1], Vk))
  Il.append(calc_In(Gl, V[-1], Vl))

  V.append(calc_V(V[-1], dt, Ik[-1], Ina[-1], Il[-1], Iapp[-1]))

  T.append(T[-1]+dt)


plt.figure(figsize=(10,7.5))
plt.plot(T, V, color="purple") ##Criação do gráfico
plt.plot(T, Iapp, color="yellow")
plt.title("Gráfico da tensão da membrana em função do tempo")
plt.xlabel("Tempo (segundos)")
plt.ylabel("Corrente (A")
plt.grid(True)

plt.show()

plt.figure(figsize=(10,7.5))
plt.plot(T, m, color="purple") ##Criação do gráfico
plt.plot(T, n, color="green")
plt.plot(T, h, color="blue")
plt.title("Gráfico da tensão da membrana em função do tempo")
plt.xlabel("Tempo (segundos)")
plt.ylabel("Corrente (A")
plt.grid(True)
plt.legend("mnh")

plt.show()