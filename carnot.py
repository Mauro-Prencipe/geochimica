# Ciclo di Carnot (su un gas ideale)

# Uso del programma:
# Una volta lanciato il programma, si usi la funzione "ciclo(T1,T2)"
# per effettuare il calcolo tra le temperature T1 e T2:

# ciclo(300,600)

# per un calcolo con T1=300K e T2=600K
# Il volume di partenza (in litri) e' fissato nella variabile vol.V1_1
# il volume al termine della prima compressione isoterma alla temperatura
# T1 e' fissato nella variabile vol.V2_1
# Cambiare questi volumi direttamente nel programma, e poi rilanciare,
# volendo fare il calcolo in condizioni diverse

import numpy as np
import matplotlib.pyplot as plt

# Quello che segue e' un esempio abbastanza avanzato di programmazione a
# oggetti. Per questo semplice programma potremmo evitare il ricorso
# a una struttura del genere pero', per imparare, e' utile far uso di
# queste tecniche appunto in brevi programmi.

# Le "classi" sono strutture che contengono insieme dati e funzioni che 
# accedono, o modificano, quei dati. Qui costruiamo la classe "volume"
# che verra' utlizzata per fissare il volume iniziale da cui parte
# il ciclo di Carnot, e il volume finale al termine della prima 
# compressione isoterma alla temperatura T1.
# Piu' avanti verra' definita una variabile "vol" che e' di tipo "volume"
# e questa variabile "vol" conterrà i due volumi e le funzioni per settarli.
# diciamo che vol e' una istanza della struttura volume
# La classe e tutte le sue possibili istanze viene inizializzata con i due
# valori di V1_1 e V2_1 (funzione __init__).
# V1_1 e V2_1 possino essere modificati con le funzioni "v_ini" e "v_fin"
# La funzione "out" visualizza i volumi.
# 
class volume():
    def __init__(self):
      self.V1_1=20
      self.V2_1=5
    def v_ini(self,Vini):
        self.V1_1=Vini
        print("Volume iniziale fissato a %5.2f l" % self.V1_1)
    def v_fin(self,Vfin):
        self.V2_1=Vfin
        print("Volume finale fissato a %5.2f l" % self.V2_1)
    def out(self):
        print("Volumi iniziale e finale: %5.2f e %5.2f l" % (self.V1_1, self.V2_1))

# Definizione delle costanti fondamentali
R=0.082              # costante dei gas (l*atm/mole K)
Cv=3*R/2             # calore spec. a V costante del gas ideale monoatomico
Cp=Cv+R              # calore spec. a P costante
gamma=Cp/Cv          # PV^gamma per adiabatica
gm1=gamma-1          # TV^(gamma-1) per adiabatica

# Creiamo adesso la variabile vol che e' un'istanza della classe volume.
# vol conterra' i due volumi vol.V1_1 e vol.V2_1. 
# Si noti la sintassi:le variabili (e anche le funzioni) che sono di classe 
# vol vengono richiamate premettendo il nome della classe (vol) 
# seguito da un ".".
vol=volume()

# Volendo modificare i due volumi in run successivi, non e'
# necessario riavviare il programma, ma e' sufficiente usare le funzioni
# vol.v_ini e vol.v_fin.

# La funzione "pressione" puo' essere utilizzata per fissare dei volumi
# iniziale e finale lungo l'isoterma T1, che corrispondano a determinati
# valori di pressione (iniziale e finale)
# Esempio:
# pressione(300,1,10)  
# che fissa il volume iniziale in modo che la pressione iniziale sia di 1 atm
# a T=300K, e il volume finale in modo che corrisponda a una pressione
# finale di 10 atm, alla stessa temperatura
def pressione(T1,Pini,Pfin):
    Vini=R*T1/Pini
    Vfin=R*T1/Pfin
    vol.v_ini(Vini)
    vol.v_fin(Vfin)
    
# Funzione che costruisce il ciclo di Carnot
def ciclo(T1,T2):

# Pressioni corrispondenti ai volumi vol.V2_1 e vol.V1_1 alla temperatura T1
    P2_1=R*T1/vol.V2_1
    P1_1=R*T1/vol.V1_1
    
# Calcolo delle costanti per la costruzione delle adiabatiche
# PV^gamma = cost
# TV^(gamma-1) = cost
    
# T1*vol.V1_1^(gamma-1)=c1
# T1*vol.V2_1^(gamma-1)=c2
    
# P1_1*vol.V1_1^gamma=c3
# P2_1*vol.V2_1^gamma=c4
    
    c1=T1*(vol.V1_1**gm1)
    c2=T1*(vol.V2_1**gm1)
    c3=P1_1*(vol.V1_1**gamma)
    c4=P2_1*(vol.V2_1**gamma)

# V1_2: volume alla temperatura T2 sull'adiabatica che parte da vol.V1_1
# V2_2: volume alla temperatura T2 sull'adiabatica che parte da vol.V2_1
    V1_2=(c1/T2)**(1/gm1)
    V2_2=(c2/T2)**(1/gm1)

# Pressioni corrispondenti ai due volumi  
    P1_2=R*T2/V1_2
    P2_2=R*T2/V2_2
    
# liste di volumi sulle due isoterme T1 e T2, tra i volumi minimo e
# massimo, e calcolo delle pressioni corrispondenti dall'equazione delle
# isoterme
    
    npunti=100
    V1_list=np.linspace(vol.V1_1,vol.V2_1,npunti)
    P1_list=R*T1/V1_list
    V2_list=np.linspace(V2_2,V1_2,npunti)
    P2_list=R*T2/V2_list

# liste di volumi sulle due adiabatiche e calcolo delle pressioni
# corrispondenti usando l'equazione per l'adiabatica
    V1a_list=np.linspace(V1_2,vol.V1_1,npunti)
    V2a_list=np.linspace(V2_2,vol.V2_1,npunti)
    P1a_list=c3/V1a_list**gamma
    P2a_list=c4/V2a_list**gamma

# ------ Sezione di plot ------
    
# Delta_x e Delta_y sono usati per fissare
# gli estremi degli assi V e P
    Delta_x=0.1*(vol.V1_1-V2_2)
    Delta_y=0.1*(P2_2-P1_1)
    Vmax=vol.V1_1+1.5*Delta_x
    Vmin=V2_2-Delta_x
    Pmin=P1_1-Delta_y
    Pmax=P2_2+Delta_y
    
# shift per posizione di etichette nel grafico
    shx=(Vmax-Vmin)*0.01
    shy=(Pmax-Pmin)*0.01
    if Vmin < 0.:
        Vmin=0.
    if Pmin < 0.:
        Pmin=0.
        
    fig=plt.figure()
    ax=fig.add_subplot(111)
    str1="isoterma T1="+str(T1)+" K"
    str2="isoterma T2="+str(T2)+" K"
    ax.plot(V1_list,P1_list,"k-",label=str1)
    ax.plot(V2_list,P2_list,"b-",label=str2)
    ax.plot(V1a_list,P1a_list,"k--", label="adiabatica da V1_1")
    ax.plot(V2a_list,P2a_list,"b--", label="adiabatica da V2_1")
    ax.set_xlabel("V (l)")
    ax.set_ylabel("P (atm)")
    ax.set_xlim(Vmin,Vmax)
    ax.set_ylim(Pmin,Pmax)
    v1=vol.V1_1+shx
    p1=P1_1+shy
    v2=vol.V2_1+shx
    p2=P2_1+shy
    v3=V1_2+shx
    p3=P1_2+shy
    v4=V2_2+shx
    p4=P2_2+shy
    ax.text(v1,p1,r'V1_1')
    ax.text(v2,p2,r'V2_1')
    ax.text(v3,p3,r'V1_2')
    ax.text(v4,p4,r'V2_2')
    ax.legend(frameon=False)
    plt.show()
    
    print("\n\nCoordinate del ciclo (V in litri, P in atmosfere, T in K)\n")
    print("V1_1: %5.2f, P1_1: %5.2f,  T1: %5.1f" % (vol.V1_1, P1_1, T1))
    print("V2_1: %5.2f, P2_1: %5.2f,  T1: %5.1f" % (vol.V2_1, P2_1, T1))
    print("V1_2: %5.2f, P1_2: %5.2f,  T2: %5.1f" % (V1_2, P1_2, T2))
    print("V2_2: %5.2f, P2_2: %5.2f,  T2: %5.1f" % (V2_2, P2_2, T2))
    
# -------- Fine sezione di plot -----
  
# Calcolo del rendimento del ciclo
#
# Poiche' la trasformazione complessiva e' appunto un ciclo
# la variazione dell'energia interna complessiva, partendo da
# vol.V1_1 e tornando allo stesso valore vol.V1_1 (percorrendo il ciclo
# in senso antiorario), deve essere 0: Delta_U=0
# Dal primo principio, allora, Delta_Q + Delta_L = 0 --> Delta_L = -Delta_Q
# 
# Il flusso di calore e' zero lungo le adiabatiche (per definizione di
# adiabatica).
#
# Lungo le isoterme, anche deve essere Delta_L = -Delta_Q perche' 
# per un gas ideale U dipende solo da T (e se T non varia, nemmeno U varia).
#
# Calcoliamo allora il flusso totale di calore lungo le due isoterme 
# calcolando il lavoro netto lungo le stesse due isoterme:
# L1 e' il lavoro fatto sul sistema (in compressione) lungo l'isoterma T1
# L2 e' il lavoro fatto dal sistema (in espansione) lungo l'isoterma T2
    L1=-1*R*T1*np.log(vol.V2_1/vol.V1_1)
    L2=-1*R*T2*np.log(V1_2/V2_2)

# I flussi di calore corrispondenti sono     
    Q1=-1*L1
    Q2=-1*L2
    
# Q1 + Q2 e' il flusso netto di calore che entra nel ciclo
# Ma Q1 + Q2 = -(L1+L2) = Delta_Q = -Delta_L = -(L1+L2+L3+L4) 
# dove L3 e L4 sono i lavori lungo le due adiabatiche;
# per cui L3+L4=0 --> L3=-L4: il lavoro lungo l'adiabatica da
# vol.V2_1 a V2_2 e' uguale e opposto al lavoro tra V1_2 a vol.V1_1
 
# Il flusso di calore entrante nel sistema e' Q2 
# La resa del ciclo è il lavoro netto (L1+L2) diviso il calore entrante Q2

    resa=100*abs((L1+L2))/Q2

# Conversioni e stampe
    fact=101325*1e-3/4.184
    Q1c=Q1*fact
    Q2c=Q2*fact
    L=(L1+L2)*fact*4.184
    print("\nCalore ceduto lungo l'isoterma T1: %5.2f cal" % Q1c)
    print("Calore assorbito lungo l'isoterma T2: %5.2f cal" % Q2c)
    print("Lavoro netto prodotto dal ciclo: %5.2f J" % L)
    print("\nResa del ciclo %4.2f %%" % resa)