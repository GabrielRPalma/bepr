"""Main module."""
from numpy import exp, log, arange, array
from scipy.integrate import odeint
import matplotlib.pyplot as plt



def model(st, t, P, mu, th, tv, tx, alfa, teta, beta, bh, bv): 
    
    # Parameters
    bs = 0.4
    bh = bh
    bv = bv
    k = 15000
    
    # Initial values
    S = st[0]
    H = st[1]
    V = st[2]
    
    dsdt = bs * S * (1 - (S + H  +V) / k) * exp((-alfa * P)/(beta*(S + H +V) + P))  \
    - mu * S \
    + (1 - th) * bh * H * (1 - (S + H  +V) / k) * exp((-alfa * P)/(beta*(S + H +V) + P)) \
    + (1- tv) * bv * V * (1 - (S + H  +V) / k) * exp((-alfa * P)/(beta*(S + H +V) + P)) 
    dhdt = th * bh * H * (1 - (S + H  +V) / k) * exp((-alfa * P)/(beta * (S + H + V) + P)) \
    - mu * H \
    + (1-tx) * tv * bv * V * (1 - (S + H  +V) / k) * exp((-alfa * P)/(beta * (S + H + V) + P))
    dvdt = tx * tv * bv * V * (1 - (S + H  +V) / k) * exp((-teta * P)/(beta * (S + H + V) + P)) \
    - mu * V

    return(dsdt, dhdt, dvdt)

def simulation(init, time, P, mu, th, tv, tx, alfa, teta, beta, bh, bv, doplot=True):
    
    t = arange(0,time)
    st = odeint(model, init, t, args= (P, mu, th, tv, tx, alfa, teta, beta, bh, bv))

    lor = st

    x = st[:, 0]
    y = st[:, 1]
    z = st[:, 2]
#    w = st[:, 3]
    if doplot:
        fig, ax = plt.subplots()    
        ax.plot(t, x, 'r-', linewidth=2, label='Hosts without H. defensa')
        ax.plot(t, y, 'b--', linewidth=2, label='Hosts with H. defensa without APSE')
        ax.plot(t, z, 'y', linewidth = 2, label = 'Hosts with H. defensa with APSE')
        ax.set(xlabel = "Time", ylabel = "number of individuous", title = "Host-simbiont-virus-Parasitoid" )
        ax.axvline(0,0)
        ax.axhline(0,0)
        ax.grid(True)
        ax.legend()
        
    return st

def bifurcation(initial,P_range, e, parms):

    args = parms
    solf = []
    
    for parametro in P_range:
        
        args[e] = parametro
        solf.append(simulation(initial, 5000, P = args[0], mu = args[1],th = args[2], tv = args[3], tx = args[4], 
                   alfa = - log(args[5]), teta = - log(args[6]), beta = args[7], bh= args[8], bv= args[9], doplot=False)[-1,:])
    
    solf = array(solf)
    
    return P_range, solf

# Visualization functions

def bifurcation_plot(P_range, solf, parametro, save, S_name):
        
        fig = plt.figure(figsize = (6,4),dpi=200)
        fig.patch.set_alpha(.5)
        
        ax = fig.add_subplot(111)
        ax.plot(P_range, solf[:, 0], ls = "-", label = 'Uninfected hosts')
        ax.set(xlabel="%s" %parametro, ylabel="Number of individuals")
        ax.axvline(0,0, c='k', ls=':')
        ax.axhline(0,0, c='k', ls=':')
        ax.grid(True, ls=':')
        ax.plot(P_range, solf[:, 1], ls = "-", label = 'Hosts infected with $H. defensa$')
        ax.plot(P_range, solf[:, 2], ls = "-", label = 'Hosts infected with $H. defensa$ and $APSE$')
        ax.legend(loc='best')
        
        if save:
            fig.savefig('%s' % S_name, dpi = 300)

def model_plot(y, save, S_name):
    
    fig = plt.figure(figsize = (6,4),dpi=200)
    fig.patch.set_alpha(.5)
    ax = fig.add_subplot(111)
    ax.plot(y[:, 0], ls = "-", label = 'Uninfected hosts')
    ax.set(xlabel="Time", ylabel="Number of individuals")
    ax.axvline(0,0, c='k', ls=':')
    ax.axhline(0,0, c='k', ls=':')
    ax.grid(True, ls=':')
    ax.plot(y[:, 1], ls = "-", label = 'Hosts infected with $H. defensa$')
    ax.plot(y[:, 2], ls = "-", label = 'Hosts infected with $H. defensa$ and $APSE$')
    ax.legend(loc='best')
    if save:
        fig.savefig('%s' % S_name, dpi = 300)

def get_resistence_loss_dynamcics(initial_values, p_future, count, my_parms, alpha=0.1 * 15000):
    args =  my_parms
    time = []
    
    p_equilibrium = simulation(initial_values, 5000, P = args[0], mu = args[1],th = args[2], tv = args[3], tx = args[4], alfa = - log(args[5]), teta = - log(args[6]), beta = args[7], bh= args[8], bv= args[9], doplot=False)[-1,]
    
    args[0] = p_future
    
    for i in count:
        time.append(i)
        
        dinamics = simulation(p_equilibrium, i, P = args[0], mu = args[1],th = args[2], tv = args[3], tx = args[4], alfa = - log(args[5]), teta = - log(args[6]), beta = args[7], bh= args[8], bv= args[9], doplot=False)
        
        if dinamics[:,2][-1] <= alpha:
            break
        else:
            continue
    print('The model spends %s iterations for the protected hosts be extinct'% len(time))        
    return(dinamics, time)

def get_resistence_dynamcics(initial_values, p_future, count, my_parms, alpha=5):
    args =  my_parms
    time = []
    
    p_equilibrium = simulation(initial_values, 5000, P = args[0], mu = args[1],th = args[2], tv = args[3], tx = args[4], alfa = - log(args[5]), teta = - log(args[6]), beta = args[7], bh= args[8], bv= args[9], doplot=False)[-1,]
    
    args[0] = p_future
    p_equilibrium[2] = p_equilibrium[0] * .1
    
    for i in count:
        time.append(i)
        
        dinamics = simulation(p_equilibrium, i, P = args[0], mu = args[1],th = args[2], tv = args[3], tx = args[4], alfa = - log(args[5]), teta = - log(args[6]), beta = args[7], bh= args[8], bv= args[9], doplot=False)
        
        if dinamics[:,2][-1] >= alpha:
            break
        else:
            continue
    print('The model spends %s iterations for the protected hosts population expresive at the community'% len(time))        
    return(dinamics, time)