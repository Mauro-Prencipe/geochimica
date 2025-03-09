# Solar spectrum

# Determination of the temperature of photosphere

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import curve_fit

class Solar:
    
    wmin=500.
    temp_guess=5000.
    ampl_guess=5.e-5
    name='solar_spectrum.dat'
    plot_sampling=500
    plot_wmin=150.
    plot_wmax=2500.
    plot_npoint=200
    
    low_limit=5.
    
    @classmethod
    def set_wmin(cls, wmin):
        cls.wmin=wmin
   
    @classmethod
    def load(cls, name=''):
        
        real_name=cls.name
        
        if name != '':
           real_name=name
           
        data=np.loadtxt(real_name)
        cls.wave=data[:,0]
        cls.inten=data[:,1]
    
    @classmethod
    def planck(cls, wave, temp, amplitude):
        a=3.7469543e+20
        b=14397733.98663405
        c=299000000.
        f1 = np.exp(b/(wave*temp)) - 1. 
        denom=f1*wave**5
        plk=a*amplitude/denom
        
        return plk
    
    @classmethod
    def select(cls):
        test=cls.wave >= cls.wmin
        wsel=np.where(test)
        cls.w_selected=cls.wave[wsel]
        cls.int_selected=cls.inten[wsel]

    @classmethod
    def fitting(cls):
        cls.select()
        fit, cov=curve_fit(cls.planck, cls.w_selected, cls.int_selected,\
                                   p0=(cls.temp_guess, cls.ampl_guess))
    
        err=np.diag(cov)
        err=np.sqrt(err[0])
        
        cls.temp=fit[0]
        cls.amplitude=fit[1]
        cls.res=cls.make_integral()

        print("\nTotal number of points:    %6i" % cls.wave.size)
        print("Number of selected points: %6i" % cls.w_selected.size )

        print("\nTemperature: %6.1f (%2.1f) K" % ( cls.temp, err))
        print("Amplitude:   %6.3e W/nm m^2" % cls.amplitude)
        print("Solar flux: %6.1f W/m^2" % cls.res[0])
    
        cls.make_plot()
    
    @classmethod
    def make_plot(cls):
        wave_plot=cls.wave[::cls.plot_sampling]
        inten_plot=cls.inten[::cls.plot_sampling]
        wave_calc=np.linspace(cls.plot_wmin, cls.plot_wmax, cls.plot_npoint)
    
        inten_calc=np.array([])

        for iw in wave_calc:
            jinten=cls.planck(iw, cls.temp, cls.amplitude)
            inten_calc=np.append(inten_calc, jinten)
            
        max_inten_calc=np.max(inten_calc)
        max_inten_plot=np.max(inten_plot)
        
        max_inten=1.1*np.max([max_inten_calc, max_inten_plot])
        max_inten=max_inten.round(1)

        plt.figure(figsize=(4,3))
        plt.plot(wave_plot, inten_plot, "b*", markersize=2, label="Satellite data")
        plt.plot(wave_calc, inten_calc, "k-", label="Black body law")
        plt.ylim(0., max_inten)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Solar flux (W/nm m^2)")
        plt.legend(frameon=False)
        plt.title("Solar flux")
        plt.show()

    @classmethod
    def make_integral(cls):
        res=integrate.quad(cls.planck, cls.low_limit, np.inf, args=(cls.temp, cls.amplitude))
        return res


def start(wmin=550., name=''):
    Solar.set_wmin(wmin)
    Solar.load(name)
    Solar.fitting()
    
    
start()
    
