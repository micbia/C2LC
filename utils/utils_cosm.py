import numpy as np
import astropy.units as u
import astropy.constants as cst
from astropy.cosmology import FlatLambdaCDM

from scipy.interpolate import interp1d
from tqdm import tqdm


def z2nu(z):
    nu0 = (1.42 * u.GHz).to(u.MHz)
    return nu0 / (1+z)


def get_z_output(zmin, zmax, fdepth_MHz):
    nu0, fdepth_MHz = 1420., fdepth_MHz.to(u.MHz).value
    zi, zi1 = zmin, 0
    output_z = [zi]
    while zi1 <= zmax:
        zi1 = (zi + fdepth_MHz/nu0 * (1+zi)) / (1-fdepth_MHz/nu0*(1+zi))
        output_z.append(zi1)
        zi = zi1
    return np.array(output_z)

def MHI_Modi2019(Mh, z, cosmo, model='A'):
    # from Modi+ 2019
    if(model == 'A'):
        alpha = (1+2*z)/(2+2*z)
        Mcut = (3e9*(1+10*(3/(1+z))**8) * u.Msun / cosmo.h).value # in Msun units
        Ah = 8e5*(1+(3.5/z)**6)*(1+z)**3 * u.Msun / cosmo.h
    elif(model == 'C'):
        alpha = 0.9
        Mcut = 1.e10 / cosmo.h # in Msun units
        Ah = 3.e6*(1.+1./z)*(1+z)**3 * u.Msun / cosmo.h
    else:
        ValueError(' Model B is not implemented')
    M_HI = Ah * np.power(Mh/Mcut, alpha) * np.exp(-Mcut/Mh)
    return M_HI


class ExtendCosmology(FlatLambdaCDM):
    def __init__(self, cosmo):
        super(ExtendCosmology, self).__init__(H0=cosmo.H0, Om0=cosmo.Om0, Ob0=cosmo.Ob0, Tcmb0=cosmo.Tcmb0, Neff=cosmo.Neff, m_nu=cosmo.m_nu)
        self.cosmo = cosmo
        
        # primordial nucleosynthesis quantities
        self.abu_he = 0.074
        self.abu_h = 1.0-self.abu_he
        self.abu_he_mass = 0.2486 
        self.abu_h_mass = 1.0-self.abu_he_mass
        self.mean_molecular = 1.0/(1.0-self.abu_he_mass)
        
        # 21cm astrophysics quantities
        self.nu0 = 1.42 * u.GHz
        self.lamb0 = (cst.c/self.nu0).cgs
        self.nH0 = ((1-self.abu_he_mass)*cosmo.Ob0*cosmo.critical_density0/cst.m_p).cgs

        # reionisation quantities
        A10, Tstar = 2.85e-15/u.s, (cst.h*self.nu0/cst.k_B).to(u.K)
        self.meanT = (3*self.lamb0**3*A10*Tstar/(32.*np.pi) * self.nH0 / cosmo.H0).to(u.mK)

        # internal use
        self.table_z = np.hstack((np.arange(0,1,0.02), 10**np.linspace(0,2,200)))
        self.table_cdist = self.cosmo.comoving_distance(self.table_z).value
        
    def z2nu(self, z):
        return (self.nu0 / (1+z)).to(u.MHz)

    def MHI_Modi2019(self, Mh, z, model='A'):
        return MHI_Modi2019(Mh=Mh, z=z, cosmo=self.cosmo, model=model)

    def MHI_Padmanabhan2017(self, Mh, z, delta_c):
        ''' Mh must be in Msun units (without small h)'''
        fHc = (1-self.abu_he_mass) * self.cosmo.Ob0 / self.cosmo.Om0 
        
        # Barnes+ (2014), Eq 3 : https://arxiv.org/abs/1403.1873v3
        #delta_c = 180.
        vM_c0 = 96.6 * np.power(delta_c*self.cosmo.Om0*self.cosmo.h*self.cosmo.h / 24.4, 1./6) * np.sqrt((1+z)/3.3) * np.power(Mh/1e11, 1./3) # * u.km / u.s
                
        #from Padmanabhan+ (2017), Eq. 1 and Table 3 : https://arxiv.org/abs/1611.06235
        alpha, cHI, v_c0, beta, gamma = 0.9, 28.65, np.power(10, 1.56), -0.58, 1.45
        M_HI = alpha * fHc * Mh * np.power(Mh/1e11*self.cosmo.h, beta) * np.exp(-1*np.power(v_c0 / vM_c0, 3))
        return M_HI.astype(np.float32)

    def dTb(self, xHI, z, Ts=None):
        ''' Here we assume that in the data array the peculliar velocity is included '''
        if(xHI.shape != np.shape(z) and type(z) != float):
            z = z[..., np.newaxis, np.newaxis]

        if(Ts == None):
            dT = self.meanT * xHI / self.cosmo.efunc(z)
        else:
            dT = self.meanT * xHI / self.cosmo.efunc(z) * (1-self.cosmo.Tcmb0/Ts)
        return dT

    def cdist2deg(self, cMpc, z):
        val = cMpc / self.cosmo.comoving_transverse_distance(z)
        return np.rad2deg(val * u.rad)

    def cdist2z(self, dist):
        ''' Calculate the redshift correspoding to the given comoving distance for 0 <= z < 100 '''
        dist = np.atleast_1d(dist)
        func = interp1d(self.table_cdist, self.table_z, kind='cubic', bounds_error=False, fill_value="extrapolate")
        return func(dist)