import numpy as np
import numpy.polynomial.polynomial as poly
# from MW import mw_generic as gene
# import matplotlib.pylab as plt
import lmfit

path_to_data = 'C:\\Users\\gusarov\\CPB_USC_project\\data\\'
path_to_arrays = 'C:\\Users\\gusarov\\CPB_USC_project\\data_arrays\\'

FileName = '2024_08_21_CPB_c3r6_RO1_Coil_8.hdf5'
DataType = 'Coil_VNA'

# FileName = '2024_08_21_CPB_c3r6_RO1_Power_4.hdf5' # -56 uA, single resonance
# DataType = 'VNA_Power'

# FileName = '2024_08_21_CPB_c3r6_RO1_Power_10.hdf5' # -45 uA, two resonances
# DataType = 'VNA_Power'


# FileName = "2024_08_21_CPB_c3r6_RO1_Power_11.hdf5" # -43 uA, single resonance
# DataType = 'VNA_Power'

ExpName = FileName[:-5]


def LoadDataLabber(FileName, DataType, verbose=False):
    ''' Load Labber file. Returns x, y, z
    Datatypes: 'xdata_ydata'
        * 'VNA_Freq': 1D data
            * frequency along x
        * 'Coil_VNA': 
            * coil parameter along x (voltage)
            * VNA freq along y
        * 'VNA_Power':
            * VNA frequency along x
            * VNA power along y
    '''
    import Labber
    
    file = Labber.LogFile(FileName)

    if verbose==True:
            print(f'Opening file:\n   {FileName}')
            print(file.getEntry())

    if DataType == 'Coil_VNA':
        S21 = file.getData()
        V_flux = file.getStepChannels()[0]['values']
        freq = file.getTraceXY()[0]
        return V_flux, freq, S21, file
    
    elif DataType == 'VNA_Power':
        S21 = file.getData()
        VNApower = file.getStepChannels()[0]['values']
        freq = file.getTraceXY()[0]
        return freq, VNApower, np.transpose(S21), file
    
    elif DataType == 'VNA_Freq':
        S21 = file.getData()
        freq = file.getTraceXY()[0]
        return freq, S21, file




## conventions for S21 taken from Chris Axline thesis 
## --> works in linear regime = low power
def S21_transm(w, w0, ktot, kin, kout):
    """Transmitted signal S21 lambda/2 (Chrix Axline thesis convention).
    * w0 is center freq*2pi
    * k = dissipation rates (k=omega/Q). Total, internal, and output.
    """
    return np.sqrt(kin*kout)/(1j*(w-w0) - 0.5*ktot)

def S21_refl(w, w0, ktot, kc):
    """ Reflected signal in reflection lambda/4 (Chrix Axline thesis convention)
    * w0 is center freq*2pi
    * k = dissipation rates (k=omega/Q). Total, internal, and output.
    """
    return ( 1j*(w-w0) + kc - ktot*0.5 ) / ( 1j*(w-w0) - ktot*0.5 )


def S21_hang_v2(w, w0, ktot, kc, phi):
    """ Transmission signal in notch/hanged configuration (Millimeter four-wave / Simo convention)
    * w0 is center freq*2pi
    * k = dissipation rates (k=omega/Q). Total, internal, and output.
    * phi accounts for impedance mismatch. Use exp(iphi)/cos(phi)
    """
    #return S21
    return  1 - ( kc*np.exp(1j*phi) ) / ( (ktot + 2j*(w-w0))*(np.cos(phi)) ) #Simo version

def S21_linear_general(w, w_r, k_tot, k_ext, amp, alpha, phi, config):
    '''Parameters:
        * w0: center frequency
        * k_tot: total dissipation rate
        * k_ext: external dissipation rate
        * amp: amplitude of signal
        * alpha: global phase offset
        * phi: impedance mismatch
        * config: 'notch', 'reflection' or 'transmission'
    '''
    k_in = k_tot - k_ext
    if config=='notch' or config==0:
        return amp*np.exp(-1j*alpha)*S21_hang_v2(w, w_r, k_tot, k_ext, phi)
    elif config=='reflection' or config==1:
        return amp*np.exp(-1j*alpha)*S21_refl(w, w_r, k_tot, k_ext)
    elif config=='transmission' or config==2:
        return amp*np.exp(-1j*alpha)*S21_transm(w, w_r, k_tot, k_in, k_ext)



def fit_res_linear_lmfit(freq, data, config, guess=None, fmin=None, fmax=None):
    """Fit linear resonator S21.
    Parameters:
        * freq: Hz
        * data: S21
        * config: 'notch', 'reflection' or 'transmission'
        * guess: w_r, k_tot, k_ext, amp, alpha, phi 
    """

    ## restrict freq range
    if fmin==None and fmax==None:
        ff = freq
        zz = data
    # else:
    #     ff, zz = gene.getf(freq, data, fmin, fmax)

    if config=='notch':
        config_int = 0
    elif config=='reflection':
        config_int = 1
    elif config=='transmission':
        config_int = 2

    ## define lmfit model
    ResModel = lmfit.Model(S21_linear_general, independent_vars=['w'])

    ## guess parameters
    if guess==None:
        w_r = np.mean(ff)*2*np.pi
        k_tot = 1e6*2*np.pi
        k_ext = k_tot/2
        amp = np.mean(abs(zz))
        alpha = 0
        phi = 0

    else:
        w_r, k_tot, k_ext, amp, alpha, phi = guess


    params = ResModel.make_params(
        w_r=dict(value=w_r, vary=True),
        k_ext=dict(value=k_ext, min=0.01*k_ext, max=100*k_ext, vary=True),
        k_tot=dict(value=k_tot, min=0.01*k_tot, max=100*k_tot, vary=True),
        amp=dict(value=amp, min=0.001, max=100, vary=True),
        alpha=dict(value=alpha, min=-np.pi, max=np.pi, vary=True),
        phi=dict(value=phi, min=-np.pi, max=np.pi, vary=True),
        config=dict(value=config_int, vary=False)
        )

    # params.pretty_print()

    result = ResModel.fit(data=zz, params=params,  w=2*np.pi*ff)

    return result

#