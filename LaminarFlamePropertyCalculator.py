# -*- coding: utf-8 -*-
"""
AE471 Calculator Script

For input conditions, this returns:
    Flame speed
    Flame thickness
    Effective Lewis Number of Mixture
    Mixture Viscosity
    Temperature Sensitivity of Mixture Viscosity
    
    

1D Premixed Flame Simulation

Mechanism used is the UCSD Ethylene/Acetylene mechanism with 52 species, 367 reactions
Reference report is included as a PDF, downloaded from: 
    https://gitlab.multiscale.utah.edu/common/ChemicalMechanisms/-/tree/master/ethylene-usc



requirements:
    conda install --channel conda-forge cantera
        if that shits the bed, then install through pip
    pip install cantera
        note that the pip one isn't multiprocessing compiled 

20260305
Carl Hall
carl.hall.ctr@afacademy.af.edu
carl@LEAero.us
"""
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct


###############################################################################
###############################################################################
####
####    Inputs
####
###############################################################################
###############################################################################

# Simulation parameters
p        = 0.77055*ct.one_atm  # pressure [Pa]
fuel     = 'C2H4:1'   # 2375K at phi 1
oxidizer = 'O2:0.21, N2:0.79'

# arrays for parametric study
Tin = [294.8]  # K
phi = [1.531]







###############################################################################
###############################################################################
####
####    Start of Processing Code
####
###############################################################################
###############################################################################

# Solver stuff
#loglevel = 1        # display the intermediate solver convergence results
loglevel = 0  # default (no diag output)
delta_Tin = 0.01  # deg K/C for finite differencing to get E_a and dnudT

# this seems to be the mech that converges well with published good agreement with flamespeed
gas = ct.Solution('ethylene-usc.yaml')

# storage arrays
flamespeed_cm_s = []
flamethickness_mm = []

print(f'\nPressure: {p:.0f} Pa, {p*760/101325:.1f} torr, {p/101325:.3f} atm')
print('\nSimulations performed with therm/tran from 1999 USC report by Hai Wang and Alexander Laskin titled')
print('    A Comprehensive Kinetic Model of Ethylene and Acetylene Oxidation at High Temperatures')


# solver loop
for _phi, _Tin in zip(phi,Tin):
    print(f'\nSolving case: phi {_phi:0.2f}, T {_Tin:0.2f}')
    
    gas.TP = _Tin, p
    gas.set_equivalence_ratio(phi=_phi, fuel=fuel, oxidizer=oxidizer)
    rho_u1 = gas.density_mass                    # (Mass) density [kg/m^3]
    f1 = ct.FreeFlame(gas, width=0.05)                                       # Initial domain size, adaptively expanded by solver (initial value strongly affects convergence) [m]
    f1.set_refine_criteria(ratio=3, slope=0.06, curve=0.10)                  # fine refinement
    #f1.set_refine_criteria(ratio=10.0, slope=0.8, curve=0.8, prune=0.05)     # coarse refinement
    #f1.transport_model = 'mixture-averaged'
    #f1.flux_gradient_basis = "mass"                                          # only relevant for mixture-averaged model
    f1.transport_model = 'multicomponent'
    f1.soret_enabled = True
    f1.solve(loglevel=loglevel, auto=True)
    if loglevel>0:
        f1.show()                                                            # print solution to stdout
    flamespeed_cm_s.append(f1.velocity[0]*100.)
    
    # calculate flame thickness based on peak gradient and overall temperature rise
    dTdx_max = np.max(np.diff(f1.T)/np.diff(f1.grid))
    del_x = (np.max(f1.T)-np.min(f1.T))/dTdx_max
    flamethickness_mm.append(del_x*1000)   # mm

    print(f"  Flamespeed        = {f1.velocity[0]*100:.1f} cm/s")
    print(f"  Flamethickness    = {del_x*1000:.2f} mm")
    
    # run simulation again, to calculate activation energy via finite differencing
    gas.TP = _Tin+delta_Tin, p
    gas.set_equivalence_ratio(phi=_phi, fuel=fuel, oxidizer=oxidizer)
    rho_u2 = gas.density_mass                    # (Mass) density [kg/m^3]
    f2 = ct.FreeFlame(gas, width=0.05)                                       # Initial domain size, adaptively expanded by solver (initial value strongly affects convergence) [m]
    f2.set_refine_criteria(ratio=3, slope=0.06, curve=0.10)                  # fine refinement
    #f2.set_refine_criteria(ratio=10.0, slope=0.8, curve=0.8, prune=0.05)     # coarse refinement
    #f2.transport_model = 'mixture-averaged'
    #f2.flux_gradient_basis = "mass"                                          # only relevant for mixture-averaged model
    f2.transport_model = 'multicomponent'
    f2.soret_enabled = True
    f2.solve(loglevel=loglevel, auto=True)
    if loglevel>0:
        f2.show()                                                            # print solution to stdout
    
    R     = 8.31446261815324/1000.              # kJ/K/mol
    rho_u = [rho_u1, rho_u2]                    # (Mass) density [kg/m^3]
    vel_u = [f1.velocity[0], f2.velocity[0]]    # Unburned gas velocity [m/s]
    T_b   = [f1.T[-1], f2.T[-1]]                # Burned gas temperature [K]
    E_a   = -2*R * ( np.log(rho_u[1]*vel_u[1]) - np.log(rho_u[0]*vel_u[0]) )/\
                   ( 1./T_b[1]                 - 1./T_b[0]                 )     # F.N. EGOLFOPOULOS and C.K. LAW, Chain Mechanisms in the Overall Reaction Orders in Laminar Flame Propagation, COMBUSTION AND FLAME 80: 7-16 (1990)
    #print(f'  Activation Energy = {E_a:.2f} kJ/mol') 
    
    # calculate Zeldovich number
    Z = E_a * (T_b[0] - _Tin)/(R*T_b[0]*T_b[0])
    #print(f'  Zeldovich Number  = {Z:.2f}') 
    
    # calculate Lewis numbers
    gas.TP = _Tin, p
    gas.set_equivalence_ratio(phi=_phi, fuel=fuel, oxidizer=oxidizer)
    D_ox = gas.mix_diff_coeffs_mass[gas.species_index('O2')]            # Mixture-averaged diffusion coefficients [m^s/s]
    D_fu = gas.mix_diff_coeffs_mass[gas.species_index('C2H4')]          # Mixture-averaged diffusion coefficients [m^s/s]
    lam = gas.thermal_conductivity                                      # Thermal conductivity. [W/m/K]
    rho = gas.density_mass                                              # (Mass) density [kg/m^3].
    cp  = gas.cp_mass                                                   # Specific heat capacity at constant pressure and composition [J/kg/K].
    alpha = lam / (rho * cp)                                            # thermal diffusivity [m^2/s]
    Le_fu = alpha / D_fu                                                # mass diffusivity of C2H4 (for lean cases)
    Le_ox = alpha / D_ox                                                # mass diffusivity of O2 (for rich cases )

    # calculate effective Le, using methodology in Matalon/Bechtold 2001
    if _phi>1:
        A = 1 + Z*(_phi-1)
        Le_eff = 1 + ((Le_fu-1) + (Le_ox-1)*A)/(1+A)
    else:
        A = 1 + Z*(1/_phi-1)
        Le_eff = 1 + ((Le_ox-1) + (Le_fu-1)*A)/(1+A)
    
    #print(f'  Fuel Le           = {Le_fu:.3f}') 
    #print(f'  Oxidizer Le       = {Le_ox:.3f}') 
    print(f'  Effective Le      = {Le_eff:.3f}') 
    
    # calculate viscosity and the sensitivity with respect to temperature
    gas.TP = _Tin, p
    gas.set_equivalence_ratio(phi=_phi, fuel=fuel, oxidizer=oxidizer)
    nu = gas.viscosity                                                  # Viscosity [Pa-s]
    print(f'  Mixture Viscosity = {nu:.3E} kg/m/s') 
    
    gas.TP = _Tin+delta_Tin, p
    gas.set_equivalence_ratio(phi=_phi, fuel=fuel, oxidizer=oxidizer)
    _nu = gas.viscosity                                                  # Viscosity [Pa-s]
    dnudT = (nu-_nu)/delta_Tin
    print(f'  Mixture Viscosity Sensitivity to Temperature = {dnudT:.3E} (kg/m/s)/(K deg)') 
    
    

    # plot solution components
    solution = f1.to_array()

    # Find the region that covers the heat release
    zlow, zhigh = solution.grid[ np.where(solution.heat_release_rate > 0.01*np.max(solution.heat_release_rate))[0][[0,-1]] ]   # 1% and 99% locations of HRR
    width = zhigh - zlow
    zlow  -= 3*width
    zhigh += 3*width


    fig,ax = plt.subplots(2,2,figsize=(8,8))
    ax[0,0].plot(solution.grid*1000,solution.T)
    ax[0,0].set_ylabel('Temperature [K]')
    ax[0,0].set_xlabel('flame coordinate [mm]')
    ax[0,0].set_xlim([zlow*1000, zhigh*1000])
    ax[0,0].grid()

    ax[0,1].plot(solution.grid*1000,solution.heat_release_rate)
    ax[0,1].set_ylabel('Heat Release Rate [W/m$^3$]')
    ax[0,1].set_xlabel('flame coordinate [mm]')
    ax[0,1].set_xlim([zlow*1000, zhigh*1000])
    ax[0,1].grid()
    
    ax[1,0].plot(solution.grid*1000,solution('C2H4').X,label='C$_2$H$_4$')
    ax[1,0].plot(solution.grid*1000,solution('O2').X,label='O$_2$')
    ax[1,0].plot(solution.grid*1000,solution('CO2').X,label='CO$_2$')
    ax[1,0].plot(solution.grid*1000,solution('H2O').X,label='H$_2$O')
    ax[1,0].set_ylabel('Molefraction [-]')
    ax[1,0].set_xlabel('flame coordinate [mm]')
    ax[1,0].set_xlim([zlow*1000, zhigh*1000])
    ax[1,0].grid()
    ax[1,0].legend()
    
    ax[1,1].plot(solution.grid*1000,solution('OH').X)
    ax[1,1].set_ylabel('OH Molefraction [-]')
    ax[1,1].set_xlabel('flame coordinate [mm]')
    ax[1,1].set_xlim([zlow*1000, zhigh*1000])
    ax[1,1].grid()

    fig.tight_layout()

    #fig.savefig(f'flamesim_ethylene-air_phi-{_phi:.2f}_T-{_Tin:.2f}.png',dpi=300)
    
    #plt.close(fig)



