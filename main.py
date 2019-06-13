import numpy as np
import psi4
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline


### template for the z-matrix
mol_tmpl = """H
F 1 **R**"""

### We will probably forget about the molecules array, so let's have it pre-declared!
molecules = []
r_array = [0.5, 0.55, 0.57, 0.6, 0.65, 0.67, 0.7, 0.75, 0.77, 0.8, 0.85, 0.87, 0.9, 0.95, 0.97, 1.0, 1.5, 1.7, 2.0, 2.1, 2.2, 2.25]

RHF_E_array = []
MP2_E_array = []
CCSD_E_array = []

### loop over the different bond-lengths, create different instances
### of HF molecule
for r in r_array:
    molecule = psi4.geometry(mol_tmpl.replace("**R**", str(r)))
    molecules.append(molecule)
    


### loop over instances of molecules, compute the RHF, MP2, and CCSD
### energies and store them in their respective arrays
for mol in molecules:
    #print(mol.x)
    energy = psi4.energy("SCF/cc-pVTZ", molecule=mol)
    RHF_E_array.append(energy)
    energy = psi4.energy("MP2/cc-pVTZ", molecule=mol)
    MP2_E_array.append(energy)
    energy = psi4.energy("CCSD/cc-pVTZ",molecule=mol)
    CCSD_E_array.append(energy)


r_array_au = []
for i in r_array:
    r_array_au.append(i*1.89)
    
print(r_array_au)

print(RHF_E_array)
plt.plot(r_array_au, RHF_E_array, '-r*', label='RHF')
plt.plot(r_array_au, MP2_E_array, '-g*', label='MP2')
plt.plot(r_array_au, CCSD_E_array, '-b*', label='CCSD')
plt.legend()
plt.show()

RHF_E_Spline = InterpolatedUnivariateSpline(r_array_au, RHF_E_array, k=3)
MP2_E_Spline = InterpolatedUnivariateSpline(r_array_au, MP2_E_array, k=3)
CCSD_E_Spline = InterpolatedUnivariateSpline(r_array_au, CCSD_E_array, k=3)

print(MP2_E_Spline(3.3))

### form a much finer grid to evaluate spline object at
r_fine = np.linspace(0.5*1.89,2.25*1.89,200)

### compute the interpolated/extrapolated values for RHF Energy on this grid
RHF_E_fine = RHF_E_Spline(r_fine)

### compute the interpolated/extrapolated values for RHF Energy on this grid
MP2_E_fine = MP2_E_Spline(r_fine)

### compute the interpolated/extrapolated values for RHF Energy on this grid
CCSD_E_fine = CCSD_E_Spline(r_fine)


### plot the interpolated data with lines against computed data in *'s
plt.plot(r_fine, RHF_E_fine, 'red', r_array_au, RHF_E_array, 'r*', label='RHF')
plt.plot(r_fine, MP2_E_fine, 'green', r_array_au, MP2_E_array, 'g*', label='MP2')
plt.plot(r_fine, CCSD_E_fine, 'blue', r_array_au, CCSD_E_array, 'b*', label='CCSD')
plt.legend()
plt.show()

RHF_Force_Spline = RHF_E_Spline.derivative()
MP2_Force_Spline = MP2_E_Spline.derivative()
CCSD_Force_Spline = CCSD_E_Spline.derivative()

RHF_Force_fine = -RHF_Force_Spline(r_fine)
MP2_Force_fine = -MP2_Force_Spline(r_fine)
CCSD_Force_fine = -CCSD_Force_Spline(r_fine)

plt.plot(r_fine, RHF_Force_fine, 'red', label='RHF Force')
plt.plot(r_fine, MP2_Force_fine, 'green', label='MP2 Force')
plt.plot(r_fine, CCSD_Force_fine, 'blue', label='CCSD Force')
plt.legend()
plt.show()

### Find Equilibrium Bond-Lengths for each level of theory
RHF_Req_idx = np.argmin(RHF_E_fine)
MP2_Req_idx = np.argmin(MP2_E_fine)
CCSD_Req_idx = np.argmin(CCSD_E_fine)

### find the value of the separation corresponding to that index
RHF_Req = r_fine[RHF_Req_idx]
MP2_Req = r_fine[MP2_Req_idx]
CCSD_Req = r_fine[CCSD_Req_idx]

### print equilibrium bond-lengths at each level of theory!
print(" Equilibrium bond length at RHF/cc-pVDZ level is ",RHF_Req, "atomic units")
print(" Equilibrium bond length at MP2/cc-pVDZ level is ",MP2_Req, "atomic units")
print(" Equilibrium bond length at CCSD/cc-pVDZ level is ",CCSD_Req, "atomic units")

print("RHF Req is ", RHF_Req/1.89, " Angstroms")
print("MP2 Req is ", MP2_Req/1.89, " Angstroms")
print("CCSD Req is ", CCSD_Req/1.89, " Angstroms")

#mass of hydrogen in atomic units
mH = 1836.

#mass of fluorine in atomic units
mF = 34883.

#reduced mass of HF
mu = mH*mF / (mH+mF)
print("Reduced mass is ", mu, "atomic units")

