from pyscf import gto, scf, fci

mol = gto.Mole()


H2 = [["H", (0.00, 0.00, 0.00)], ["H", (0.00, 0.00, 0.74)]]
H4 = [["H", (0.00, 0.00, 0.00 + i * 1.0)] for i in range(4)]
H6 = [["H", (0.00, 0.00, 0.00 + i * 1.0)] for i in range(6)]
H8 = [["H", (0.00, 0.00, 0.00 + i * 1.0)] for i in range(8)]

mol.atom = H2

mol.basis = 'sto-3g'
mol.spin = 0
mol.charge = 0
mol.build()

mf = scf.RHF(mol).run()
e_hf = mf.e_tot

e_fci, fcivec = fci.FCI(mf).kernel()

print("=================================================")
print(f"{mol.atom}: {mol.basis})")
print(f"Hartree-Fock energy (E_HF): {e_hf:.10f} Hartree")
print(f"FCI energy (E_FCI): {e_fci:.10f} Hartree")
print("=================================================")

