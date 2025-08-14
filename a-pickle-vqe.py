import time
from functools import partial
import numpy as np
import pyscf
import pyscf.mcscf
import ffsim
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP, L_BFGS_B
from qiskit.primitives import Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_algorithms import VQE
from qiskit.circuit.library import TwoLocal, ExcitationPreserving, EfficientSU2
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian, solve_sci_batch
import pickle
from pathlib import Path

num_atoms = 6
atoms = [["H", (0.00, 0.00, 0.00 + i * 1.0)] for i in range(num_atoms)]
mol = pyscf.gto.Mole()
mol.build(
    atom=atoms,
    basis="sto-3g",
)
scf = pyscf.scf.RHF(mol).run()

atoms = "; ".join([f"{symbol}  {coords[0]:.2f}  {coords[1]:.2f}  {coords[2]:.2f}" for symbol, coords in atoms])
driver = PySCFDriver(
    atom=atoms,
    basis="sto-3g",
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
)
n_frozen = 0
active_space = range(n_frozen, mol.nao_nr())
num_orbitals = len(active_space)
n_electrons = int(sum(scf.mo_occ[active_space]))
num_elec_a = (n_electrons + mol.spin) // 2
num_elec_b = (n_electrons - mol.spin) // 2
cas = pyscf.mcscf.CASCI(scf, num_orbitals, (num_elec_a, num_elec_b))
mo = cas.sort_mo(active_space, base=0)
hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), num_orbitals)
nelec = (num_elec_a, num_elec_b)

es_problem = driver.run()
mapper = JordanWignerMapper()
uccsd_ansatz = UCCSD(
    es_problem.num_spatial_orbitals,
    es_problem.num_particles,
    mapper,
    initial_state=HartreeFock(
        es_problem.num_spatial_orbitals,
        es_problem.num_particles,
        mapper,
    ),
)
uccsd_initial_point = [0.0] * (uccsd_ansatz.num_parameters)

# TwoLocal
tl_ansatz = TwoLocal(
    rotation_blocks=["h", "rx"],
    entanglement_blocks="cz",
    entanglement="full",
    reps=2,
    parameter_prefix="y",
)
tl_initial_point = [0.0] * (2*tl_ansatz.num_parameters)

# ExcitationPreserving
ep_ansatz = ExcitationPreserving(
    num_qubits=es_problem.num_spatial_orbitals,
)
ep_initial_point = [0.0] * (2*ep_ansatz.num_parameters)

# EfficientSU2
effsu2_ansatz = EfficientSU2(
    num_qubits=es_problem.num_spatial_orbitals,
    entanglement="full",
)
effsu2_initial_point = [0.01] * (effsu2_ansatz.num_parameters*2)

ansatz = effsu2_ansatz
initial_point = effsu2_initial_point
optimizer = L_BFGS_B()
backend = AerSimulator()
estimator = Estimator()
sampler = Sampler(mode=backend)


evaluation_count=[]
parameters_vars=[]
estimated_value=[]
meta_dict=[]
def vqe_callback(counts, parameters, value, metadata):
    evaluation_count.append(counts)
    parameters_vars.append(parameters)
    estimated_value.append(value)
    meta_dict.append(metadata)

    # Save VQE info
    vqe_info = (ansatz, evaluation_count, parameters_vars, estimated_value, meta_dict)
    Path(f"vqe_info_H{num_atoms}.pickle").write_bytes(pickle.dumps(vqe_info))

    # print(f"iter: {counts:4d}, energy: {value:.5f}, parameters: {parameters}")
    print(f"iter: {counts:4d}, energy: {value:.5f}, total_energy: {value+nuclear_repulsion_energy:.5f}")

solver = VQE(estimator, ansatz, optimizer, callback=vqe_callback)
solver.initial_point = initial_point

calc = GroundStateEigensolver(mapper, solver)
res = calc.solve(es_problem)
print('total_energy: ', res.total_energies[0])
