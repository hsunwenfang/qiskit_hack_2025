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
from qiskit.circuit.library import TwoLocal
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian, solve_sci_batch
import pickle
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

backend = AerSimulator()
estimator = Estimator()
sampler = Sampler(mode=backend)

# Load VQE info
for vqe_iter in list(range(100, 1000, 100))+list(range(1000, 10000, 1000)):
    vqe_info = pickle.loads(Path(f"vqe_info_H{num_atoms}.pickle").read_bytes())
    (ansatz, evaluation_count, parameters_vars, estimated_value, meta_dict) = vqe_info

    if vqe_iter > len(evaluation_count):
        break

    evaluation_count = evaluation_count[:vqe_iter]
    parameters_vars = parameters_vars[:vqe_iter]
    estimated_value = estimated_value[:vqe_iter]
    meta_dict = meta_dict[:vqe_iter]

    ansatz.measure_all()

    pass_manager = generate_preset_pass_manager(
        optimization_level=3, backend=backend, #initial_layout=initial_layout
    )
    pass_manager.pre_init = ffsim.qiskit.PRE_INIT
    isa_circuit = pass_manager.run(ansatz)
    job = sampler.run([(isa_circuit, parameters_vars[-1])], shots=10_000)
    primitive_result = job.result()
    # print('primitive result:', primitive_result)
    pub_result = primitive_result[0]
    # print(pub_result)
    bit_array = pub_result.data.meas
    counts = pub_result.data.meas.get_counts()

    def sqd_callback(results: list[SCIResult]):
        result_history.append(results)
        # iteration = len(result_history)
        # print(f"Iteration {iteration}")
        # for i, result in enumerate(results):
        #     print(f"\tSubsample {i}")
        #     print(f"\t\tEnergy: {result.energy + nuclear_repulsion_energy}")
        #     print(f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}")

    # SQD options
    energy_tol = 1e-3
    occupancies_tol = 1e-3
    max_iterations = 5

    # Eigenstate solver options
    num_batches = 1
    samples_per_batch = 1000
    symmetrize_spin = True
    carryover_threshold = 1e-4
    max_cycle = 100

    sci_solver = partial(solve_sci_batch, spin_sq=0.0, max_cycle=max_cycle)
    result_history = []

    try:
        result = diagonalize_fermionic_hamiltonian(
            hcore,
            eri,
            bit_array,
            samples_per_batch=samples_per_batch,
            norb=num_orbitals,
            nelec=nelec,
            num_batches=num_batches,
            energy_tol=energy_tol,
            occupancies_tol=occupancies_tol,
            max_iterations=max_iterations,
            sci_solver=sci_solver,
            symmetrize_spin=symmetrize_spin,
            carryover_threshold=carryover_threshold,
            callback=sqd_callback,
        )
        print(f"vqe_iter: {vqe_iter}, total_energy: {estimated_value[-1]+nuclear_repulsion_energy}, sqd_iter: {len(result_history)}, total_energy: {result.energy+nuclear_repulsion_energy}, subspace dims.: {np.prod(result.sci_state.amplitudes.shape)}")
    except IndexError:
        print(f"vqe_iter: {vqe_iter}, total_energy: {estimated_value[-1]+nuclear_repulsion_energy}")
        continue
