import os, sys
import time
from functools import partial
import numpy as np
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

def get_esproblem(atoms):
    import pyscf
    import pyscf.mcscf
    # atoms = [["H", (0.00, 0.00, 0.00)], ["H", (0.00, 0.00, 1.0)], ["H", (0.00, 0.00, 2.0)], ["H", (0.00, 0.00, 3.0)]]
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
    return es_problem, hcore, eri, nuclear_repulsion_energy, nelec, num_orbitals


def get_uccsd_ansatz(es_problem, mapper):
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
    return uccsd_ansatz

def main():

    atoms = [["H", (0.00, 0.00, 0.00)], ["H", (0.00, 0.00, 1.0)], ["H", (0.00, 0.00, 2.0)], ["H", (0.00, 0.00, 3.0)]]
    atoms = [["H", (0.00, 0.00, 0.00 + i * 1.0)] for i in range(4)]
    atoms = [["H", (0.00, 0.00, 0.00 + i * 1.0)] for i in range(16)]

    es_problem, hcore, eri, nuclear_repulsion_energy, nelec, num_orbitals \
        = get_esproblem(atoms=atoms)

    mapper = JordanWignerMapper()

    from qiskit.circuit.library import ExcitationPreserving, EfficientSU2

    # UCCSD
    uccsd_ansatz = get_uccsd_ansatz(es_problem, mapper)
    uccsd_initial_points = [0.0] * uccsd_ansatz.num_parameters

    # TwoLocal
    tlc_ansatz = TwoLocal(
        rotation_blocks=["h", "rx"],
        entanglement_blocks="cz",
        entanglement="full",
        reps=2,
        parameter_prefix="y",
    )
    # the initial point cannot be too small
    tlc_initial_points = [0.001] * (es_problem.num_spatial_orbitals * 2 * 3)

    # ExcitationPreserving
    ecipre_ansatz = ExcitationPreserving(
        num_qubits=es_problem.num_spatial_orbitals,
        # num_particles=es_problem.num_particles,
        # mapper=mapper,
    )
    ecipre_initial_points = [0.0] * (2*ecipre_ansatz.num_parameters)

    # EfficientSU2
    effsu2_ansatz = EfficientSU2(
        num_qubits=es_problem.num_spatial_orbitals,
        entanglement="full",
    )
    effsu2_initial_points = [0.01] * (effsu2_ansatz.num_parameters*2)

    optimizer = L_BFGS_B()
    backend = AerSimulator()
    estimator = Estimator()
    sampler = Sampler(mode=backend)

    evaluation_count=[]
    parameters_vars=[]
    estimated_value=[]
    meta_dict=[]

    def vqe_callback_list(counts, parameters, value, metadata):
        evaluation_count.append(counts)
        parameters_vars.append(parameters)
        estimated_value.append(value)
        meta_dict.append(metadata)
        # print(f"iter: {counts:4d}, energy: {value:.5f}, parameters: {parameters}")
        print(f"iter: {counts:4d}, energy: {value:.5f}")

    ansatz = effsu2_ansatz
    solver = VQE(estimator, ansatz, optimizer, callback=vqe_callback_list)
    solver.initial_point = effsu2_initial_points

    calc = GroundStateEigensolver(mapper, solver)
    res = calc.solve(es_problem)
    print('total_energy: ', res.total_energies[0])


    pass_manager = generate_preset_pass_manager(
        optimization_level=3, backend=backend, #initial_layout=initial_layout
    )
    pass_manager.pre_init = ffsim.qiskit.PRE_INIT

    ansatz.measure_all()
    isa_circuit = pass_manager.run(ansatz)
    job = sampler.run([(isa_circuit, parameters_vars[-1])], shots=10_000)
    primitive_result = job.result()
    print('primitive result:', primitive_result)
    pub_result = primitive_result[0]

    print("Pub")
    print(pub_result)

    bit_array = pub_result.data.meas
    counts = pub_result.data.meas.get_counts()

    # SQD options
    energy_tol = 1e-3
    occupancies_tol = 1e-3
    max_iterations = 5

    # Eigenstate solver options
    num_batches = 1
    samples_per_batch = 300
    symmetrize_spin = True
    carryover_threshold = 1e-4
    max_cycle = 100

    sci_solver = partial(solve_sci_batch, spin_sq=0.0, max_cycle=max_cycle)

    print(bit_array.shape)

    result_history = []
    def sqd_callback(results: list[SCIResult]):
        result_history.append(results)
        iteration = len(result_history)
        print(f"Iteration {iteration}")
        for i, result in enumerate(results):
            print(f"\tSubsample {i}")
            print(f"\t\tEnergy: {result.energy + nuclear_repulsion_energy}")
            print(f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}")

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


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")