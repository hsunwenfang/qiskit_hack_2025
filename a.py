import os, sys
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

def get_esproblem(atoms):
    import pyscf
    import pyscf.mcscf
    from qiskit_nature.units import DistanceUnit
    from qiskit_nature.second_q.drivers import PySCFDriver
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
    from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
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
    uccsd_initial_points = [0.0] * uccsd_ansatz.num_parameters
    return uccsd_ansatz, uccsd_initial_points

def get_tlc_ansatz(es_problem):
    from qiskit.circuit.library import TwoLocal
    tlc_ansatz = TwoLocal(
        rotation_blocks=["h", "rx"],
        entanglement_blocks="cz",
        entanglement="full",
        reps=2,
        parameter_prefix="y",
    )
    # the initial point cannot be too small
    tlc_initial_points = [0.001] * (es_problem.num_spatial_orbitals * 2 * 3)
    return tlc_ansatz, tlc_initial_points

def get_ecipre_ansatz(es_problem):
    from qiskit.circuit.library import ExcitationPreserving
    ecipre_ansatz = ExcitationPreserving(
        num_qubits=es_problem.num_spatial_orbitals,
    )
    ecipre_initial_points = [0.0] * (2*ecipre_ansatz.num_parameters)
    return ecipre_ansatz, ecipre_initial_points

def get_effsu2_ansatz(es_problem, reps):
    from qiskit.circuit.library import EfficientSU2
    print(es_problem.num_spatial_orbitals)
    effsu2_ansatz = EfficientSU2(
        num_qubits=es_problem.num_spatial_orbitals*2,
        entanglement="linear",
        reps=reps,
    )
    effsu2_initial_points = [0.01] * (effsu2_ansatz.num_parameters)
    return effsu2_ansatz, effsu2_initial_points

def main():


    skip_vqe = True

    h_count = 6
    atoms = [["H", (0.00, 0.00, 0.00 + i * 1.0)] for i in range(h_count)]

    n_count = 2
    atoms = [["N", (0.00, 0.00, 0.00)],  ["N", (0.00, 0.00, 1.19)]]

    atoms = [["N", (0.00, 0.00, 0.00)],  ["N", (0.00, 0.00, 1.19)]]

    atoms = [["C", (0, 0, 0)], ["O", (-1.1970, 0, 0)], ["O", (1.1970, 0, 0)]]

    es_problem, hcore, eri, nuclear_repulsion_energy, nelec, num_orbitals \
        = get_esproblem(atoms=atoms)

    print(nuclear_repulsion_energy)
    sys.exit()

    from qiskit_nature.second_q.mappers import JordanWignerMapper
    mapper = JordanWignerMapper()

    ansatz, initial_points = get_effsu2_ansatz(es_problem, 3)

    from qiskit_algorithms.optimizers import SLSQP, L_BFGS_B
    from qiskit.primitives import Estimator
    # api_token hsunwenfang
    API_TOKEN = "TIurfftvnSMw9Hpt-ebxpDfVhSFVaR76t6suu1tCba0Z"
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    optimizer = L_BFGS_B()
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=API_TOKEN,
        instance='crn:v1:bluemix:public:quantum-computing:us-east:a/507d516b33cb4e7e8c37ef9b295e9e85:70f7ee4f-ff64-43d0-b5de-867e52073b96::',
        name='q222',
        overwrite=True
    )
    service = QiskitRuntimeService(name="q222")
    backend = service.least_busy(operational=True, simulator=False)
    print(backend)
    estimator = Estimator()
    sampler = Sampler(mode=backend)

    if not skip_vqe:
        evaluation_count=[]
        parameters_vars=[]
        estimated_value=[]
        meta_dict=[]

        def vqe_callback(counts, parameters, value, metadata):
            evaluation_count.append(counts)
            parameters_vars.append(parameters)
            estimated_value.append(value+nuclear_repulsion_energy)
            meta_dict.append(metadata)
            # Save VQE info
            import pickle
            vqe_info = (ansatz, evaluation_count, parameters_vars, estimated_value, meta_dict)
            # Path("vqe_info_H{h_count}.pickle").write_bytes(pickle.dumps(vqe_info))
            Path("vqe_info_N{h_count}.pickle").write_bytes(pickle.dumps(vqe_info))

            # print(f"iter: {counts:4d}, energy: {value:.5f}, parameters: {parameters}")
            print(f"iter: {counts:4d}, energy: {value+nuclear_repulsion_energy:.5f}")

        from qiskit_algorithms import VQE
        solver = VQE(estimator, ansatz, optimizer, callback=vqe_callback)
        solver.initial_point = initial_points

        from qiskit_nature.second_q.algorithms import GroundStateEigensolver
        calc = GroundStateEigensolver(mapper, solver)
        res = calc.solve(es_problem)
        print('total_energy: ', res.total_energies[0])

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pass_manager = generate_preset_pass_manager(
        optimization_level=3, backend=backend, #initial_layout=initial_layout
    )
    import ffsim
    pass_manager.pre_init = ffsim.qiskit.PRE_INIT

    ansatz.measure_all()
    isa_circuit = pass_manager.run(ansatz)
    if skip_vqe:
        job = sampler.run([(isa_circuit, initial_points)], shots=10_000)
    else:
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

    from functools import partial
    from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian, solve_sci_batch
    sci_solver = partial(solve_sci_batch, spin_sq=0.0, max_cycle=max_cycle)

    print(bit_array.shape)

    result_history = []
    def sqd_callback(results: list[SCIResult]):
        result_history.append(results)
        iteration = len(result_history)
        print(f"Iteration {iteration}")
        for i, result in enumerate(results):
            print(f"\tSubsample {i}")
            print(f"\t\tEnergy: {result.energy+nuclear_repulsion_energy}")
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