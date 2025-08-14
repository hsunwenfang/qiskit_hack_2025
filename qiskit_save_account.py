from qiskit_ibm_runtime import QiskitRuntimeService

your_api_key = input("Enter your API token: ").strip()
your_crn = input("Enter your CRN: ").strip()
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token=your_api_key,
    instance=your_crn,
    name="qiskit_hack_2025",
    overwrite=True
)

service = QiskitRuntimeService(name="qiskit_hack_2025")

backends = service.backends()

print("Backends:")
for backend in backends:
    print(f"\t{backend.name}")
