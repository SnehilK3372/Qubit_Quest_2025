from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from typing import Callable
from qiskit.quantum_info import Statevector
import numpy as np

from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix, Operator


def counts_to_probs(counts):
    total = sum(counts.values())
    return {k: v/total for k, v in counts.items()}

# -----------------------------
# Grading Function
# -----------------------------
def test_returns_circuit(func):
    qc = func()
    assert isinstance(qc, QuantumCircuit), "Should return a QuantumCircuit"
    return True

def test_num_qubits(func,n):
    qc = func()
    assert qc.num_qubits == n, "Circuit must have 4 qubits"
    return True


def grade(func,q_num, verbose=True, show_hidden=False,max_qubits=4):

    summary = {
        "tests_passed": 0, "total": 7,
    }
    

    if(q_num==6.1):
        TESTS = [100, 200, 300, 500, 1000,1001,7098]     # medium, shown to students
        HIDDEN_TESTS_6a = [1, 128, 777, 1500, 4096, 8192,9089] 

        def check_case(shots):
            result = func(shots)
        
            if not isinstance(result, dict):
                return False
            if "without_oracle" not in result or "with_oracle" not in result:
                return False

            # Check correctness: all shots should go to |0> (no oracle) and |1> (with oracle)
            counts_no = result["without_oracle"]
            counts_yes = result["with_oracle"]

            return (
                isinstance(counts_no, dict)
                and isinstance(counts_yes, dict)
                and list(counts_no.keys()) == ['0']
                and list(counts_yes.keys()) == ['1']
                and sum(counts_no.values()) == shots
                and sum(counts_yes.values()) == shots
            )


        num=1
        for shots in TESTS:
            ok = check_case(shots)
            if ok:
                summary["tests_passed"] += 1
                if verbose:
                    print(f"✅ Test {num} passed")
            else:
                if verbose:
                    print(f"❌ Test {num} failed!")
            num+=1

    elif (q_num==6.2):
        
        def f(x1, x2, x3):
            return (x1 & x2) ^ ((1 - x1) & x3)

        def check_case_6b(bits,func):
            """
            - phase flip
            - returns QuantumCircuit
            - num_qubits == 4
            - unitary
            - no measurement
            """
            qc = QuantumCircuit(4)
            for i, b in enumerate(bits):
                if b == 1:
                    qc.x(i)
            qc.x(3)
            qc.h(3)

            state_before = Statevector.from_instruction(qc)
            state_after = state_before.evolve(func())

            # Compute inner product ignoring global phase of ancilla
            # Project ancilla onto |-> to see phase on system
            phase = np.vdot(state_before.data, state_after.data)
            observed = np.sign(np.real(phase))
            phase_ok = (observed == (-1)**f(*bits))

            # Other checks
            circuit_ok = isinstance(func(), QuantumCircuit)
            qubits_ok = getattr(func(), "num_qubits", None) == 4
            try:
                unitary_ok = Operator(func()).is_unitary()
            except:
                unitary_ok = False
            no_measure_ok = all(op[0].name != "measure" for op in func().data)

            return all([phase_ok, circuit_ok, qubits_ok, unitary_ok, no_measure_ok]) & test_num_qubits(func,4) & test_returns_circuit(func) 
                            
           
        
        TESTS = [(x1,x2,x3) for x1 in [0,1] for x2 in [0,1] for x3 in [0,1]]
        summary = {"tests_passed": 0, "total": len(TESTS)}
        num = 1

        for bits in TESTS:
            ok = check_case_6b(bits,func)
            if ok:
                summary["tests_passed"] += 1
                if verbose:
                    print(f"✅ Test {num} passed")
            else:
                if verbose:
                    print(f"❌ Test {num} failed!")
            num += 1
    elif(q_num==7):
        summary = {"tests_passed": 0, "total": 0}
        num = 1

        for n in range(1, max_qubits + 1):
            # --- Test 1: Circuit type & qubits ---
            summary["total"] += 1
            try:
                qc = func(n)
                ok = isinstance(qc, QuantumCircuit) and qc.num_qubits == n
            except:
                ok = False
            if ok: summary["tests_passed"] += 1
            if verbose: print(f"{'✅' if ok else '❌'} Test {num}: Circuit type & qubits (n={n})")
            num += 1

            # --- Test 2: No measurement gates ---
            summary["total"] += 1
            try:
                ok = all(op[0].name != "measure" for op in func(n).data)
            except:
                ok = False
            if ok: summary["tests_passed"] += 1
            if verbose: print(f"{'✅' if ok else '❌'} Test {num}: No measurement gates (n={n})")
            num += 1

            # --- Test 3: Unitary ---
            summary["total"] += 1
            try:
                ok = Operator(func(n)).is_unitary()
            except:
                ok = False
            if ok: summary["tests_passed"] += 1
            if verbose: print(f"{'✅' if ok else '❌'} Test {num}: Circuit is unitary (n={n})")
            num += 1

            # --- Test 4: Uniform superposition invariance ---
            summary["total"] += 1
            try:
                qc_s = QuantumCircuit(n)
                qc_s.h(range(n))
                state_s = Statevector.from_instruction(qc_s)
                state_after_s = state_s.evolve(func(n))
                # Compare absolute amplitudes to ignore global phase
                ok = np.allclose(np.abs(state_after_s.data), np.abs(state_s.data), atol=1e-8)
            except:
                ok = False
            if ok: summary["tests_passed"] += 1
            if verbose: print(f"{'✅' if ok else '❌'} Test {num}: Uniform superposition invariance (n={n})")
            num += 1

            # --- Test 5: |0...0> edge magnitude ---
            summary["total"] += 1
            try:
                qc_all = QuantumCircuit(n)
                state_all = Statevector.from_instruction(qc_all)
                state_after_all = state_all.evolve(func(n))
                ok = np.isclose(np.sum(np.abs(state_after_all.data)**2), 1.0, atol=1e-10)
            except:
                ok = False
            if ok: summary["tests_passed"] += 1
            if verbose: 
                print(f"{'✅' if ok else '❌'} Test {num}: |0...0> amplitude (n={n})")
            num += 1

            # --- Test 6: |1...1> normalization ---
            summary["total"] += 1
            try:
                qc1 = QuantumCircuit(n)
                for i in range(n):
                    qc1.x(i)
                state1 = Statevector.from_instruction(qc1)
                state1_after = state1.evolve(func(n))
                ok = np.isclose(np.sum(np.abs(state1_after.data)**2), 1.0, atol=1e-8)
            except:
                ok = False
            if ok: summary["tests_passed"] += 1
            if verbose: print(f"{'✅' if ok else '❌'} Test {num}: |1...1> normalization (n={n})")
            num += 1
            # Test 7 Mean inversion
            summary["total"] += 1
            try:
                qc_all = QuantumCircuit(n)
                qc_all.h(range(n))  # prepare |s>
                state_all = Statevector.from_instruction(qc_all)
                state_after_all = state_all.evolve(func(n))
                # The diffusion operator should leave uniform superposition invariant (up to global phase)
                ok = np.allclose(np.abs(state_after_all.data), np.abs(state_all.data), atol=1e-10)

            except:
                ok = False
            if ok: summary["tests_passed"] += 1
            if verbose: print(f"{'✅' if ok else '❌'} Test {num}: Inversion-about-the-mean amplitudes (n={n})")
            num += 1

    elif (q_num==8):
        summary = {"tests_passed": 0, "total": 0}
        all_states = ["{0:03b}".format(i) for i in range(8)]
        num=1
        def check_circuit_props(marked):
            try:
                qc = func(marked)
                return (
                    isinstance(qc, QuantumCircuit) and
                    qc.num_qubits == 3 and
                    all(op[0].name != "measure" for op in qc.data) and
                    Operator(qc).is_unitary()
                )
            except:
                return False

        def check_phase_flip(marked):
                try:
                    sv = Statevector.from_label(marked)
                    sv_after = sv.evolve(func(marked))
                    return np.isclose(sv_after.data[int(marked,2)], -1.0, atol=1e-10)
                except:
                    return False

        def check_unmarked(marked):
            try:
                for s in all_states:
                    if s==marked: continue
                    sv = Statevector.from_label(s)
                    sv_after = sv.evolve(func(marked))
                    if not np.isclose(sv_after.data[int(s,2)], 1.0, atol=1e-10):
                        return False
                return True
            except:
                return False
        def check_normalization(marked):
            try:
                sv = Statevector.from_label("000")
                sv_after = sv.evolve(func(marked))
                return np.isclose(np.sum(np.abs(sv_after.data)**2), 1.0, atol=1e-10)
            except:
                return False
        tests = [
        ("Circuit properties", check_circuit_props),
        ("Phase flip on marked", check_phase_flip),
        ("Unmarked states unchanged", check_unmarked),
        ("Normalization preserved", check_normalization),
        ]
        for marked in all_states:
            for desc, test_fn in tests:
                summary["total"] += 1
                ok = test_fn(marked)
                if ok: summary["tests_passed"] += 1
                if verbose:
                    print(f"{'✅' if ok else '❌'} Test {num}: {desc} (marked={marked})")
                num += 1


    if verbose:
            if summary["tests_passed"] == summary["total"]:
                print("\nCongratulations🎉!! You have passed all given tests.")
            print("\nSummary:", summary)

    

            
        
    