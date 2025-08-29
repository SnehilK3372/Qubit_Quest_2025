# AutoGrader.py
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import AerSimulator

# --- Helper Functions and Setup ---
# This simulator can be used by any of the tests
simulator = AerSimulator()

def run_qiskit_circuit(qc, shots):
    """A helper to run a circuit and get counts."""
    qc.measure_all()
    job = simulator.run(qc, shots=shots)
    return job.result().get_counts()

# --- The Main Grade Function ---
def grade2(func, q_num, verbose=True, show_hidden=False):
    """
    A centralized grading function that selects tests based on the question number.

    Args:
        func (function): The student's function to be tested.
        q_num (int): The question number.
        verbose (bool): If True, prints detailed results for each test case.
        show_hidden (bool): If True, runs the hidden test cases.
    """
    
    all_tests = {}
    
    # ======================================================================
    # == QUESTION 1: identify_input
    # ======================================================================
    all_tests[1] = {
        "visible": [
            (0, "bit"), (1, "bit"), ([1, 0], "qubit"),
            ((0.6, 0.8), "qubit"), ([1/math.sqrt(2), 1/math.sqrt(2)], "qubit"),
        
        
            (2, "neither"), ("1", "neither"), ([1, 1], "neither"),
            ([1, 0, 0], "neither"), ([1/math.sqrt(2), (1j)/math.sqrt(2)], "qubit"),
        ]
    }
    # ======================================================================
    # == QUESTION 2: double_hadamard
    # ======================================================================
    # Tuple format: (shots, expected_output_dict)
    all_tests[2] = {
        "visible": [
            (100, {'0': 100}), (2000, {'0': 2000}), (50, {'0': 50}),
            (20000, {'0': 20000})
        ]
    }

    # ======================================================================
    # == QUESTION 3: create_2qubit_phase_superposition
    # ======================================================================
    # Tuple format: (check_string, expected_value)
    all_tests[3] = {
        "visible": [
            ("RETURNS_TUPLE", True), ("RETURNS_QC_AND_SV", True), ("QC_HAS_2_QUBITS", True),
            ("STATEVECTOR_HAS_4_DIMS", True), ("STATEVECTOR_IS_CORRECT", True),
    
            ("GATE_COUNT_IS_OPTIMAL", True), ("CIRCUIT_DEPTH_IS_LOW", True),
            
        ]
    }
    
    # ======================================================================
    # == QUESTION 4: build_bell_state
    # ======================================================================
    # Tuple format: (check_string, expected_value)
    all_tests[4] = {
        "visible": [
            ("RETURNS_QC_WITH_2_QUBITS", True), ("VERIFIES_CORRECT_STATE", True),
            ("VERIFIES_INCORRECT_STATE", False), ("SIMULATES_CORRELATED_OUTCOMES", True),
    
            ("OPTIMAL_GATE_COUNT", True), ("VERIFIES_GLOBAL_PHASE", True),
            ("SIMULATES_ZERO_SHOTS", True)
        ]
    }
    
    # ======================================================================
    # == QUESTION 5: make_superposition
    # ======================================================================
    # Tuple format: (num_qubits, check_string)
    all_tests[5] = {
        "visible": [
            (2, "QC_HAS_2_QUBITS"), (4, "QC_HAS_4_QUBITS"), (2, "GATE_COUNT_IS_2"),
            (4, "GATE_COUNT_IS_4"), (1, "STATEVECTOR_IS_PLUS_STATE"),
    
            (2, "STATEVECTOR_IS_2Q_SUPERPOSITION"), (5, "CIRCUIT_DEPTH_IS_1"),
        ]
    }

    # --- Select and run the tests for the given q_num ---
    if q_num not in all_tests:
        print(f"ERROR: No tests found for question number {q_num}.")
        return

    tests_to_run = all_tests[q_num]["visible"]
    if show_hidden:
        tests_to_run.extend(all_tests[q_num]["hidden"])
        
    summary = {"tests_passed": 0, "total": len(tests_to_run)}
    if verbose:
        print(f"--- Running Tests for Question {q_num} ---")

    for i, test_case in enumerate(tests_to_run, 1):
        passed = False
        try:
            # Custom logic for each question's tests
            if q_num == 1:
                inp, expected = test_case
                actual = func(inp)
                passed = (actual == expected)

            elif q_num == 2: # double_hadamard
                shots, expected = test_case
                actual = func(shots)
                passed = (actual == expected)

            elif q_num == 3: # create_2qubit_phase_superposition
                check, _ = test_case
                qc, state = func() # Function takes no args
                if check == "RETURNS_TUPLE": passed = isinstance((qc, state), tuple)
                elif check == "RETURNS_QC_AND_SV": passed = isinstance(qc, QuantumCircuit) and isinstance(state, Statevector)
                elif check == "QC_HAS_2_QUBITS": passed = qc.num_qubits == 2
                elif check == "STATEVECTOR_HAS_4_DIMS": passed = state.dim == 4
                elif check == "STATEVECTOR_IS_CORRECT": passed = state.equiv(Statevector([0.5, 0.5, -0.5, -0.5]))
                elif check == "GATE_COUNT_IS_OPTIMAL": passed = qc.count_ops().get('h',0)==2 and qc.count_ops().get('z',0)==1
                elif check == "CIRCUIT_DEPTH_IS_LOW": passed = qc.depth() <= 2
                elif check == "STATEVECTOR_IS_NORMALIZED": passed = math.isclose(state.norm(), 1)

            elif q_num == 4: # build_bell_state
                check, expected = test_case
                user_qc = func()
                if check == "RETURNS_QC_WITH_2_QUBITS": passed = isinstance(user_qc, QuantumCircuit) and user_qc.num_qubits == 2
                elif check == "VERIFIES_CORRECT_STATE": passed = Statevector.from_instruction(user_qc).equiv(Statevector([1/math.sqrt(2),0,0,1/math.sqrt(2)]))
                elif check == "VERIFIES_INCORRECT_STATE": passed = not Statevector.from_instruction(user_qc).equiv(Statevector([0.5,0.5,0.5,0.5]))
                elif check == "SIMULATES_CORRELATED_OUTCOMES": passed = set(run_qiskit_circuit(user_qc, 100).keys()).issubset({"00", "11"})
                elif check == "OPTIMAL_GATE_COUNT": passed = user_qc.count_ops().get('h',0)==1 and user_qc.count_ops().get('cx',0)==1
                elif check == "VERIFIES_GLOBAL_PHASE": passed = Statevector.from_instruction(user_qc).equiv(Statevector([-1/math.sqrt(2),0,0,-1/math.sqrt(2)]))
                elif check == "SIMULATES_ZERO_SHOTS": passed = run_qiskit_circuit(user_qc, 0) == {}
            
            elif q_num == 5: # make_superposition
                num_qubits, check = test_case
                user_qc = func(num_qubits)
                if check == "QC_HAS_2_QUBITS": passed = user_qc.num_qubits == 2
                elif check == "QC_HAS_4_QUBITS": passed = user_qc.num_qubits == 4
                elif check == "QC_HAS_0_QUBITS": passed = user_qc.num_qubits == 0
                elif check == "GATE_COUNT_IS_2": passed = user_qc.count_ops().get('h',0) == 2
                elif check == "GATE_COUNT_IS_4": passed = user_qc.count_ops().get('h',0) == 4
                elif check == "STATEVECTOR_IS_PLUS_STATE": passed = Statevector.from_instruction(user_qc).equiv(Statevector([1/math.sqrt(2), 1/math.sqrt(2)]))
                elif check == "STATEVECTOR_IS_2Q_SUPERPOSITION": passed = Statevector.from_instruction(user_qc).equiv(Statevector([0.5, 0.5, 0.5, 0.5]))
                elif check == "CIRCUIT_DEPTH_IS_1": passed = user_qc.depth() <= 1 if num_qubits > 0 else True


            # --- Update summary and print result ---
            if passed:
                summary["tests_passed"] += 1
                if verbose: print(f"✅ Test {i} passed")
            else:
                if verbose: print(f"❌ Test {i} failed!")

        except Exception as e:
            if verbose: print(f"  💥 Test {i}: Crashed! Error: {e}")

    if verbose:
        print("\nSummary:", summary)
        if summary["tests_passed"] == summary["total"]:
            print("\nCongratulations🎉!! You have passed all given tests.")
        
    