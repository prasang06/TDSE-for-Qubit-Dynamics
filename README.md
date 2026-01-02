# Time-Dependent Schrödinger Equation Simulator for Qubit Dynamics

This project is a Python-based simulator for studying **static and driven qubit dynamics** by numerically solving the **time-dependent Schrödinger equation (TDSE)**.
It focuses on clean physics modeling, modular numerical solvers, and clear visualization of quantum dynamics.

The simulator supports:

* static qubits (time-independent Hamiltonians)
* driven qubits (time-dependent control fields)
* probability evolution
* Bloch-sphere representations of qubit states

---

## Overview

The core idea is simple:

1. Define a qubit Hamiltonian (static or driven)
2. Solve the TDSE numerically
3. Visualize how the quantum state evolves in time

The project uses:

* **JAX** for fast numerical computation
* **Matplotlib** for analysis and probability plots
* **Manim** for high-quality Bloch-sphere animations (for demonstrations)

---

## Running the Matplotlib Simulations

These scripts numerically solve the TDSE and produce **probability plots**.

### Static qubit

```bash
python static_qubit.py
```

This shows:

* constant state probabilities
* phase evolution without population transfer

### Driven qubit

```bash
python driven_qubit.py
```

This shows:

* time-dependent population transfer
* Rabi oscillations between the qubit states

These plots are useful for **analysis, debugging, and verification of the physics**.

---

## Bloch-Sphere Demonstrations

### Driven qubit
<p align="center">
  <img src="media/gifs/driven_bloch.gif" width="450">
</p>

### Static qubit
<p align="center">
  <img src="media/gifs/static_bloch.gif" width="450">
</p>

## Results (Matplotlib)

The Matplotlib simulations demonstrate:

* **Static qubit**

  * Constant populations
  * Bloch vector precessing around the z-axis
  * Pure phase evolution
<p align="center">
  <img src="media/static_qubit.png" width="450">
</p>

* **Driven qubit**

  * Population transfer between |0⟩ and |1⟩
  * Rabi oscillations
  * Strong dependence on drive amplitude and frequency
<p align="center">
  <img src="media/driven_qubit.png" width="450">
</p>

These results are consistent with the expected physics of two-level quantum systems.

---

## Notes

* The numerical solver uses a Runge–Kutta (RK4) method.
* The solver is written to be modular and extensible.
* The project is designed for learning, exploration, and visualization of qubit dynamics rather than production-level quantum simulation.

---

