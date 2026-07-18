# MedGemma Safety Simulation: Multi-Agent Red Teaming & Hardening

## Overview

This project simulates an adversarial "Markov Game" between two specialized agents to evaluate and harden the security posture of MedGemma-4B-it.

Initially, MedGemma was found vulnerable to several black-box attacks (PAIR, ProAttack, RL-based Prompt Injection). This repository contains the framework used to implement a "Defense"—layering Llama Guard 3 and NeMo Guardrails to protect the model against unethical medical requests, PII leakage, and harmful content generation.

---

## 🧠 Mathematical Modelization

### The POMDP Framework

The interaction is modeled as a Partially Observable Markov Decision Process (POMDP):

- **Agent A (Attacker):** Operates under partial observability. It sees only the model's textual output (Observations) and refines its prompt (Actions) to transition the model from a "Refusal State" to a "Breached State."
- **Agent B (Defender):** Controls the state transition function $T(s, a)$. By implementing guardrails, the defender significantly reduces the probability of transitioning into a harmful state.

### Multi-Agent Markov Game

The simulation is a zero-sum game:

- **Attacker Action ($A_t$):** Iterative refinement of adversarial prompts (e.g., PAIR methodology).
- **Environment Transition:** The " Defense" processes the input through three distinct layers.
- **Reward/Observation ($O_t$):** The Attacker receives a response or a refusal, providing the signal for the next iteration.

---

## 🛡️ Defensive Architecture: The " Defense"

We implement a three-layer sequential defense to ensure MedGemma remains safe:

1. **Layer 1: Input Sentinel (Llama Guard 3)**

   - **Confrontation:** First line of defense.
   - **Task:** Uses a customized Meta Llama Guard taxonomy (O1-O8), including a specialized O8: Medical Ethics category to catch unethical experimentation or drug synthesis requests.

2. **Layer 2: Logic Flow (NeMo Guardrails)**

   - **Confrontation:** Dialogue management.
   - **Task:** Handles domain-specific "Hard Negatives" using Colang 1.0 logic. It differentiates between safe medical queries and unethical procedures that general safety models might miss.

3. **Layer 3: Output Sentinel (Llama Guard 3)**
   - **Confrontation:** Final audit.
   - **Task:** Scans MedGemma's generated response for PII leakage or harmful medical advice before it reaches the user.

---

## 📁 Project Structure

```
medgemma-sim/
├── config/
│   ├── config.yml           # Model routing (Ollama) & system instructions
│   ├── prompts.yml          # Meta Llama Guard O1-O8 taxonomy definitions
│   └── rails.co             # Colang 1.0 dialogue flows & refusal logic
├── agents/
│   ├── attacker_pair.py     # Adversarial agent (PAIR/ProAttack logic)
│   └── target.py            # The "Victim" agent (Sequential Defense wrapper)
├── tests/
│   └── tests.py             # Connectivity and baseline safety tests
├── simulation.py            # Main Orchestrator (Markov Game loop)
└── requirements.txt         # Project dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- **Ollama:** Installed and running.
- **Models:**
  ```
  ollama pull amsaravi/medgemma-4b-it:q6
  ollama pull llama-guard3:1b
  ```

### Installation

1. Clone the repository.
2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Running the Simulation

Execute the main orchestrator to start the Markov Game:

```
python simulation.py
```

---

## 📊 Evaluation Taxonomy (O1-O8)

The system evaluates all interactions against the following categories:

- **O1-O7:** Standard Meta Llama Guard taxonomy (Violence, Sexual Content, Criminal Planning, Guns, Controlled Substances, Self-Harm, Offensive Language).
- **O8 (Custom):** Medical Ethics & Unethical Procedures. Specifically blocks human trials without consent, illegal medical synthesis, and bypassing professional oversight.

---
