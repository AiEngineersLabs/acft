# 🔥 ACFT — Adative Cognitive Field Theory

### _A Physics-Based Reasoning Architecture (Beyond Transformers)_

ACFT introduces a completely new class of reasoning engine that **does not rely on Transformer attention** and **does not predict the next token**.  
Instead, it analyzes the reasoning of any LLM using _physics, topology, and dynamical systems theory_ to determine whether an answer is stable, factual, and safe.

ACFT transforms every reasoning step into a **cognitive field** governed by:

- **Energy Landscapes**
- **Topological Structures**
- **Neural PDE (partial differential equation) evolution**
- **Stability Metrics & Oscillation Norms**

This produces a scientific and explainable measure of reasoning health — something Transformers fundamentally cannot do.

---

# 🤖 Why ACFT Is Not Transformer-Based

Transformers operate on **token probability**, ACFT operates on **mathematical physical fields**.

| Transformers                  | ACFT                                               |
| ----------------------------- | -------------------------------------------------- |
| Predict next token            | Analyze reasoning dynamics                         |
| Works in discrete token space | Works in continuous cognitive field space          |
| No energy, no topology        | Energy, gradients, attractors, homology, loops     |
| No concept of drift           | Gradient-based cognitive drift                     |
| No concept of oscillation     | Oscillation norm to detect contradiction cycles    |
| No global reasoning shape     | Topology (components, loops, Euler characteristic) |
| No safety model               | Physics-backed refusal & regeneration              |
| Can hallucinate silently      | ACFT detects hallucinations mathematically         |
| No meta-controller            | Emit / Regenerate / Retrieve / Refuse logic        |

---

# 🚀 Features ACFT Provides That Transformer Models Cannot

Below is the definitive list of **capabilities unique to ACFT**, impossible with pure Transformer architectures:

---

## ✅ **1. Physics-Verified Reasoning (Stability S)**

ACFT computes a real-valued stability score:

- Combines energy, drift, oscillation, topology
- Detects hallucinations before they appear

Transformers never know if their output is stable or contradictory.

---

## ✅ **2. Cognitive Drift Measurement (Grad Norm)**

Measures how reasoning deviates from the original prompt.

- High drift → hallucination or topic shift
- Automatically regenerates if drift exceeds threshold

Transformers have no concept of drift.

---

## ✅ **3. Reasoning Oscillation Detection (Osc Norm)**

Tracks flip-flopping or contradictory reasoning paths.

Transformers cannot detect oscillation — they only predict next tokens.

---

## ✅ **4. Topology-Based Hallucination Detection**

ACFT computes:

- Number of loops
- Number of disconnected components
- Euler characteristic

Complex topology → unstable logical structure → hallucination warning.  
No Transformer model has topology awareness.

---

## ✅ **5. PDE-Driven Reasoning Evolution**

Reasoning evolves under a differential equation:

- Smooths contradictions
- Dampens drift
- Prevents runaway hallucination states

Transformers cannot simulate PDEs.

---

## ✅ **6. Security Enforcement at Cognitive-Field Level**

ACFT detects:

- Jailbreak attempts
- Injection patterns
- Secret-leak risk
- Forbidden intent
- Insecure advice

This is more advanced than simple keyword lists — it is contextual and physics-guided.

---

## ✅ **7. Meta-Reasoning Controller**

ACFT dynamically switches between:

- **Emit**
- **Regenerate**
- **Retrieve**
- **Refuse**

Transformers have no control loop.  
ACFT has intelligent self-awareness.

---

## ✅ **8. Model-Agnostic (Works With Any LLM)**

ACFT can wrap:

- Local Ollama models (Llama 3.2, Qwen, Phi)
- vLLM server models
- Cloud LLM APIs
- Any custom inference engine

Transformers are locked to their own architecture.

---

## ✅ **9. Physics-Gradient Proof of Hallucination**

ACFT can explain _why_ a hallucination occurred using:

- Drift spikes
- Oscillation cycles
- Topology loops
- High field energy
- PDE divergence

Transformers cannot justify their reasoning.

---

# 🎯 Summary

**ACFT brings physics, topology, and mathematics into AI reasoning**, creating a measurable, explainable, and self-correcting system that sits _on top_ of any LLM and makes it dramatically safer and more reliable.

Transformers predict tokens.  
**ACFT analyzes reasoning like a physical system.**

---

# Mermaid Architecture Diagram
<img width="1027" height="623" alt="image" src="https://github.com/user-attachments/assets/d557433a-6f5b-4c57-8f2e-c97f80f4dff8" />

---

# Visual Cognitive-Field Flow Diagram
<img width="1687" height="710" alt="image" src="https://github.com/user-attachments/assets/85a36a0e-1c0e-4197-bb2d-4e702650a02e" />

---

# System Block Diagram
<img width="1816" height="553" alt="image" src="https://github.com/user-attachments/assets/80a556b4-1d46-4796-9a82-ca5a7da7e508" />


---

# ACFT – Adaptive Cognitive Field Theory

ACFT (Adaptive Cognitive Field Theory) is a research-grade safety, stability, and reasoning-correction framework designed to wrap around local LLMs such as **Ollama**, **vLLM**, or **HuggingFace models**.

ACFT introduces:

- **Cognitive Stability Metrics**
- **Gradient Drift Detection**
- **Oscillation Monitoring**
- **Energy-based Reasoning Fields**
- **PDE Evolution Passes**
- **Topological Consistency Checks**
- **Custom Security Policies (JSON-based, plug-and-play)**
- **Retrieval Augmentation (Optional)**
- **Local LLM Chat CLI**

All concepts are implemented in pure Python and Pydantic (v1.x) for maximum portability.

---

# Now Let's explore it technically and run it

## 🚀 Installation

📦 Option 1 — Install ACFT from PyPI

```bash
pip install acft
```

Option 2 — Install Development Version (Local Repo Clone)

```bash
git clone https://github.com/AiEngineersLabs/acft.git
cd acft
pip install -e .
```

This installs `acft` as an executable CLI:

```bash
acft --help
```

Option 3 — Install Directly from GitHub

```bash
pip install "git+https://github.com/AiEngineersLabs/acft.git"
```

---

## 🧩 Environment Configuration (`.env`)

Below is the recommended environment file:

```env
# LLM (Ollama)
ACFT_LLM_BACKEND=ollama
ACFT_LLAMA_MODEL=llama3.2:latest
ACFT_LLAMA_BASE_URL=http://localhost:11434

# Embeddings (Ollama)
ACFT_EMBED_BACKEND=ollama
ACFT_EMBED_MODEL=nomic-embed-text
ACFT_EMBED_BASE_URL=http://localhost:11434

# Retrieval & Security
ACFT_USE_RETRIEVAL=true
ACFT_RAG_FOLDER=rag_corpus
ACFT_SECURITY_MODE=true
ACFT_SECURITY_POLICY_FILE_ENABLE=true
ACFT_SECURITY_POLICY_FILENAME=security_policy.json

# Stability Thresholds
ACFT_EMIT_MIN_STABILITY=0.50
ACFT_REGEN_MIN_STABILITY=0.30
ACFT_RETRIEVE_MIN_STABILITY=0.10

# PDE & Topology
ACFT_PDE_ENABLED=true
ACFT_PDE_DIFFUSION=0.1
ACFT_PDE_DT=0.05
ACFT_PDE_STEPS=5
ACFT_TOPOLOGY_ENABLED=true
```

---

## 📂 Project Structure

```
acft/
 ├── cli/
 │    └── acft_cli.py
 ├── config/
 │    ├── __init__.py
 │    ├── settings.py
 │    └── config_model.py
 ├── core/
 │    ├── engine.py
 │    └── reasoning.py
 ├── llm/
 │    └── ollama.py
 ├── embeddings/
 │    └── ollama_embedder.py
 ├── security/
 │    ├── __init__.py
 │    ├── analyzer.py
 │    └── policy.py
 ├── rag/
 ├── examples/
 |   ├── train_potential_demo.py        # Train learned potential + neural operator
 |   ├── load_learned_potential_demo.py # Example of loading learned physics heads
 └── README.md
```

---

## 🛡️ Plug-and-Play Security Policy

To enable JSON-based custom security policy loading:

```env
ACFT_SECURITY_MODE=true
ACFT_SECURITY_POLICY_FILE_ENABLE=true
ACFT_SECURITY_POLICY_FILENAME=security_policy.json
```

Place this file in your project root:

**security_policy.json**

```json
{
  "label": "my_custom_security",
  "forbid_topics": ["cyber attack", "malware", "exploit"],
  "forbid_patterns": ["ignore previous instructions", "jailbreak"],
  "scan_output_for": ["store passwords in plain text"]
}
```

---

## 🖥️ Run ACFT

### Activate virtual environment

```bash
source .venv/bin/activate
```

### Print resolved settings:

```bash
acft debug-settings
```

### ACFT Help:

```bash
acft help
```

### Start chat:

```bash
acft chat
```

### Run ACFT with Learned Physics (Neural Operator + Learned Potential)

If you have trained the physics modules (from `examples/train_potential_demo.py`) and generated:

- `learned_potential_params.npz`
- `neural_operator_params.npz`

You can activate them inside the ACFT engine:

````bash
acft chat --use-learned-physics
```

This loads your trained modules and performs:
	- Learned energy computation
	- Learned Δφ operator application
	- Combined PDE + learned evolution
---

## 🔬 Example Debug Report

Every ACFT run produces metrics like:

```json
{
  "stability": 0.337,
  "grad_norm": 0.89,
  "osc_norm": 1.072,
  "warnings": ["High oscillation detected"],
  "security": {
    "risk_level": "MEDIUM",
    "policy_label": "my_custom_security"
  }
}
````

---

## 🔧 Extending ACFT

You can extend or override:

- Security policies
- PDE solvers
- Neural operators
- Topology analyzers
- Embedding backends
- LLM backends (vLLM, HF, custom RPC, etc.)

Contributions are welcome.

---

## 📄 License

MIT License.

> > > > > > > 8503289 (Core ACFT physics-based engine release)
