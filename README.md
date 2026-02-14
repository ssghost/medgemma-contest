# MedGemma Local Triage: Secure Edge AI for Offline Clinical Support

[![Competition: Kaggle MedGemma Impact Challenge](https://img.shields.io/badge/Competition-Kaggle_MedGemma-blue?style=for-the-badge)](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
[![Model: MedGemma 1.5 4B](https://img.shields.io/badge/Model-MedGemma_1.5_4B-lightgrey?style=for-the-badge)](https://huggingface.co/google/medgemma-1.5-4b-it)

MedGemma Local Triage is a privacy-first, fully autonomous clinical navigation system designed for high-stakes medical environments with zero connectivity. By leveraging **MedGemma 1.5 4B** on edge hardware, the system delivers expert-level clinical reasoning while ensuring 100% data residency.


## Agentic Architecture

The system utilizes an **Agentic Orchestration** layer built on **LangGraph**, transforming a static medical model into a stateful clinical agent.

![LangGraph Architecture Flow](./assets/langchain-chart.png)
*[Figure 1: Stateful Agentic Workflow featuring self-correcting RAG and persistent memory nodes.]*

### Core Features:
* **Offline Clinical RAG**: Sub-second retrieval from indexed **MSF Clinical Guidelines** and **US Army First Aid** manuals via local **ChromaDB**.
* **Stateful Memory**: Persistent session tracking through `MemorySaver`, allowing the agent to correlate symptoms across multi-turn interactions.
* **Self-Correcting Loops**: Dynamic **Query Rewriting** and **Relevance Grading** nodes that refine search parameters if initial clinical context is insufficient.
* **Edge Optimization**: Tailored for **Intel Mac** environments using **Metal-accelerated llama.cpp** for AMD GPUs.


## Technical Stack

* **LLM Engine**: MedGemma 1.5 4B (quantized to **GGUF Q4_K_M**).
* **Orchestration**: LangGraph (Agentic state management).
* **Inference**: llama.cpp with Metal/Metal-Enabled acceleration.
* **Vector DB**: ChromaDB (Local-first).
* **Frontend**: Streamlit (Medical Deep Blue #003366 Theme).


## Reproducibility & Setup

This project uses **PDM** for dependency management to ensure a reproducible environment.

### 1. Prerequisites
* Python 3.10+
* Intel Mac with AMD GPU (Metal enabled)
* Hugging Face account with granted access to MedGemma 1.5 4B

### 2. Environment Installation
```bash
# Install PDM if not already present
curl -sSL [https://pdm-project.org/install-python.py](https://pdm-project.org/install-python.py) | python3 -

# Initialize project and dependencies
pdm install
```
### 3. Model Quantization (Edge Preparation)

We use the Q4_K_M quantization level to maintain clinical integrity while operating within local hardware constraints (16GB RAM).

```bash
# Clone llama.cpp and compile with METAL=1 for AMD GPU support
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_METAL=1

# Convert HF weights to GGUF and quantize to Q4
pdm run python convert.py models/medgemma-1.5-4b-it/
./quantize models/medgemma-1.5-4b-it/ggml-model-f16.gguf models/medgemma-q4_k_m.gguf Q4_K_M
```

### 4. Running the Application

Ensure your local environment is activated via PDM before launching the services.

```bash
# Index the medical knowledge base (MSF & US Army manuals)
pdm run python src/rag/build_db.py

# Launch the MedGemma Local Triage UI
pdm run streamlit run src/ui/app.py
```

## Project Structure

The repository is organized into a modular structure to ensure technical review transparency.

```
├── assets/           # Professional medical iconography
├── example/          # Documented clinical triage benchmarks
├── src/
│   ├── agent/        # LangGraph nodes and clinical edge logic
│   ├── rag/          # ChromaDB RAG and guideline indexing
│   └── ui/           # Streamlit interface with medical dark-mode theme
├── pyproject.toml    # PDM dependency configuration
└── README.md         # Project documentation and reproducibility guide
```

## License & Acknowledgments

* **Model License**: This project uses **MedGemma 1.5 4B**, which is subject to the **Gemma Terms of Use**.
* **Medical References (Open Access)**:
    * **MSF Clinical Guidelines**: Sourced from *Médecins Sans Frontières* (Doctors Without Borders). These are provided as open-access clinical protocols for global humanitarian use.
    * **US Army First Aid Manual**: Sourced from public domain military medical training materials.

> ### ⚠️ Clinical Disclaimer
> **IMPORTANT**: As per Google's official documentation, **MedGemma is not yet clinical-grade**. This application is a technical demonstration for the Kaggle challenge and must not be used for actual medical diagnosis or treatment in a real-world clinical setting.