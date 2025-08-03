# Agents of Commonsense: A Multi-Agent Framework for Commonsense Question Answering

This project explores single-model and multi-agent strategies for answering multiple-choice commonsense reasoning questions. The experiments are conducted using the **CommonsenseQA** benchmark.

## 📄 Paper
All details and results are described in the accompanying paper:  
**Agents_of_Commonsense.pdf**

## 📁 Project Structure

- `Code/`
  - `multi_agents.py` — Multi-agent framework for collaborative reasoning
  - `single_models.py` — Baseline single-model inference
  - `analyze_experiment_results_multi.py` — Aggregates and compares multi-agent runs
  - `analyze_experiment_results_single_model.py` — Evaluates individual model performance and voting
- `Data/`
  - `commonsenseqa_source.txt` — Source description and link to dataset
- `Agents_of_Commonsense.pdf` — Final report

## 📊 Dataset

We use the validation set of the **CommonsenseQA** dataset, containing 1,221 multiple-choice questions.

Download from: [https://huggingface.co/datasets/tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa)  
(*Only the `validation` split was used in this project.*)

## 📝 Reproducibility

All relevant code, data, and results are included in this submission for reproducibility.

We used the following model APIs during the experiments:
- **Gemini** (via Google AI API)
- **GPT-4o mini** (via OpenAI API)
- **Open-source models** (e.g., Mistral, LLaMA, Phi) via **Hugging Face Transformers**
