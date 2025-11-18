# RouteLLM - BERT Router Reproduction & Improvements

This repository contains a reproduction of the BERT-based router from the RouteLLM paper, with critical findings and improvements to address training issues.

[ [Original Paper](https://arxiv.org/abs/2406.18665) ] [ [Blog](http://lmsys.org/blog/2024-07-01-routellm/) ]

## 🎯 Key Findings & Contributions

### Issues Identified in Original BERT Router Implementation
- **Majority Class Overfitting**: The authors' BERT routers were overfitting to the majority class, achieving poor macro F1 scores (0.23-0.35) across all their checkpoints
- **Dataset Imbalance**: Training datasets were heavily skewed (51% strong model wins), causing models to learn trivial majority predictors
- **Incomplete Dataset Release**: Authors claimed 65k samples but preprocessing yielded only 19k; they didn't release their exact D_arena dataset

### Our Fixes & Improvements
- **Dataset Rebalancing**: Implemented oversampling to balance classes, enabling meaningful training convergence
- **Training Pipeline Implementation**: Created complete, reproducible training scripts for BERT-based routers
- **Benchmarking Results**: Achieved competitive performance on MMLU, GSM8k, and MT-bench benchmarks

### Performance Results
Our rebalanced BERT router outperforms the original authors' checkpoints on multiple benchmarks while using significantly less data (19k vs 130k+ samples).

**Original Authors' Routers (Majority Class Overfitting):**
- D_arena: Macro F1 = 0.23 (99% accuracy on majority class only)
- D_arena + D_judge: Macro F1 = 0.29 (63% accuracy on majority class only)
- D_arena + D_gold: Macro F1 = 0.35 (98% accuracy on majority class only)

**Our Rebalanced Approach:** Competitive routing performance with balanced class predictions.

## 📁 Repository Structure

- **`./reproduce/`**: Complete training and evaluation pipeline for BERT-based routers
  - `train_roberta.py`: Training script using HuggingFace Trainer
  - `roberta-classifier.py`: Main training orchestration (Jupyter-compatible)
  - `benchmark_bert_models.py`: Evaluation against MMLU, GSM8k, MT-bench
  - `utils.py`: Data processing and evaluation utilities

- **`./routellm/`**: Original RouteLLM framework (see [README original.md](README%20original.md) for details)

## 🚀 Quick Start - Training Your Own BERT Router

```bash
# Install dependencies
pip install -e .[serve,eval]

# Train on rebalanced Chatbot Arena data
cd reproduce
python roberta-classifier.py  # Follow the notebook-style training
```

## 🔧 Technical Details

### Training Setup
- **Model**: XLM-RoBERTa-base classifier
- **Task**: 3-class classification (strong_win, tie, weak_win)
- **Data**: Rebalanced Chatbot Arena preference data
- **Framework**: HuggingFace Trainer with custom metrics

### Evaluation Benchmarks
- **MMLU**: 14k+ multiple choice questions across 57 subjects
- **GSM8k**: 1.3k grade school math problems
- **MT-bench**: 160 open-ended questions with LLM-as-judge scoring

## 📋 Recommendations for Future Work

1. **Class-Weighted Loss**: Implement proper class imbalance handling in the loss function for more stable training
2. **Larger Datasets**: Collect more balanced preference data to reduce overfitting risk
3. **Multi-Task Learning**: Explore joint training on multiple data sources (D_arena, D_gold, D_judge)

## 📖 Original Framework Documentation

For the complete RouteLLM framework documentation, server setup, and other routers (MF, SW-Ranking, Causal LLM), please see [README original.md](README%20original.md).

## 🤝 Contributing

We welcome contributions to the BERT router reproduction and improvements. For contributions to the broader RouteLLM framework, see [README original.md](README%20original.md).

# Citation

The code in this repository is based on the research from the [paper](https://arxiv.org/abs/2406.18665). Please cite if you find the repository helpful.

```
@misc{ong2024routellmlearningroutellms,
      title={RouteLLM: Learning to Route LLMs with Preference Data},
      author={Isaac Ong and Amjad Almahairi and Vincent Wu and Wei-Lin Chiang and Tianhao Wu and Joseph E. Gonzalez and M Waleed Kadous and Ion Stoica},
      year={2024},
      eprint={2406.18665},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.18665},
}
```
