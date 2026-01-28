# Week 2 Lab Report: Fine-Tuning a Generative AI Model for Dialogue Summarization

**Course:** Generative AI with Large Language Models (DeepLearning.AI)
**Author:** Kris K
**Date:** January 2026

---

## Introduction

This report documents the findings from Lab 2 of the Generative AI with LLMs course, which focused on fine-tuning large language models for dialogue summarization tasks. Unlike the previous lab that involved hands-on prompt engineering experiments, this lab was primarily a guided walkthrough exploring the differences between full fine-tuning and Parameter-Efficient Fine-Tuning (PEFT) using the LoRA technique.

The lab demonstrated how pre-trained models can be adapted to specific tasks while balancing computational efficiency against model performance—a critical consideration for real-world deployment scenarios.

---

## Technical Setup

### Model Architecture

The lab utilized **FLAN-T5-base** (`google/flan-t5-base`), a sequence-to-sequence language model with instruction-tuning capabilities. Key specifications:

| Parameter | Value |
|-----------|-------|
| Total Parameters | 247,577,856 (~248M) |
| Architecture | Encoder-Decoder (Seq2Seq) |
| Precision | bfloat16 |
| Source | Hugging Face Transformers |

### Dataset

The **DialogSum** dataset (`knkarthick/dialogsum`) was used for training and evaluation:

| Split | Samples |
|-------|---------|
| Training | 12,460 |
| Validation | 500 |
| Test | 1,500 |

The dataset contains dialogues with manually labeled summaries and topics, making it well-suited for summarization fine-tuning tasks.

### Libraries and Dependencies

- **transformers** (v4.38.2) - Model loading and training utilities
- **datasets** (v2.17.0) - Dataset management
- **peft** (v0.3.0) - Parameter-Efficient Fine-Tuning implementation
- **evaluate** (v0.4.0) - ROUGE metric computation
- **torch** (v2.5.1) - Deep learning framework
- **accelerate** (v0.28.0) - Training acceleration

---

## Methodology

### Approach Overview

The lab followed a structured walkthrough comparing three model configurations:

1. **Original Model (Zero-Shot):** Base FLAN-T5 without any task-specific fine-tuning
2. **Instruct Model (Full Fine-Tuning):** FLAN-T5 with all parameters trained on the dialogue summarization task
3. **PEFT Model (LoRA):** FLAN-T5 with only LoRA adapter parameters trained

### Prompt Template

All models used a consistent prompt structure for dialogue summarization:

```
Summarize the following conversation.

{dialogue}

Summary:
```

### LoRA Configuration

The PEFT approach employed Low-Rank Adaptation (LoRA) with the following hyperparameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| r (rank) | 32 | Dimension of the low-rank matrices |
| lora_alpha | 32 | Scaling factor for LoRA layers |
| target_modules | ["q", "v"] | Query and Value attention layers |
| lora_dropout | 0.05 | Dropout probability for LoRA layers |
| task_type | SEQ_2_SEQ_LM | Sequence-to-sequence language modeling |

### Evaluation Metrics

Model performance was assessed using **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) metrics:

- **ROUGE-1:** Unigram overlap between generated and reference summaries
- **ROUGE-2:** Bigram overlap
- **ROUGE-L:** Longest common subsequence
- **ROUGE-Lsum:** Summary-level ROUGE-L

---

## Results and Observations

### Qualitative Comparison

The lab demonstrated clear qualitative differences between the three approaches using a sample dialogue about system upgrades:

**Human Baseline Summary:**
> "#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system."

| Model | Generated Summary |
|-------|-------------------|
| Original (Zero-Shot) | "The computer you're using is outdated." |
| Full Fine-Tuned | "#Person1# suggests #Person2# upgrading #Person2#'s system, hardware, and CD-ROM drive. #Person2# thinks it's great." |
| PEFT (LoRA) | Comparable quality to full fine-tuning |

The zero-shot model failed to capture the instructional nature of the conversation, while both fine-tuned versions produced contextually appropriate summaries.

### Quantitative Performance

Based on the lab's pre-computed results across the full test dataset, the PEFT model achieved performance metrics remarkably close to the fully fine-tuned model, with only minor percentage differences in ROUGE scores. The key finding was that PEFT delivered near-equivalent results despite training a fraction of the parameters.

### Efficiency Comparison

| Aspect | Full Fine-Tuning | PEFT (LoRA) |
|--------|------------------|-------------|
| Trainable Parameters | 247,577,856 (100%) | ~0.3-1% of original |
| Model Checkpoint Size | ~945 MB | ~MBs (adapter only) |
| Training Resources | Multiple GPUs typically required | Single GPU feasible |
| Training Time | Hours | Significantly reduced |

---

## Key Findings

### 1. PEFT Achieves Comparable Performance

The most significant finding was that Parameter-Efficient Fine-Tuning with LoRA achieved performance nearly equivalent to full fine-tuning on the dialogue summarization task. The slight reduction in ROUGE metrics was marginal compared to the efficiency gains.

### 2. Dramatic Resource Reduction

PEFT reduced the number of trainable parameters from ~248 million to a small fraction (typically less than 1%), enabling fine-tuning on significantly less powerful hardware. The lab noted that full fine-tuning would require hours on GPU, while PEFT completes much faster.

### 3. Adapter Modularity

The LoRA approach produces small adapter files that can be combined with the base model at inference time. This enables:

- Multiple task-specific adapters sharing one base model
- Reduced storage requirements for serving multiple fine-tuned variants
- Easier model versioning and experimentation

### 4. Zero-Shot Limitations

The original FLAN-T5 model, despite being instruction-tuned, struggled with dialogue summarization in zero-shot settings. This underscores that domain-specific fine-tuning remains valuable even for capable base models.

---

## Limitations and Considerations

### Lab-Specific Constraints

Since this was a guided walkthrough rather than an independent experiment, several limitations apply:

1. **Pre-computed Results:** The lab used checkpoint models and pre-computed metrics rather than training from scratch, so actual training time comparisons were not directly observed
2. **Subsampled Training:** The interactive portions used heavily subsampled data (125 training samples) for time efficiency
3. **Single Configuration:** Only one LoRA configuration (r=32) was tested; different ranks may yield different efficiency-performance tradeoffs

### General PEFT Considerations

- Performance gap may widen for more complex tasks or smaller base models
- LoRA adds inference latency when combining adapter with base model
- Optimal LoRA rank requires task-specific tuning

---

## Conclusions

This lab effectively demonstrated that PEFT with LoRA represents a practical and efficient approach to fine-tuning large language models. The key takeaway is that organizations can achieve near-full-fine-tuning performance while dramatically reducing computational requirements—a critical consideration for deploying customized LLMs in resource-constrained environments.

The efficiency-performance tradeoff offered by PEFT makes it particularly attractive for:

- Rapid prototyping and experimentation
- Organizations with limited GPU resources
- Multi-tenant applications requiring multiple specialized models
- Scenarios where quick iteration is more valuable than marginal performance gains

---

## References

1. FLAN-T5 Model Documentation - Hugging Face. https://huggingface.co/docs/transformers/model_doc/flan-t5
2. DialogSum Dataset - Hugging Face. https://huggingface.co/datasets/knkarthick/dialogsum
3. PEFT Library Documentation - Hugging Face. https://huggingface.co/docs/peft
4. LoRA: Low-Rank Adaptation of Large Language Models. Hu et al., 2021. https://arxiv.org/abs/2106.09685
5. ROUGE Metric - Wikipedia. https://en.wikipedia.org/wiki/ROUGE_(metric)
6. Generative AI with Large Language Models - DeepLearning.AI. https://www.deeplearning.ai/courses/generative-ai-with-llms/
