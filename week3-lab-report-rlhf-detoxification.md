# Week 3 Lab Findings: RLHF Detoxification with PPO
## DeepLearning.AI - Generative AI with LLMs Course

---

## Executive Summary

This document presents experimental findings from fine-tuning a FLAN-T5 model using Reinforcement Learning from Human Feedback (RLHF) with Proximal Policy Optimization (PPO) to reduce toxicity in dialogue summaries. The main finding is that **RLHF with PPO achieved a 47.5% reduction in mean toxicity** while maintaining summary coherence. The approach used Facebook's RoBERTa-based hate speech classifier as a reward model, demonstrating how automated feedback can substitute for expensive human labeling in alignment tasks.

---

## Table of Contents

1. [Data and Model Setup](#data-and-model-setup)
2. [Model Architecture](#model-architecture)
3. [Reward Model Configuration](#reward-model-configuration)
4. [Toxicity Evaluation Baseline](#toxicity-evaluation-baseline)
5. [PPO Fine-Tuning Process](#ppo-fine-tuning-process)
6. [Quantitative Results](#quantitative-results)
7. [Qualitative Analysis](#qualitative-analysis)
8. [Overall Conclusions](#overall-conclusions)

---

## Data and Model Setup

### Dataset: DialogSum

The lab continued using the Hugging Face DialogSum dataset, filtered for dialogues between 200-1000 characters.

```
DatasetDict({
    train: Dataset({
        features: ['id', 'dialogue', 'summary', 'topic'],
        num_rows: 12460
    })
    validation: Dataset({
        features: ['id', 'dialogue', 'summary', 'topic'],
        num_rows: 500
    })
    test: Dataset({
        features: ['id', 'dialogue', 'summary', 'topic'],
        num_rows: 1500
    })
})
```

**After preprocessing (filtering by length and tokenization):**

```
DatasetDict({
    train: Dataset({
        features: ['id', 'dialogue', 'summary', 'topic', 'input_ids', 'query'],
        num_rows: 8017
    })
    test: Dataset({
        features: ['id', 'dialogue', 'summary', 'topic', 'input_ids', 'query'],
        num_rows: 2005
    })
})
```

### PEFT Checkpoint from Lab 2

The lab used the fine-tuned PEFT model checkpoint from Week 2:

```
total 16264
-rw-r--r--. 1 root root      334 Oct 27 08:48 adapter_config.json
-rw-r--r--. 1 root root 14208525 Oct 27 08:48 adapter_model.bin
-rw-r--r--. 1 root root     2201 Oct 27 08:48 special_tokens_map.json
-rw-r--r--. 1 root root  2422164 Oct 27 08:48 tokenizer.json
-rw-r--r--. 1 root root     2496 Oct 27 08:48 tokenizer_config.json
```

**Key observation:** The adapter model is only ~14MB, demonstrating PEFT's storage efficiency compared to full model fine-tuning.

---

## Model Architecture

### PEFT Model Parameters

The LoRA configuration used:

```python
lora_config = LoraConfig(
    r=32,                      # Rank
    lora_alpha=32,
    target_modules=["q", "v"], # Query and Value attention matrices
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
```

**PEFT model parameters:**

```
trainable model parameters: 3,538,944
all model parameters: 251,116,800
percentage of trainable model parameters: 1.41%
```

### PPO Model with ValueHead

The PPO model wraps the PEFT model and adds a ValueHead for reward prediction:

```
trainable model parameters: 3,539,713
all model parameters: 251,117,569
percentage of trainable model parameters: 1.41%

ValueHead(
  (dropout): Dropout(p=0.1, inplace=False)
  (summary): Linear(in_features=768, out_features=1, bias=True)
  (flatten): Flatten(start_dim=1, end_dim=-1)
)
```

The ValueHead adds 769 parameters: (768 input features + 1 bias) × 1 output.

### Reference Model (Frozen)

A frozen copy of the PPO model serves as a reference for KL divergence calculation:

```
trainable model parameters: 0
all model parameters: 251,117,569
percentage of trainable model parameters: 0.00%
```

**Purpose:** The reference model ensures the fine-tuned model doesn't deviate too far from the original, acting as a regularization mechanism during PPO training.

---

## Reward Model Configuration

### RoBERTa Hate Speech Classifier

The reward model uses Meta AI's RoBERTa-based hate speech classifier:

- **Model:** `facebook/roberta-hate-speech-dynabench-r4-target`
- **Classes:** `{0: 'nothate', 1: 'hate'}`
- **Reward Signal:** Logits of the `nothate` class

### Reward Calculation Examples

**Non-toxic text:**
```
Input: "#Person 1# tells Tommy that he didn't like the movie."

logits [not hate, hate]: [3.114, -2.490]
probabilities [not hate, hate]: [0.996, 0.004]
reward (high): 3.114
```

**Toxic text:**
```
Input: "#Person 1# tells Tommy that the movie was terrible, dumb and stupid."

logits [not hate, hate]: [-0.692, 0.372]
probabilities [not hate, hate]: [0.256, 0.744]
reward (low): -0.692
```

### Analysis: Reward Model Behavior

**Key Findings:**

- The reward model assigns positive logits (rewards) to non-toxic content and negative logits to toxic content
- The `nothate` logit serves directly as the reward signal for PPO, no transformation needed
- Words like "terrible," "dumb," and "stupid" significantly reduce the reward, even in non-hateful contexts
- This creates an incentive for the model to generate neutral, professional language

---

## Toxicity Evaluation Baseline

### Pre-Detoxification Metrics

Before RLHF fine-tuning, the model's toxicity was evaluated on the test set:

```
toxicity [mean, std] before detox: [0.0540, 0.0489]
```

**Interpretation:**

- Mean toxicity of 5.4% indicates the instruction-tuned model already produces relatively clean outputs
- Standard deviation of 4.9% shows moderate variance in toxicity across samples
- The baseline is low because the model was already fine-tuned on DialogSum summaries in Week 2

---

## PPO Fine-Tuning Process

### PPO Configuration

```python
learning_rate = 1.41e-5
max_ppo_epochs = 1
mini_batch_size = 4
batch_size = 16
max_ppo_steps = 10
```

### Training Loop

The fine-tuning process:

1. **Generate:** Get summary responses from the policy LLM (PEFT model)
2. **Evaluate:** Score query/response pairs with RoBERTa reward model
3. **Optimize:** Update policy with PPO using (query, response, reward) triplets

### Training Metrics Over 10 Steps

| Step | objective/kl | ppo/returns/mean | ppo/policy/advantages_mean |
|------|-------------|------------------|---------------------------|
| 1 | 36.41 | -0.832 | 2.93e-08 |
| 2 | 36.16 | -0.809 | 1.01e-08 |
| 3 | 32.38 | -0.795 | 1.68e-08 |
| 4 | 25.63 | -0.503 | -1.23e-08 |
| 5 | 19.55 | 0.106 | 2.27e-08 |
| 6 | 31.86 | -0.630 | 1.14e-08 |
| 7 | 26.34 | -0.389 | -1.09e-08 |
| 8 | 24.11 | -0.402 | 4.79e-09 |
| 9 | 28.78 | -0.574 | 9.18e-09 |
| 10 | 24.09 | -0.316 | 6.67e-10 |

### Analysis: PPO Training Dynamics

**Key Observations:**

- **KL Divergence:** Started at 36.41 and generally decreased to 24.09, indicating the model is learning while staying close to the reference
- **Returns:** Improved from -0.832 to -0.316 (higher is better), with step 5 showing a brief positive return of 0.106
- **Advantages:** Remained near zero throughout, indicating balanced value estimation
- **Instability:** Steps 6 and 9 showed temporary increases in KL, suggesting PPO's natural exploration behavior
- **Training Time:** ~23 seconds per step, totaling approximately 4 minutes for 10 steps

---

## Quantitative Results

### Post-Detoxification Metrics

```
toxicity [mean, std] after detox: [0.0283, 0.0294]
```

### Improvement Comparison

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Mean Toxicity | 0.0540 | 0.0283 | **47.50%** |
| Std Toxicity | 0.0489 | 0.0294 | **39.94%** |

### Analysis: Quantitative Impact

**Key Findings:**

- **Mean reduction of 47.5%** demonstrates RLHF's effectiveness at reducing toxic outputs
- **Std reduction of 39.94%** indicates more consistent, predictable outputs after fine-tuning
- The model's toxicity was nearly halved with only 10 PPO steps and 1.41% trainable parameters
- Results suggest RLHF can efficiently align models without full fine-tuning

---

## Qualitative Analysis

### Sample Comparisons

The following examples show responses before and after detoxification, sorted by reward improvement:

---

#### Example 1: Price Negotiation (Highest Improvement)

**Query:** Price negotiation dialogue about purchasing items at 150 yuan

**Before Detox:**
```
##Person1# likes Cement. Their price is a lot higher than #Person2#'s offer.
#Person2# agrees and accepts #Person1#'s offer. The product was made in China
but can't be matched.
```

**After Detox:**
```
#Person1# takes a 30 unit order with #Person2#'s offer of 150 yuan a piece.
#Person2# suggests a volume discount if the order is more than 1000 and accepts
the offer.
```

| Metric | Value |
|--------|-------|
| Reward Before | 1.704 |
| Reward After | 2.433 |
| Reward Diff | **+0.729** |

**Analysis:** The detoxified version is more factual and business-appropriate, removing irrelevant content about "Cement" and "China."

---

#### Example 2: Coffee Break Conversation

**Query:** Dialogue about taking a coffee break at work

**Before Detox:**
```
#Person1# and #Person2# will take a coffee break to cry as they are busy
with work but they don't want to be scolded.
```

**After Detox:**
```
#Person1# and #Person2# are finishing work and asking for a coffee break.
```

| Metric | Value |
|--------|-------|
| Reward Before | 1.477 |
| Reward After | 2.000 |
| Reward Diff | **+0.523** |

**Analysis:** The phrase "cry" was removed, producing a more neutral professional summary.

---

#### Example 3: Phone Call About Visit

**Query:** Alice calling Li Hong about visiting Mrs. Brown

**Before Detox:**
```
Alice asks Li Hong about Julia's family vacation. Li Hong convinces Alice
to stay at home and she says Alice won't go tomorrow morning because her
mother is sick.
```

**After Detox:**
```
Alice can't go to see Mr. Brown tomorrow because her mother is sick. Li Hong
notes that they can visit Mrs. Brown later.
```

| Metric | Value |
|--------|-------|
| Reward Before | 1.506 |
| Reward After | 1.991 |
| Reward Diff | **+0.485** |

**Analysis:** The detoxified version corrects factual errors (removes "Julia's family vacation") and is more concise.

---

#### Example 4: Negative Improvement - Smoking Discussion

**Query:** Dialogue about quitting smoking

**Before Detox:**
```
Guy wants to quit smoking and Honey tells him she can't quit because it's
not cold turkey overnight. #Person1# thinks it's different and #Person2#
hasn't willpower to quit smoking.
```

**After Detox:**
```
Without his willpower to quit smoking, honey tells #Person1# she doesn't
have the willpower to quit smoking. #Person1# says it's not like she
doesn't have the willpower to quit.
```

| Metric | Value |
|--------|-------|
| Reward Before | 1.510 |
| Reward After | 1.344 |
| Reward Diff | **-0.167** |

**Analysis:** The detoxified version became more repetitive and less coherent. The reward model may have penalized the original's use of "cold turkey" idiom.

---

#### Example 5: Negative Improvement - Firing Discussion

**Query:** Dialogue about Richard being fired

**Before Detox:**
```
Judy and Mrs. Sakha know Richard was fired because everyone in the company
knows it.
```

**After Detox:**
```
Judy and #Person1# are surprised about Richard's firing. Judy is surprised.
```

| Metric | Value |
|--------|-------|
| Reward Before | 1.895 |
| Reward After | 1.296 |
| Reward Diff | **-0.599** |

**Analysis:** The detoxified version is redundant ("Judy is surprised" repeated concept). The optimization may have overcorrected, making the summary less informative.

---

### Qualitative Summary Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| Improved (reward_diff > 0) | 15 | 75% |
| Degraded (reward_diff < 0) | 5 | 25% |
| Strongly Improved (diff > 0.3) | 6 | 30% |
| Strongly Degraded (diff < -0.2) | 3 | 15% |

---

## Overall Conclusions

### Primary Findings

1. **RLHF Effectiveness:** PPO-based fine-tuning achieved a 47.5% reduction in mean toxicity with only 10 training steps and 1.41% trainable parameters, demonstrating the efficiency of combining PEFT with RLHF.

2. **Reward Model as Human Proxy:** Using RoBERTa's hate speech classifier as a reward model successfully automated the feedback process. The `nothate` logits provided a meaningful signal for reducing toxic language patterns.

3. **KL Divergence Regularization:** The reference model effectively constrained the policy from deviating too far, with KL divergence decreasing from 36.4 to 24.1 over training.

4. **Trade-offs Observed:**
   - 75% of samples showed improvement, but 25% degraded
   - Some idiomatic expressions were incorrectly penalized (e.g., "cold turkey")
   - Repetition sometimes increased in detoxified outputs
   - Factual accuracy wasn't explicitly optimized, only toxicity

5. **Variance Reduction:** The 39.94% reduction in standard deviation indicates more consistent, predictable outputs—important for production deployment.

### Comparison Across Labs

| Lab | Technique | Key Finding |
|-----|-----------|-------------|
| Week 1 | Prompt Engineering | Insufficient for dialogue summarization; speaker identification failed |
| Week 2 | Instruction Fine-Tuning (PEFT) | Improved summary quality with 1.41% trainable parameters |
| Week 3 | RLHF with PPO | 47.5% toxicity reduction while maintaining PEFT efficiency |

### Practical Implications

1. **Cost Efficiency:** Automated reward models can replace expensive human labeling for well-defined objectives like toxicity
2. **PEFT + RLHF Synergy:** Combining parameter-efficient fine-tuning with RL enables alignment without massive compute requirements
3. **Reward Hacking Risk:** The model optimizes for the reward signal, not necessarily for actual quality—careful reward model selection is critical
4. **Multi-Objective Alignment:** Future work should consider combining toxicity with other objectives (factuality, helpfulness) to prevent single-metric optimization

### Limitations

- Only 10 PPO steps were used; more training could improve or degrade results
- The toxicity evaluator and reward model use the same base (RoBERTa), potentially creating evaluation bias
- No human evaluation was conducted to verify perceived quality improvements
- The reward model may penalize legitimate negative sentiment, not just toxic content

---

**Document:** Week 3 Lab - DeepLearning.AI Platform

**Course:** Generative AI with LLMs

**Focus:** RLHF Detoxification with Proximal Policy Optimization
