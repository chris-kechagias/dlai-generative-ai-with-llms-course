# DeepLearning.AI: Generative AI with Large Language Models

Lab reports and findings from the DeepLearning.AI course on LLM applications.

## About

I’m documenting my learning journey through this course. The platform doesn’t allow code sharing, so I’m publishing my experimental findings and analysis instead.

## Course Structure

- **Week 1:** Transformer architecture, prompt engineering, generative configuration
- **Week 2:** Fine-tuning and evaluation
- **Week 3:** Reinforcement learning and LLM-powered applications (upcoming)

## Lab Reports

### Week 1: Dialogue Summarization Experiments

[Full report →](./week1-lab-report-prompt-engineering-experiments.md)

Key findings:

- Zero-shot, one-shot, and few-shot prompting all struggled with speaker identification
- Temperature settings affected creativity but didn’t fix core comprehension issues
- Configuration parameters (top_k, top_p, temperature) showed limited impact compared to fine-tuning needs

Main takeaway: Prompt engineering alone isn’t enough for complex dialogue tasks. Fine-tuning or chain-of-thought prompting needed.

### Week 2: Fine-Tuning with PEFT and LoRA

[Full report →](./week2-lab-report-fine-tuning-dialogue-summarization.md)

Key findings:

- PEFT with LoRA achieved near-equivalent results to full fine-tuning
- Trained less than 1% of parameters vs 100% for full fine-tuning
- Model checkpoint size reduced from ~945 MB to a few MBs (adapter only)
- Zero-shot FLAN-T5 still struggled despite instruction-tuning

Main takeaway: You don’t need enterprise-level GPUs to fine-tune LLMs. LoRA makes it practical for individual developers.

## Background

This is part of my career transition from retail operations to AI engineering. Building practical LLM applications while working full-time.

Timeline: Nov 2025 → Sep 2026

## Connect

- LinkedIn: [Chris Kechagias](https://www.linkedin.com/in/chkechagias)
- Portfolio: [In progress]

-----

Course by DeepLearning.AI - Auditing free, certificates purchased upon completion.