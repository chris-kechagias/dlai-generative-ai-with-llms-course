# DeepLearning.AI: Generative AI with Large Language Models

Lab reports and findings from the DeepLearning.AI course on LLM applications.

## About

I'm documenting my learning journey through this course. The platform doesn't allow code sharing, so I'm publishing my experimental findings and analysis instead.

## Course Structure

- **Week 1:** Transformer architecture, prompt engineering, generative configuration
- **Week 2:** Fine-tuning with PEFT/LoRA and evaluation metrics
- **Week 3:** Reinforcement Learning from Human Feedback (RLHF) and model alignment

## Lab Reports

### Week 1: Dialogue Summarization Experiments

[Full report →](./week1-lab-report-prompt-engineering-experiments.md)

Key findings:
- Zero-shot, one-shot, and few-shot prompting all struggled with speaker identification
- Temperature settings affected creativity but didn't fix core comprehension issues
- Configuration parameters (top_k, top_p, temperature) showed limited impact compared to fine-tuning needs

Main takeaway: Prompt engineering alone isn't enough for complex dialogue tasks. Fine-tuning or chain-of-thought prompting needed.

### Week 2: Fine-Tuning with PEFT/LoRA

[Full report →](./week2-lab-report-fine-tuning-dialogue-summarization.md)

Key findings:
- Parameter-Efficient Fine-Tuning (PEFT) with LoRA enabled training with only 1.41% of parameters
- Instruction fine-tuning significantly improved summary quality over prompt engineering
- ROUGE metrics provided quantitative evaluation of summary improvements

Main takeaway: PEFT/LoRA offers an efficient path to model customization without full fine-tuning costs.

### Week 3: RLHF Detoxification with PPO

[Full report →](./week3-lab-report-rlhf-detoxification.md)

Key findings:
- RLHF with PPO achieved 47.5% reduction in mean toxicity scores
- RoBERTa hate speech classifier served as automated reward model, replacing expensive human labeling
- Only 10 PPO training steps needed with 1.41% trainable parameters (PEFT efficiency preserved)
- 75% of samples improved, though 25% showed degradation (reward hacking trade-off)

Main takeaway: RLHF can efficiently align models for specific objectives like toxicity reduction, but careful reward model selection is critical to avoid unintended optimization.

## Background

This is part of my career transition from retail operations to AI engineering. Building practical LLM applications while working full-time.

Timeline: Nov 2025 → Sep 2026

## Connect

- **LinkedIn:** [linkedin.com/in/chkechagias](https://www.linkedin.com/in/chkechagias)
- **GitHub:** [github.com/chris-kechagias](https://github.com/chris-kechagias)
- **Email:** ck.chris.kechagias@gmail.com
- **Location:** Thessaloniki, Greece
- **Open to:** Remote AI Engineer positions (EU/Worldwide)
- **Portfolio:** [In progress]

---

Course by DeepLearning.AI
