# Week 1 Lab Findings: Dialogue Summarization with LLMs
## Deep Learning AI Platform - Agentic AI with LLMs Course

---

## Executive Summary

This document presents experimental findings from dialogue summarization tasks using various prompt engineering techniques and generative configuration parameters. The main finding across all experiments is that **prompt engineering approaches (Zero-Shot, One-Shot, Few-Shot) consistently underperformed or failed** when compared to baseline human summaries. The model exhibited persistent issues with speaker identification and content accuracy across different prompting strategies.

---

## Table of Contents

1. [Zero-Shot Inference](#zero-shot-inference)
2. [One-Shot Inference](#one-shot-inference)
3. [Few-Shot Inference](#few-shot-inference)
4. [Generative Configuration Parameters](#generative-configuration-parameters)

---

## Zero-Shot Inference

### Original Prompt Tests

#### Example 1: Train Conversation

**INPUT PROMPT:**
```
Summarize the following conversation.

#Person1#: What time is it, Tom?
#Person2#: Just a minute. It's ten to nine by my watch.
#Person1#: Is it? I had no idea it was so late. I must be off now.
#Person2#: What's the hurry?
#Person1#: I must catch the nine-thirty train.
#Person2#: You've plenty of time yet. The railway station is very close. It won't take more than twenty minutes to get there.

Summary:
```

**BASELINE HUMAN SUMMARY:**
```
#Person1# is in a hurry to catch a train. Tom tells #Person1# there is plenty of time.
```

**MODEL GENERATION - ZERO SHOT:**
```
The train is about to leave.
```

---

#### Example 2: System Upgrade Conversation

**INPUT PROMPT:**
```
Summarize the following conversation.

#Person1#: Have you considered upgrading your system?
#Person2#: Yes, but I'm not sure what exactly I would need.
#Person1#: You could consider adding a painting program to your software. It would allow you to make up your own flyers and banners for advertising.
#Person2#: That would be a definite bonus.
#Person1#: You might also want to upgrade your hardware because it is pretty outdated now.
#Person2#: How can we do that?
#Person1#: You'd probably need a faster processor, to begin with. And you also need a more powerful hard disc, more memory and a faster modem. Do you have a CD-ROM drive?
#Person2#: No.
#Person1#: Then you might want to add a CD-ROM drive too, because most new software programs are coming out on Cds.
#Person2#: That sounds great. Thanks.

Summary:
```

**BASELINE HUMAN SUMMARY:**
```
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
```

**MODEL GENERATION - ZERO SHOT:**
```
#Person1#: I'm thinking of upgrading my computer.
```

---

### Zero-Shot Personal Prompting Experiments

#### TEST 1: Basic Summarization Prompt
*Prompt: "Briefly summarize the following dialogue:"*

**Example 1 - MODEL GENERATION:**
```
Tom is late for the train. He has to catch the nine-thirty train.
```

**Example 2 - MODEL GENERATION:**
```
#Person1#: Have you considered upgrading your system? #Person2#: Yes, but I'm not sure what exactly I would need. #Person1#: You could consider adding a painting program to your software
```

---

#### TEST 2: Contextual Summarization with Key Points
*Prompt: "Summarize the key points from this dialogue in 2-3 sentences:"*

**Example 1 - MODEL GENERATION:**
```
The train is about to leave Tom's place.
```

**Example 2 - MODEL GENERATION:**
```
#Person1: You could add a painting program to your software. #Person2: You could also upgrade your hardware. #Person1: You might also want to add a CD-ROM drive.
```

---

#### TEST 3: Action-Oriented Summary
*Prompt: "What are the main topics discussed in this dialogue? Provide a brief summary:"*

**Example 1 - MODEL GENERATION:**
```
Tom is late for work.
```

**Example 2 - MODEL GENERATION:**
```
#Person1: You might want to upgrade your system. #Person2: You might want to upgrade your hardware. #Person1: You might want to upgrade your computer.
```

---

#### TEST 4: Concise One-Line Summary
*Prompt: "Provide a one-sentence summary of this dialogue:"*

**Example 1 - MODEL GENERATION:**
```
Tom is late for the train.
```

**Example 2 - MODEL GENERATION:**
```
#Person1 wants to upgrade his computer. #Person2 wants to upgrade his hardware.
```

---

#### TEST 5: Speaker-Focused Summary
*Prompt: "Summarize what the speakers discussed and any agreements or decisions made:"*

**Example 1 - MODEL GENERATION:**
```
The speakers discussed the timetable for the train.
```

**Example 2 - MODEL GENERATION:**
```
#Person1: I'm thinking of upgrading my computer. #Person2: I'm not sure what exactly I would need. #Person1: I'd probably need a painting program. #Person2:
```

---

#### TEST 6: Bullet-Point Summary
*Prompt: "Create a bullet-point summary of the main points from this dialogue:"*

**Example 1 - MODEL GENERATION:**
```
Tom is late for the train. He has to catch the nine-thirty train.
```

**Example 2 - MODEL GENERATION:**
```
The person who is considering upgrading their computer is asking about what they need.
```

---

#### TEST 7: Technical/Business Summary
*Prompt: "Summarize this dialogue as a professional summary of key decisions and action items:"*

**Example 1 - MODEL GENERATION:**
```
The train is about to leave, but Tom is late.
```

**Example 2 - MODEL GENERATION:**
```
#Person1#: You could consider adding a painting program to your software. #Person2#: That would be a bonus. #Person1#: You might also want to upgrade your hardware. #Person1
```

---

### Analysis: Zero-Shot Performance

**Key Findings:**
- The model consistently failed to identify speakers correctly, particularly confusing "#Person1#" with "Tom" in Example 1
- Summaries were often incomplete, providing only partial information rather than comprehensive overviews
- In Example 2, the model frequently provided verbatim excerpts or lists of dialogue content instead of actual summaries
- Varying the prompt structure (basic, contextual, action-oriented, etc.) did not significantly improve performance
- The model showed a tendency to either oversimplify (missing key details) or regurgitate dialogue fragments (failing to synthesize)

---

## One-Shot Inference

### Experimental Setup

**One-Shot Example Provided:**
```
Dialogue:

#Person1#: What time is it, Tom?
#Person2#: Just a minute. It's ten to nine by my watch.
#Person1#: Is it? I had no idea it was so late. I must be off now.
#Person2#: What's the hurry?
#Person1#: I must catch the nine-thirty train.
#Person2#: You've plenty of time yet. The railway station is very close. It won't take more than twenty minutes to get there.

What was going on?
#Person1# is in a hurry to catch a train. Tom tells #Person1# there is plenty of time.
```

**Test Dialogue:**
```
Dialogue:

#Person1#: Have you considered upgrading your system?
#Person2#: Yes, but I'm not sure what exactly I would need.
#Person1#: You could consider adding a painting program to your software. It would allow you to make up your own flyers and banners for advertising.
#Person2#: That would be a definite bonus.
#Person1#: You might also want to upgrade your hardware because it is pretty outdated now.
#Person2#: How can we do that?
#Person1#: You'd probably need a faster processor, to begin with. And you also need a more powerful hard disc, more memory and a faster modem. Do you have a CD-ROM drive?
#Person2#: No.
#Person1#: Then you might want to add a CD-ROM drive too, because most new software programs are coming out on Cds.
#Person2#: That sounds great. Thanks.

What was going on?
```

**BASELINE HUMAN SUMMARY:**
```
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
```

**MODEL GENERATION - ONE SHOT:**
```
#Person1 wants to upgrade his system. #Person2 wants to add a painting program to his software. #Person1 wants to add a CD-ROM drive.
```

---

### Analysis: One-Shot Performance

**Key Findings:**
- Despite providing a clear example of the desired summary format, the model failed to capture the teaching/advisory relationship between speakers
- The model incorrectly attributed desires to both speakers (suggesting both wanted upgrades) when actually #Person1# was advising #Person2#
- The summary fragmented the conversation into separate points rather than synthesizing the overall interaction
- One-shot learning did not resolve the fundamental issue of understanding speaker roles and dialogue dynamics

---

## Few-Shot Inference

### Experimental Setup

Multiple examples were provided to guide the model's summarization behavior.

---

### TEST 1a: Three Example Dialogues [7, 71, 198]

**Examples Provided:**
1. Divorce discussion between Kate and #Person1#
2. Nobel Prize announcement to Tom
3. System upgrade advice (same dialogue with different summary)

**BASELINE HUMAN SUMMARY:**
```
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
```

**MODEL GENERATION - FEW SHOT:**
```
#Person1 is giving #Person2# some advice for upgrading #Person2#'s system, such as adding a painting program and a faster processor.
```

**Note:** This test performed well because dialogue 198 was the same conversation with a slightly different human summary, effectively providing direct guidance.

---

### TEST 1b: Three Different Dialogues [10, 100, 150]

**Examples Provided:**
1. Brian's birthday party
2. Director and actor discussing scene performance
3. Taxi ride to Friendship Hotel

**BASELINE HUMAN SUMMARY:**
```
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
```

**MODEL GENERATION - FEW SHOT:**
```
#Person1 recommends upgrading their system.
```

---

### TEST 2a: Four-Shot Learning [9, 71, 110, 183]

**Examples Provided:**
1. Birthday party dialogue
2. Nobel Prize announcement
3. Nuclear weapons demonstration discussion
4. Weather conversation in Beijing

**BASELINE HUMAN SUMMARY:**
```
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
```

**MODEL GENERATION - FEW SHOT:**
```
The computer system of Person1 is outdated.
```

---

### TEST 2b: Five-Shot Learning [7, 40, 116, 156, 182]

**Examples Provided:**
1. Divorce discussion
2. Train schedule conversation
3. Study notes borrowing
4. Company neighborhood information
5. Favorite classes discussion

**BASELINE HUMAN SUMMARY:**
```
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
```

**MODEL GENERATION - FEW SHOT:**
```
#Person1 recommends upgrading the system and hardware.
```

---

### Analysis: Few-Shot Performance

**Key Findings:**
- Few-shot learning showed marginal improvement over zero-shot and one-shot approaches, but still failed to produce accurate summaries
- Increasing the number of examples (3 → 4 → 5 shots) did not consistently improve performance and sometimes degraded results
- The most successful case (TEST 1a) succeeded because it essentially included the target dialogue in the examples, which is not a generalizable solution
- With diverse example dialogues, the model struggled to extract the correct pattern and continued to misidentify speaker roles and dialogue intent
- The quality of few-shot performance appears highly dependent on example selection rather than demonstrating true pattern learning

---

## Generative Configuration Parameters for Inference

Configuration parameters were adjusted to observe their impact on output quality and consistency.

---

### TEST 1: Default Few-Shot with Limited Tokens

**Configuration:**
```python
generation_config = GenerationConfig(max_new_tokens=50)
```

**MODEL GENERATION:**
```
#Person1 recommends upgrading the system and hardware.
```

**BASELINE HUMAN SUMMARY:**
```
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
```

---

### TEST 2: Severely Limited Tokens

**Configuration:**
```python
generation_config = GenerationConfig(max_new_tokens=10)
```

**MODEL GENERATION:**
```
#Person1 recommends upgrading the system and
```

**BASELINE HUMAN SUMMARY:**
```
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
```

**Observation:** Choosing `max_new_tokens=10` makes the output too short, resulting in truncated summaries.

---

### TEST 3: Low Temperature (Conservative)

**Configuration:**
```python
generation_config = GenerationConfig(
    max_new_tokens=50,
    do_sample=True,
    temperature=0.1
)
```

**MODEL GENERATION:**
```
#Person1 wants to upgrade his system. #Person2 wants to add a painting program to his software. #Person1 wants to upgrade his hardware.
```

**BASELINE HUMAN SUMMARY:**
```
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
```

---

### TEST 4: Medium Temperature

**Configuration:**
```python
generation_config = GenerationConfig(
    max_new_tokens=50,
    do_sample=True,
    temperature=0.5
)
```

**MODEL GENERATION:**
```
#Person2 explains how to upgrade his system and hardware.
```

**BASELINE HUMAN SUMMARY:**
```
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
```

---

### TEST 5: High Temperature (Maximum Creativity)

**Configuration:**
```python
generation_config = GenerationConfig(
    max_new_tokens=50,
    do_sample=True,
    temperature=1.0
)
```

**MODEL GENERATION:**
```
Some things to consider is: a painting program, a hard disk, more memory and a faster modem.
```

**BASELINE HUMAN SUMMARY:**
```
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
```

---

### Custom Configuration Tests

#### TEST 1: Conservative Approach

**Configuration:**
```python
generation_config = GenerationConfig(
    max_new_tokens=50,
    do_sample=False,  # Deterministic results
    no_repeat_ngram_size=2,  # Prevents repetition
    early_stopping=True
)
```

**MODEL GENERATION:**
```
#Person1 recommends upgrading the system and hardware.
```

---

#### TEST 2: Creative Configuration

**Configuration:**
```python
generation_config = GenerationConfig(
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    no_repeat_ngram_size=2,
    early_stopping=True
)
```

**MODEL GENERATION:**
```
Person1 offers to upgrade their system.
```

---

#### TEST 3: Maximum Variation

**Configuration:**
```python
generation_config = GenerationConfig(
    max_new_tokens=50,
    do_sample=True,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    no_repeat_ngram_size=2,
    early_stopping=True
)
```

**MODEL GENERATION:**
```
Person1 suggested upgrading their system, and she would consider adding software which allowed the user to make up their own flyers and banners.
```

**BASELINE HUMAN SUMMARY:**
```
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
```

---

### Analysis: Generative Configuration Impact

**Key Findings:**
- Configuration parameters showed modest impact on output quality compared to prompt engineering approaches
- Higher temperature values (0.7-1.0) introduced creative variation but also increased the likelihood of gender misattribution (e.g., "she" instead of neutral pronouns)
- At temperature=1.0, the model began making assumptions about speaker characteristics not present in the dialogue
- Conservative configurations (temperature=0.1, do_sample=False) produced more consistent but still inaccurate summaries
- Token limitations directly affected summary completeness, with max_new_tokens=10 causing truncation
- Repetition controls (no_repeat_ngram_size) helped prevent redundant phrasing but did not improve content accuracy

**Performance by Temperature:**
- **Low (0.1):** Conservative, consistent speaker role confusion
- **Medium (0.5):** Moderate creativity, reversed speaker roles in some cases
- **High (1.0):** Maximum variation, introduced gender assumptions and content fabrication

---

## Configuration Parameters Reference

| Parameter | Value | Effect |
|-----------|-------|--------|
| `max_new_tokens` | 50 | Limits output to 50 new tokens (appropriate for summaries) |
| `max_new_tokens` | 10 | Too restrictive, causes truncation |
| `do_sample` | False | Deterministic output (same result every time) |
| `do_sample` | True | Probabilistic sampling (varied outputs) |
| `temperature` | 0.1-0.5 | Conservative generation, more similar outputs |
| `temperature` | 0.7 | Balanced creativity and coherence |
| `temperature` | 1.0 | Maximum creativity/variation, risk of hallucination |
| `no_repeat_ngram_size` | 2 | Prevents repeating 2-word phrases |
| `top_k` | 50 | Samples from top 50 most likely next tokens |
| `top_p` | 0.95 | Nucleus sampling: smallest set with 95% cumulative probability |
| `early_stopping` | True | Stops when model outputs end token |

---

## Overall Conclusions

### Primary Findings

1. **Prompt Engineering Limitations:** Zero-shot, one-shot, and few-shot prompting all failed to produce accurate summaries comparable to human baselines. The fundamental issue was not the lack of examples but the model's inability to correctly identify speaker roles and dialogue dynamics.

2. **Persistent Speaker Identification Issues:** Across all experiments, the model consistently:
   - Failed to correctly identify "Tom" as #Person2# in Example 1
   - Confused which speaker was giving advice versus receiving it in Example 2
   - Attributed actions and intentions to the wrong speakers

3. **Content vs. Summary Confusion:** The model frequently:
   - Reproduced dialogue fragments instead of generating summaries
   - Listed conversation topics rather than synthesizing the interaction
   - Missed the relational dynamic (teaching, advising, asking) between speakers

4. **Few-Shot Learning Dependency:** Few-shot performance was highly dependent on example selection. When the target dialogue appeared in the examples (TEST 1a), performance improved, but this does not represent genuine pattern learning.

5. **Configuration Parameter Effects:**
   - Temperature adjustments affected creativity but introduced new problems (gender assumptions at high values)
   - Token limits directly impacted completeness but not accuracy
   - Conservative settings produced consistent errors; creative settings produced varied errors

### Implications for Week 1 Testing

These findings suggest that for dialogue summarization tasks:
- Simple prompt engineering may be insufficient without model fine-tuning or more sophisticated techniques
- Speaker role identification requires either better contextual understanding or explicit role labeling
- Configuration parameters can modify output characteristics but cannot compensate for fundamental comprehension issues
- Future experiments should explore: fine-tuning, chain-of-thought prompting, or structured output formats

---

## Recommended Testing Strategy

For continued experimentation:

```python
# Baseline: Conservative and reproducible
generation_config = GenerationConfig(
    max_new_tokens=50,
    do_sample=False
)

# Add repetition control
generation_config = GenerationConfig(
    max_new_tokens=50,
    do_sample=False,
    no_repeat_ngram_size=2
)

# Balanced creativity
generation_config = GenerationConfig(
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    no_repeat_ngram_size=2
)
```

---

**Document Created:** Week 1 Lab - Deep Learning AI Platform
**Course:** Agentic AI with LLMs
**Focus:** Dialogue Summarization Performance Analysis
