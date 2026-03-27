# Metric Development

This directory contains the human-labeled judge validation datasets for LLM and LALM judge metrics. 

The judge scores on the validation datasets are below. For each dataset, we run the judge model for that metric 3 times and report the average score and standard deviation.
Judge accuracy is reported alongside macro-averaged F1 score. 
- Accuracy measures the overall proportion of samples where the judge's predicted label matches the ground truth label. 
- Macro F1 is the unweighted average of the per-class F1 scores, treating each label equally regardless of how frequently it appears in the dataset. 

Together, these two metrics capture both overall correctness and per-class balance — accuracy reflects aggregate performance, while macro F1 penalizes the judge for weak performance on any individual label, including minority classes.
We plan to expand these datasets to be less specific to the airline domain and to improve the class balance by adding more samples.


Faithfulness - Claude Opus 4.6
- **Dataset size:** 137 records
- **Class distribution:** Rating 1: 57, Rating 2: 28, Rating 3: 52 
- **accuracy:** 83.9% ±2.9%
- **macro f1:** 80.7% ±2.9%


Agent Speech Fidelity - Gemini 3.1 Pro (low thinking)
- **Dataset size:** 147 records
- **Class distribution:** Rating 0: 122, Rating 1: 25
- **accuracy:** 89.6% ±2.6%
- **macro f1:** 85.6% ±2.4%



Conciseness - GPT-5.2
- **Dataset size:** 100 records (479 rated turns across records)
- **Class distribution:** Rating 1: 93, Rating 2: 49, Rating 3: 337
- **accuracy:** 92.3% ±0.8%
- **macro f1:** 83.8% ±1.1%


Conversation Progression - GPT-5.2
- **Dataset size:** 136 records
- **Class distribution:** Rating 1: 23, Rating 2: 58, Rating 3: 55
- **accuracy:** 79.9% ±1.1%
- **macro f1:** 78.2% ±1.3%


Transcription Accuracy Key Entities - GPT-5.2
- **Dataset size:** 38 records (567 entities across records)
- **Class distribution:** Correct: 386, Incorrect: 181
- **entity_f1_lenient:** 94.5% ±0.3%
- **correctness_with_penalty:** 86.2% ±0.8%
  
- entity_f1_lenient measures how well the model identifies the correct entities by comparing extracted text against ground truth using lenient matching. Text is normalized (lowercased, whitespace-compressed, punctuation-stripped), then two strings match if either contains the other or if their sequence similarity exceeds 0.6 (computed via SequenceMatcher.ratio(), which scores positional character overlap on a 0–1 scale). F1 is the harmonic mean of precision and recall over matched entities. 
- correctness_with_penalty extends lenient entity matching by penalizing unmatched extractions — any predicted entity that doesn't match a ground truth entity counts against the score. This captures cases where recall is high but precision is lower, reflecting the cost of spurious extractions.
