# IDEM: The IDioms with EMotions Dataset for Emotion Recognition

IDEM is a novel dataset in the field of Emotion Recognition.

Language is rich with idioms, expressions that carry weight beyond their literal meanings. While these are widely used and readily understood by humans, they can prove a significant hurdle for NLP models. Our project tackles this gap by providing a dataset of idiom-containing sentences labelled with one of 36 emotion types. 

The IDEM dataset consists of 8729 train sentences and 956 test sentences generated with the help of GPT-4.

In this repository, you will also find the baseline implementations of Transformer-based emotion recognition approaches, evaluated on the IDEM dataset. The best-performing model, a fine-tuned RoBERTa sequence classifier, achieved a weighted F1-score of 58.73%.

Additionally, we also present a solution to the problem of automatic idiom identification in sentences. 

We encourage you to train your own models using IDEM. Together, let's advance the understanding of emotions in idiomatic language.


## Repository structure
```
├── code  
│   ├── dataset_generation      <- code that generated the idem dataset
│   │   └── evaluations         <- results of manual evaluation
│   ├── experiments             <- performance baselines on the IDEM datatset
│   │   ├── bert  
│   │   └── gpt4  
│   ├── idiom_finding           <- identify idioms in text
│   └── SLIDE_dataset           <- the original SLIDE idiom lexicon from which we generated sentences
├── dataset                     <- our idem dataset alongside its idiom lexicon
```