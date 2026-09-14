# SynthAug-Bench

## When Does Diffusion-Based Synthetic Data Augmentation Help Computer Vision Classifiers?

### A Benchmark Across Generator Fidelity, Data-Scarcity Regimes, and CNN/Transformer Architectures

**Original proposal and advisor:** Dr. Amir Jafari  
**Program:** Data Science, The George Washington University, Washington, DC

This README follows the original proposal and incorporates the advisor's feedback on object detection, image captioning, pretrained features, and hybrid modeling. It describes planned work. Dataset preparation, code, and experimental results will be added in later stages.

## 1. Objective

The goal of this project is to build a reproducible benchmark that examines when synthetic data augmentation improves computer vision models. We will compare conventional augmentation, GANs, diffusion models trained from scratch, and pretrained diffusion models with and without adaptation. The main question is whether improvements in generated-image quality lead to better downstream performance, or whether the benefit plateaus or reverses.

We will study this question across controlled few-shot, constructed long-tail, fine-grained, and naturally imbalanced medical datasets. We will also examine whether CNN and Transformer classifiers respond differently to the same synthetic training data.

The original classification benchmark remains required. Based on the advisor's feedback, the project will also evaluate object detection and image captioning, investigate pretrained features and auxiliary information through a defined hybrid comparison, and compare augmentation benefits across tasks.

### Key objectives

1. Implement the five original augmentation families: conventional augmentation, class-conditional StyleGAN2-ADA, a small class-conditional DDPM, pretrained Stable Diffusion with class prompts and textual-inversion tokens, and LoRA-adapted Stable Diffusion with a defined DreamBooth-style objective.
2. Complete the four classification data-scarcity tiers, preserving every named dataset and regime in the original proposal.
3. Compare ResNet-50, ConvNeXt-Tiny, ViT-B/16, and Swin-T on real-only, conventionally augmented, and real-plus-synthetic training data.
4. Measure generated-sample quality with FID and Improved Precision & Recall, then evaluate whether these measurements predict downstream benefit.
5. Release a config-driven benchmark with experiment records, reproducible notebooks, and a single-GPU reproduction guide.
6. Evaluate generative augmentation for object detection and image captioning with valid task-specific annotations and controls.
7. Investigate pretrained visual features, auxiliary information, and a defined hybrid approach against matched controls.
8. Compare improvements, failures, and computational costs across the three tasks.

## 2. Datasets

The classification study retains the four tiers in the original proposal. Detection and captioning require additional task-specific data. Dataset versions, access terms, preprocessing, and split procedures will be documented before use.

| Tier | Dataset | Planned setting |
|---|---|---|
| A: Controlled few-shot | [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) | 5, 10, 20, and 50 training images per class; preserve the official test set |
| B: Controlled long-tail | [CIFAR-100-LT](https://github.com/richardaecn/class-balanced-loss#datasets), constructed from CIFAR-100 | Exponential long-tail splits with imbalance ratios of 100, 50, and 10; include a reweighting comparison |
| C: Fine-grained low-data | [Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | Documented low-data training, validation, and test protocol |
| C: Fine-grained low-data | [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) | Documented low-data training, validation, and test protocol |
| C: Fine-grained low-data | [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/) | Documented low-data training, validation, and test protocol |
| D: Naturally imbalanced medical data | [HAM10000](https://doi.org/10.7910/DVN/DBW86T) | Skin-lesion classification; keep images of the same lesion in one split |
| Detection extension | Dataset to be finalized; COCO is a candidate | Images with object categories and bounding boxes |
| Captioning extension | Dataset to be finalized; COCO is a candidate | Images with reference captions |
### Data and generator preparation

We will use fixed training, validation, and test splits, with recorded seeds for few-shot and long-tail sampling. Classification loaders will share a common interface. Detection and captioning loaders will retain their task-specific annotations while using the same split and provenance records.

For each dataset and generator condition, we will record the real and synthetic sample counts, synthetic:real ratio, fitting time, generation time, preprocessing, prompts, checkpoint revision, and quality measurements. Generator adaptation and synthetic training samples will use training data only. Validation data will support model and parameter selection; final test data will remain held out.

Preprocessing will be documented in a notebook for each tier. HAM10000 preparation will include lesion-group and duplicate checks. Detection samples will be checked for valid boxes and object labels. Captioning samples will be checked for agreement between the image and its description.

## 3. Rationale

Limited training data can make it difficult to learn rare classes, distinguish similar categories, or generalize to a different image domain. Synthetic data offers a way to expand the training set, but more images do not automatically provide more useful information. Generated examples can repeat existing patterns, change class-defining details, or introduce incorrect annotations.

SynthAug-Bench will examine these effects under a shared experimental protocol. The study will connect sample-quality measurements with downstream performance, architectural differences, data scarcity, and computational cost. The related-work review will cover the studies named in the original proposal, including DA-Fusion, ALIA, StableRep, Azizi et al., He et al., and Sariyildiz et al., with their exact settings and any published corrections recorded.

HAM10000 provides a required domain-mismatch case study. We will compare pretrained and adapted generation without assuming that adaptation must improve results or that a pretrained model has never encountered similar images.

Detection and captioning extend the question beyond classification. A generated image may preserve its category while changing object position, count, or attributes. These tasks let us examine whether an augmentation method remains useful when training depends on spatial annotations or descriptive language. The feature and hybrid study will test whether pretrained representations provide additional information under a controlled comparison.

## 4. Approach

### Phase 1: Foundations and baselines

Prepare the four-tier classification pipeline, the detection and captioning data interfaces, and the experiment tracking structure. Create a generator registry and a complete run matrix before the main experiments. Record dataset, regime, generator, classifier, synthetic ratio, seed, status, and cost for each applicable comparison.

Implement real-only training, RandAugment, Mixup, and CutMix. Evaluate the three named conventional policies with separately recorded settings and results. Train all four classification architectures and include the long-tail reweighting comparison. Establish task metrics, FID, Improved Precision & Recall, and the statistical reporting protocol.

The planned software includes PyTorch, Hugging Face diffusers/transformers/accelerate/peft, timm, StyleGAN2-ADA, and clean-fid. Exact dependency versions will be pinned after compatibility checks. The original hardware target is one NVIDIA A10G with 24 GB VRAM on AWS g5.2xlarge. Runtime and memory must be measured before claiming that the full study fits the available budget.

### Phase 2: GAN baseline

Train class-conditional StyleGAN2-ADA from scratch on the low-data splits. Begin with the original methods' Tier A/B experiments at 32x32 or 64x64 resolution. Monitor training stability, sample diversity, and possible collapse.

Generate synthetic sets at ratios of 1:1, 2:1, and 5:1 relative to real training data. Retrain all four classifiers for the applicable settings and record FID, precision/recall, downstream performance, and cost.

### Phase 3: Diffusion trained from scratch

Implement a small class-conditional DDPM with a lightweight U-Net and a noise-prediction objective. Use the same initial Tier A/B scope and resolution as the GAN comparison, with DDIM sampling to reduce generation time.

Generate data at the same three ratios and retrain the four classifiers. Compare GAN and DDPM results under documented, matched from-scratch compute budgets. Measure fidelity rather than assuming that one generator family must produce better samples.

### Phase 4: Pretrained foundation diffusion

Implement class-prompt augmentation with pretrained Stable Diffusion and a separate DA-Fusion-style condition using learned textual-inversion tokens. The diffusion backbone can remain frozen while token embeddings are fitted, so learned-token augmentation will not be labelled entirely training-free.

Reproduce DA-Fusion using a matching published dataset and protocol before extending it to the required classification settings, including fine-grained and medical data. Flowers and Pets remain required project datasets; they will not be called reproduction datasets without a matching published protocol.

### Phase 5: Adapted foundation diffusion

Adapt Stable Diffusion across Tiers A-D using LoRA and an explicit DreamBooth-style class or subject-binding objective. Record the trainable components, LoRA rank, adaptation data, and optimization settings. Use the original rank range of 4-16 as planned settings to investigate.

Measure memory and runtime with mixed precision and gradient accumulation where supported. Evaluate the proposed SD-Turbo/LCM acceleration options as separately recorded configurations so a checkpoint or adapter change is not hidden inside a sampler comparison.

Generate the required synthetic ratios and complete downstream comparisons across all four classifiers. The HAM10000 analysis will test whether adaptation changes the usefulness of generated data in the medical domain.

### Required task and feature extensions

Object detection and image captioning will each include real-only and conventional controls, generative comparisons, repeated training runs, and evaluation on held-out real images. Detection augmentation must preserve or regenerate correct boxes and account for added or removed objects. Captioning augmentation must preserve or verify the objects, attributes, counts, and relationships described by its labels.

The feature study will define the pretrained encoder, feature sources, frozen and trainable components, and added computational cost. We will compare real-image features with real-plus-synthetic features and evaluate a specified hybrid combination against the same task model without the added information. Exact detector, captioner, encoder, and fusion choices remain to be finalized.

These additions will be developed alongside the classification phases. They do not replace any original dataset, classifier, method, ratio, or repeated comparison. Their exact generator coverage will be recorded explicitly; the recording does not specify every generator on every task.

### Phase 6: Cross-condition analysis

Assemble the master benchmarking DataFrame and evaluate the original four hypotheses.

| Hypothesis | Question |
|---|---|
| H1 | Does downstream benefit increase with measured generator fidelity, plateau, or reverse? |
| H2 | Do the tested CNN and Transformer architectures benefit differently from synthetic augmentation? |
| H3 | Do FID and Improved Precision & Recall predict downstream performance gains? |
| H4 | Which dataset, generator, and synthetic-ratio settings reduce performance relative to conventional augmentation? |

Use five downstream training seeds and report means, standard deviations, bootstrapped confidence intervals, and paired changes from controls. Record data-selection and generation seeds separately. Keep training budgets and sampling policies comparable so additional optimizer steps do not become an unreported advantage.

Classification will report accuracy and class-sensitive measures. Detection will use task-appropriate AP measurements, and captioning will use defined caption metrics and factual-consistency checks. Exact evaluation implementations will be fixed before the main runs. Analyze each task against its own controls rather than pooling incompatible scores.

Evaluate quality-to-utility regression within compatible settings, then test any claimed predictive relationship on held-out settings. Report failed and inconclusive experiments alongside improvements. Both students will independently check the H1-H4 analysis.

### Phase 7: Guidelines, report, and code release

Produce practical generator-selection guidelines based on data scarcity and measured compute. Prepare the original figure set: performance versus measured fidelity, FID/precision-recall versus performance gain, CNN versus Transformer benefit, and the HAM10000 domain comparison. Add the detection, captioning, and feature/hybrid findings.

Write the 8-10-page paper-format report with motivation, related work, methods, H1-H4 results, the medical analysis, cross-task findings, practical guidance, and limitations. Detection and captioning belong in the current study; segmentation retains its original future-work status.

Release the config-driven benchmark with the generator registry, complete run matrix, experiment records, a full-study execution command, notebooks per phase and tier, a cross-condition analysis notebook, pinned tested dependencies, README, and single-GPU setup and reproduction instructions. Prepare the demonstration and final presentation. The original paper-submission milestone remains to be scheduled with the advisor; journal submission planning is outside the current work stage.

## 5. Timeline

The original target is 16 weeks. The task and feature extensions are integrated into that sequence. Calendar dates and resource requirements will be established from actual access and pilot measurements.

| Weeks | Original study | Integrated feedback work |
|---|---|---|
| 1-3 | Environment, all classification tiers, four-classifier controls, registry, full matrix, metrics, and five-seed protocol | Detection/captioning data interfaces and controls; define the feature/hybrid comparison |
| 4-5 | StyleGAN2-ADA training, generation, and classifier comparisons | Validate detection boxes, caption consistency, and task evaluation |
| 6-7 | Small DDPM training and matched GAN/DDPM comparisons | Prepare task-specific generation and provenance checks |
| 8-9 | DA-Fusion reproduction and pretrained augmentation, including fine-grained and medical data | Run pretrained augmentation comparisons for detection and captioning |
| 10-11 | LoRA/DreamBooth-style adaptation and required classifier comparisons | Complete adapted-generation and feature/hybrid comparisons |
| 12-13 | Master benchmark, H1-H4, regression, and uncertainty analysis | Compare benefits, failures, and costs across tasks |
| 14 | Practical guidelines and original final figures | Task and feature findings in the figures and analysis |
| 15-16 | Paper-format report, notebooks, code release, and final presentation | Integrated report and demonstration covering all three tasks |

Small pilot runs will check the pipeline and measure cost. They change execution order only. The full dataset, method, architecture, ratio, and five-seed requirements remain. Any resource shortfall will be recorded and addressed through scheduling and resource planning. The final review retains the original three-day code freeze.

## 6. Team responsibilities

| Responsibility | Student 1 | Student 2 |
|---|---|---|
| Original method ownership | Conventional augmentation, StyleGAN2-ADA, small DDPM, FID/precision-recall, generator registry | Pretrained diffusion, textual inversion, LoRA/DreamBooth-style adaptation, four-classifier benchmarking, master result table |
| Task extensions | Detection pipeline and annotation verification | Captioning pipeline and image-caption verification |
| Shared work | Feature/hybrid study, split checks, statistical cross-checks, guidelines, writing, documentation, and presentation | Feature/hybrid study, split checks, statistical cross-checks, guidelines, writing, documentation, and presentation |

Hold weekly integration meetings so every method uses the same recorded training and evaluation protocol. Both students should be able to reproduce and explain the complete pipeline.

## 7. Possible issues and planned responses

| Issue | Planned response |
|---|---|
| Single-GPU memory and total runtime | Measure representative jobs; use supported memory optimizations and restartable runs; include generation, adaptation, evaluation, and storage in the budget |
| From-scratch generation cost | Start with the original low-resolution Tier A/B protocol, ADA, and DDIM sampling; retain unresolved tier applicability until clarified |
| GAN collapse or diffusion divergence | Monitor training and sample diversity, save checkpoints, and use predefined stopping criteria; retain failure records |
| Medical domain mismatch | Audit generated lesions and compare pretrained/adapted conditions without assuming a favorable result |
| Full experiment count | Enumerate required coverage before the main study and reuse generated sets where scientifically valid; do not remove named requirements to fit a pilot budget |
| FID reliability with small samples | Use consistent preprocessing, report reference/generated counts and uncertainty, and avoid treating small-sample scores as definitive |
| Incorrect boxes, captions, or class labels | Preserve or regenerate annotations, audit samples, and record rejected examples and review costs |
| Leakage and pretraining overlap | Split before fitting or generation, track source groups and duplicates, and document checkpoint training history where available |
| Library and checkpoint changes | Pin tested dependency versions and exact model revisions, and verify reproduction after environment changes |

**Generator applicability to clarify:** The original objectives describe all five families across all four tiers, while the detailed methods restrict from-scratch GAN/DDPM to Tiers A/B. Both statements remain recorded. We will begin with the shared A/B requirement and resolve C/D applicability with the advisor before declaring the full run matrix final.
