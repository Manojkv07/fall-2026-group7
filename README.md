# SynthAug-Bench

## When Does Diffusion-Based Synthetic Data Augmentation Help Classification, Object Detection, and Image Captioning?

### A Benchmark Across Generator Fidelity, Data Scarcity, Model Architectures, and Vision Tasks

**Original proposal and advisor:** Dr. Amir Jafari  
**Program:** Data Science, The George Washington University, Washington, DC

This project follows the original classification proposal and the advisor's feedback on object detection, image captioning, pretrained features, and hybrid modeling. The classification study retains all four data tiers and all four CNN/Transformer architectures. MS COCO 2017 is selected for the detection and captioning studies. These are two task tracks using the same dataset release.

This README describes the research plan. Dataset preparation, training code, and experimental results will be added in later stages.

## 1. Objective

The goal of this project is to build a reproducible benchmark that examines when synthetic data augmentation improves computer vision models. We will compare conventional augmentation, GANs, diffusion models trained from scratch, and pretrained diffusion models with and without adaptation.

The main research question is: under what data, model, and task conditions does diffusion-based synthetic augmentation improve performance over real-only training and conventional augmentation, and when does it provide no clear benefit or reduce performance?

We will study this question across controlled few-shot, constructed long-tail, fine-grained, and naturally imbalanced medical datasets. We will also examine whether CNN and Transformer classifiers respond differently to the same synthetic training data.

The original classification benchmark remains required. The COCO studies will test whether augmentation also helps models locate objects and describe images. The feature study will examine whether pretrained embeddings provide useful auxiliary information when combined with a task model. Downstream performance claims will be based on held-out real-image evaluation.

### Key objectives

1. Implement the five original augmentation families: conventional augmentation, class-conditional StyleGAN2-ADA, a small class-conditional DDPM, pretrained Stable Diffusion with class prompts and textual-inversion tokens, and LoRA-adapted Stable Diffusion with a defined DreamBooth-style objective.
2. Complete the four classification data-scarcity tiers, preserving every named dataset and regime in the original proposal.
3. Compare ResNet-50, ConvNeXt-Tiny, ViT-B/16, and Swin-T on real-only, conventionally augmented, and real-plus-synthetic training data.
4. Measure generated-sample quality with FID and Improved Precision & Recall, then evaluate whether these measurements predict downstream benefit. Generator families will not be assigned a quality ranking in advance.
5. Release a config-driven benchmark with experiment records, reproducible notebooks, and a single-GPU reproduction guide.
6. Evaluate generative augmentation for object detection and image captioning on MS COCO 2017, using valid task-specific annotations, controls, and metrics.
7. Investigate pretrained visual features, auxiliary information, and a defined hybrid approach against matched controls.
8. Compare improvements, failures, and computational costs across the three tasks.

### Research questions

| ID | Research question | Planned evidence |
|---|---|---|
| RQ1 / H1 | As measured generator fidelity improves, does classification benefit increase, plateau, or reverse? | Performance changes across generator conditions within the same dataset, real-data budget, classifier, and synthetic ratio |
| RQ2 / H2 | Do ResNet-50 and ConvNeXt-Tiny respond differently to synthetic data than ViT-B/16 and Swin-T? | Paired changes from each architecture's own controls, using the same real splits and generated sets |
| RQ3 / H3 | Do FID and Improved Precision & Recall predict downstream augmentation value? | Quality-to-utility regression within compatible settings, with held-out settings for any predictive claim |
| RQ4 / H4 | Which data-scarcity, generator, and synthetic-ratio settings reduce classification performance relative to conventional augmentation? | Negative and inconclusive changes reported with uncertainty, including rare-class and HAM10000 results |
| RQ5 | Do the augmentation methods tested across classification, detection, and captioning produce consistent benefits across tasks? | Changes from each task's controls, considered alongside annotation validity, training-data quantity, and computational cost |
| RQ6 | Do auxiliary pretrained features and a hybrid model improve performance beyond image augmentation alone? | Matched comparisons with and without auxiliary features, on real-only and real-plus-synthetic training data, including added cost |

RQ1-RQ4 preserve the original H1-H4 analysis. RQ5 and RQ6 address the advisor's feedback. An improvement, a decline, or an inconclusive result is a possible outcome for each question.

## 2. Dataset

The classification study retains all six named datasets across the original four tiers. Detection and captioning use MS COCO 2017 with different annotation types. Dataset versions, access terms, preprocessing, and split procedures will be recorded before use.

| Tier | Dataset | Planned setting |
|---|---|---|
| A: Controlled few-shot | [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) | 5, 10, 20, and 50 training images per class; preserve the official test set |
| B: Controlled long-tail | [CIFAR-100-LT](https://github.com/richardaecn/class-balanced-loss#datasets), constructed from CIFAR-100 | Exponential long-tail splits with imbalance ratios of 100, 50, and 10; include a reweighting comparison |
| C: Fine-grained low-data | [Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | Documented low-data training, validation, and test protocol |
| C: Fine-grained low-data | [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) | Documented low-data training, validation, and test protocol |
| C: Fine-grained low-data | [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/) | Documented low-data training, validation, and test protocol |
| D: Naturally imbalanced medical data | [HAM10000](https://doi.org/10.7910/DVN/DBW86T) | Skin-lesion classification; keep images of the same lesion in one split |
| Detection extension | [MS COCO 2017: Object Detection](https://cocodataset.org/#download) | Low-data object detection using object categories and bounding boxes; reserve val2017 for final evaluation |
| Captioning extension | [MS COCO 2017: Image Captioning](https://cocodataset.org/#download) | Low-data image captioning using human-written reference captions; reserve val2017 for final evaluation |

### COCO split and annotation plan

COCO provides object-instance annotations and image captions. The detection task covers 80 object categories. Captioning uses the human-written reference captions supplied with the dataset. The two tracks share the image source, but they require separate targets, models, and evaluation procedures. [COCO detection task](https://cocodataset.org/dataset/detection-2017.htm), [COCO captions paper](https://arxiv.org/abs/1504.00325)

We will divide train2017 into fixed, disjoint training and tuning pools. Low-data subsets will be drawn only from the training pool. Both COCO tracks will share the split manifest, and all captions and object annotations for an image will stay in that image's split. Any task-specific exclusions will be recorded. Subset sizes will be fixed after the pipeline pilot and before the main comparisons, with image counts and detection class coverage recorded for each subset.

The official val2017 split will serve as the project's held-out real-image evaluation set. It will not be used to fit generators, learn tokens, choose prompts, set filtering thresholds, select checkpoints, or tune models. This is our study protocol. It is not the COCO hidden test set or the commonly used Karpathy captioning split. Comparisons with published results will require matching the paper's release, image IDs, training data, and evaluation settings. [Captioning split reference](https://cs.stanford.edu/people/karpathy/deepimagesent/)

For COCO, synthetic:real ratios will count unique training images. Caption pairs and object instances will be reported separately. Multiple captions for one image will not be counted as multiple independent training images, and detection image counts will not be described as per-class shot counts.

### Data and generator preparation

We will use fixed training, validation, and test splits, with recorded seeds for few-shot and long-tail sampling. Official test splits will be preserved where provided. Any validation subset created from official training data will be reserved before low-data sampling or generator fitting. We will report the validation-data budget separately from the number of real images used for training. Classification loaders will share a common interface. Detection and captioning loaders will retain their task-specific annotations while using a shared provenance format.

For each dataset and generator condition, we will record the real and synthetic sample counts, synthetic:real ratio, fitting time, generation time, preprocessing, prompts, checkpoint revision, and quality measurements. Generator adaptation, learned embeddings, and source images or captions for synthetic training data will use training data only. Tuning data may guide method selection, but will not be added to the training pool. Final evaluation data will remain held out until the protocol is fixed.

Preprocessing will be documented in a notebook for each tier and task. HAM10000 preparation will include lesion-group and duplicate checks; lesion grouping does not establish patient-level independence when patient identifiers are unavailable. CUB-200-2011 has documented overlap with ImageNet, so we will record classifier and encoder pretraining and investigate identifiable overlaps. Unknown pretraining overlap will remain a limitation. [HAM10000 paper](https://arxiv.org/abs/1803.10417), [CUB-200-2011 notice](https://www.vision.caltech.edu/datasets/cub_200_2011/)

Detection samples will be checked for valid boxes and object labels. Captioning samples will be checked for agreement between the image and its description. A generation prompt is not automatically a correct annotation. Synthetic records will include source training-image or training-caption IDs when those inputs are used, class conditions where applicable, generation settings, and annotation-review records.

## 3. Rationale

Limited training data can make it difficult to learn rare classes, distinguish similar categories, or generalize to a different image domain. Synthetic data offers a way to expand the training set, but more images do not automatically provide more useful information. Generated examples can repeat existing patterns, change class-defining details, or introduce incorrect annotations.

SynthAug-Bench will examine these effects under a shared experimental protocol. The study will connect sample-quality measurements with downstream performance, architectural differences, data scarcity, and computational cost. The related-work review will cover the studies named in the original proposal, including DA-Fusion, ALIA, StableRep, Azizi et al., He et al., and Sariyildiz et al. It will record their exact datasets, splits, model initialization, training budgets, and any published corrections. The benchmark's contribution will be assessed against that review; using these datasets alone does not establish novelty.

HAM10000 provides a required domain-mismatch case study. We will compare pretrained and adapted generation without assuming that adaptation must improve results or that a pretrained model has never encountered similar images.

Detection and captioning extend the question beyond classification. A generated image may preserve its category while changing object position, count, or attributes. COCO lets us study spatial and language supervision using a common image source. The feature and hybrid study will test whether pretrained representations provide additional information under a controlled comparison.

The extended review will include work on [diffusion augmentation for object detection](https://openaccess.thecvf.com/content/WACV2024/html/Fang_Data_Augmentation_for_Object_Detection_via_Controllable_Diffusion_Models_WACV_2024_paper.html) and [diffusion augmentation for image captioning](https://arxiv.org/abs/2305.01855). Their results will be interpreted within their published protocols rather than assumed to transfer to this study.

## 4. Approach

### Phase 1: Foundations and baselines

Prepare the four-tier classification pipeline, the COCO detection and captioning interfaces, and the experiment tracking structure. Create generator_registry.csv and run_matrix.csv before the main experiments. Each planned run will identify its research question, task, dataset release, split, real-data budget, generator, task model, feature condition, synthetic ratio, seed, status, and cost. The planned project structure includes data, generators, classifiers, task models, metrics, benchmarks, and notebooks.

Implement real-only training, RandAugment, Mixup, and CutMix for classification. Evaluate the three named conventional policies with separately recorded settings and results. Train all four classification architectures and include the long-tail reweighting comparison. Establish separate COCO task controls, the evaluation metrics below, and the five-seed statistical reporting protocol.

Within a task and data budget, use the same real splits, model initialization policy, optimizer-step budget, and tuning procedure for comparable conditions. Record batch size, real/synthetic sampling probability, image exposures, and any unavoidable differences. Reuse generated sets across classifiers when the split and generation settings match. A fixed epoch count alone will not be treated as equal training effort when dataset sizes differ.

The planned software includes PyTorch, Hugging Face diffusers/transformers/accelerate/peft, timm, StyleGAN2-ADA, clean-fid, pycocotools, and a compatible COCO caption-evaluation implementation. Tested versions will be pinned during environment setup. The original hardware target is one NVIDIA A10G with 24 GB VRAM on AWS g5.2xlarge. Runtime and memory must be measured before claiming that the expanded study fits the available budget.

### Phase 2: GAN baseline

Train class-conditional StyleGAN2-ADA from scratch on the low-data splits. Begin with the original methods' Tier A/B experiments at 32x32 or 64x64 resolution. Monitor training stability, sample diversity, and possible collapse.

Generate synthetic sets at ratios of 1:1, 2:1, and 5:1 relative to real training data. Retrain all four classifiers for the applicable settings and record FID, precision/recall, downstream performance, and cost.

### Phase 3: Diffusion trained from scratch

Implement a small class-conditional DDPM with a lightweight U-Net and a noise-prediction objective. Use the same initial Tier A/B scope and resolution as the GAN comparison, with DDIM sampling to reduce generation time.

Generate data at the same three ratios and retrain the four classifiers. Compare GAN and DDPM results under documented, matched from-scratch compute budgets. Measure fidelity rather than assuming that one generator family must produce better samples.

### Phase 4: Pretrained foundation diffusion

Implement class-prompt augmentation with pretrained Stable Diffusion and a separate DA-Fusion-style condition using learned textual-inversion tokens. The diffusion backbone can remain frozen while token embeddings are fitted, so learned-token augmentation will not be labelled entirely training-free.

Reproduce DA-Fusion using a matching published dataset and protocol before extending it to the required classification settings, including fine-grained and medical data. Flowers and Pets remain required project datasets; they will not be called reproduction datasets without a matching published protocol. The authors' COCO-derived classification experiment is distinct from our COCO detection and captioning tracks. [DA-Fusion implementation and token fitting](https://github.com/brandontrabucco/da-fusion)

### Phase 5: Adapted foundation diffusion

Adapt Stable Diffusion across Tiers A-D using LoRA and an explicit DreamBooth-style class or subject-binding objective. Record the trainable components, LoRA rank, adaptation data, and optimization settings. Use the original rank range of 4-16 as planned settings to investigate.

Measure memory and runtime with mixed precision and gradient accumulation where supported. Evaluate the proposed SD-Turbo/LCM acceleration options as separately recorded configurations so a checkpoint or adapter change is not hidden inside a sampler comparison.

Generate the required synthetic ratios and complete downstream comparisons across all four classifiers. The HAM10000 analysis will test whether adaptation changes the usefulness of generated data in the medical domain.

### Required task and feature extensions

**Object detection on MS COCO 2017.** Establish a detector with a documented backbone, initialization, image resolution, and training budget. Compare real-only training, conventional augmentation that correctly updates boxes, pretrained diffusion augmentation, and adapted diffusion augmentation. Any added generation condition will have its own recorded configuration. The same detector and evaluation protocol will be used within each comparison.

Generated detection images must preserve or receive corrected category and box annotations. Checks will cover moved, added, removed, and partially visible objects. Original boxes will not be reused automatically after an edit. Rejected examples, annotation corrections, automated checks, manual review, and their costs will be recorded. The audit procedure will be fixed on development data before the main runs.

**Image captioning on MS COCO 2017.** Establish a captioning model with documented visual and text components, initialization, decoding settings, and training budget. Compare real-only training, caption-preserving conventional augmentation, pretrained diffusion augmentation, and adapted diffusion augmentation. Generated image-caption pairs will be checked for objects, attributes, counts, and relationships. Crops or flips that change the meaning of a caption will require a corrected label or exclusion.

All reference captions for an evaluation image will remain together. Caption generation and scoring settings will be fixed before final evaluation. Automatic scores will be accompanied by a predefined, blinded sample review of factual agreement with the image. The review will report its sample size and disagreements between reviewers; it will not be presented as an exhaustive check of every generated sentence.

**Pretrained features and hybrid modeling.** The planned comparison will combine a task model's visual features with embeddings from a frozen auxiliary image encoder. We will document the feature dimensions, combination rule, prediction head, trainable components, and inference inputs. Four conditions will separate the effects: the base model trained on real images, the base model trained on real-plus-synthetic images, the hybrid trained on real images, and the hybrid trained on real-plus-synthetic images. The hybrid will also have a capacity-matched control so an enlarged prediction head is not the only explanation for a gain. Evaluation inputs will contain only information available at prediction time.

These tracks will be developed alongside the classification phases. The detector, captioner, auxiliary encoder, fusion implementation, feature-study coverage, and checkpoint revisions will be selected and recorded during Phase 1 after compatibility and pretraining checks. The choice of COCO is settled; these implementation choices are still open. Public checkpoints will not be assumed to be free of COCO or other evaluation-image exposure. Each comparison will state whether it measures low-data adaptation of a pretrained model or learning from a restricted training set.

The COCO comparisons will use fixed low-data subsets, synthetic:real image ratios of 1:1, 2:1, and 5:1, and five downstream training seeds for each selected condition. Prompt-only, learned-token, and LoRA settings will remain separate when used. The run matrix will state the selected task-specific generator coverage explicitly. The feedback does not establish an all-generators-by-all-tasks requirement. All original classification datasets, methods, architectures, ratios, and repeats remain required, subject to the recorded source applicability conflict.

### Phase 6: Cross-condition analysis

Assemble the master benchmarking DataFrame and answer RQ1-RQ6, including all four original hypotheses. Every result will link to its run configuration, source split, annotations, model revision, and control.

#### Evaluation measures

| Task or analysis | Primary measure | Supporting measures and checks |
|---|---|---|
| Classification | Top-1 accuracy | Balanced accuracy, macro-F1, per-class recall, and confusion matrices, with rare-class analysis for imbalanced settings |
| COCO object detection | Bounding-box AP averaged over IoU thresholds 0.50 to 0.95 in steps of 0.05 | AP50, AP75, AP by object size, and per-category analysis |
| COCO image captioning | CIDEr, with the scoring variant and implementation recorded | SPICE, BLEU-4, METEOR, ROUGE-L, and blinded factual-agreement review |
| Generated-image quality | FID and Improved Precision & Recall | Reference and generated counts, preprocessing, feature extractor, and sample-count limitations |
| Feature/hybrid study | The selected task's primary measure | Change from matched controls, added parameters, training and inference time, and memory |
| Computational cost | Measured generation, adaptation, and downstream-training time | Peak VRAM, storage, annotation-review effort, and monetary cost where available |

Detection evaluation will use the COCO bounding-box protocol. Caption evaluation will use a tested implementation with fixed tokenization, reference handling, and score scaling. Exact versions will be saved with the results. [COCO detection evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py), [COCO caption metrics](https://github.com/tylin/coco-caption)

Use five downstream training seeds per condition and report means, standard deviations, and paired changes from controls. Report 95% bootstrap confidence intervals with the resampling unit and number of resamples specified. Seed-level uncertainty and uncertainty from evaluation-image sampling will be distinguished. If images are resampled, captions and boxes will stay grouped by image, and HAM10000 observations will stay grouped by lesion. Data-selection and generation seeds will be recorded separately; five training seeds do not imply five independent generated datasets. With only five training repeats, intervals and small effects will be interpreted cautiously.

Report each task's change from its real-only and conventional controls on its own metric scale. Accuracy, AP, and CIDEr will not be pooled into one score. The COCO tracks will support comparisons on a shared image source, while comparisons with the classification tiers will also reflect differences in datasets and models. These results will not be treated as isolating the effect of task alone.

Evaluate quality-to-utility regression within compatible settings, including synthetic ratio and architecture family where applicable. Any claim of prediction on new settings will require held-out evaluation settings that do not share the same generated set with the regression's training data. Per-seed reuse of a generated set will be accounted for rather than treated as independent quality measurements.

FID and Improved Precision & Recall describe generated-image distributions; they do not establish that a box or caption is correct. Conventional Mixup and CutMix controls will not be forced into a generator-quality ranking. Quality comparisons will use matched reference and generated sample counts where feasible, with class composition and exceptions recorded. Development quality measurements will use development references. Any final held-out quality analysis will happen after method selection and will not guide further tuning. Clean-FID standardizes preprocessing but does not remove small-sample uncertainty. [Clean-FID project](https://www.cs.cmu.edu/~clean-fid/)

Report failed and inconclusive experiments alongside improvements, with missing runs identified separately from measured negative results. Both students will independently check H1-H4 and the cross-task and hybrid analyses.

### Phase 7: Guidelines, report, and code release

Produce practical generator-selection guidelines based on data scarcity, task, and measured compute. Prepare the original figure set: performance versus measured fidelity, FID/precision-recall versus performance gain, CNN versus Transformer benefit, and the HAM10000 domain comparison. Add COCO detection and captioning results, annotation-audit summaries, and the feature/hybrid findings. Plots will identify the real-data budget and the uncertainty of each comparison.

Write the 8-10-page paper-format report with motivation, related work, methods, RQ1-RQ6 results, the medical analysis, cross-task findings, practical guidance, and limitations. Detection, captioning, and the feature/hybrid investigation belong in the current study. Segmentation retains its original future-work status.

Release the config-driven benchmark with the generator registry, complete run matrix, master results, experiment records, split manifests, annotation-audit records, a full-study execution command, notebooks per phase and tier, COCO task notebooks, a cross-condition analysis notebook, pinned tested dependencies, README, and single-GPU setup and reproduction instructions. Dataset access instructions will respect each source's terms. Prepare the demonstration and final presentation.

The original Week 16 paper-submission milestone remains part of the delivery plan, with destination and timing to be confirmed with the advisor. A completed manuscript and a submission are separate from publication acceptance. The current stage is README preparation; dataset downloads, training, and submission will follow in later steps.

## 5. Timeline

The original target is 16 weeks. COCO detection, COCO captioning, and the feature/hybrid study are integrated into that sequence. Calendar dates and resource requirements will be established from actual access and pilot measurements.

| Weeks | Original study | Integrated feedback work |
|---|---|---|
| 1-3 | Environment, all classification tiers, four-classifier controls, registry, full matrix, metrics, and five-seed protocol | Fix COCO split manifests, low-data budgets, task models and controls, annotation audits, and the feature/hybrid design |
| 4-5 | StyleGAN2-ADA training, generation, and classifier comparisons | Validate COCO boxes, caption consistency, and task evaluation; checkpoint and monitor failures |
| 6-7 | Small DDPM training and matched GAN/DDPM comparisons | Prepare task-specific generation and provenance checks |
| 8-9 | DA-Fusion reproduction and pretrained augmentation, including fine-grained and medical data | Run pretrained diffusion comparisons for COCO detection and captioning |
| 10-11 | LoRA/DreamBooth-style adaptation and required classifier comparisons | Complete adapted-generation and feature/hybrid comparisons; review synthetic samples before full retraining |
| 12-13 | Master benchmark, H1-H4, regression, and uncertainty analysis | Evaluate RQ5-RQ6, annotation findings, and task-specific costs; independently cross-check results |
| 14 | Practical guidelines and original final figures | COCO and feature findings in the figures and analysis |
| 15-16 | Paper-format report, notebooks, code release, submission milestone, and final presentation | Integrated report and demonstration covering all three tasks and the feature/hybrid study |

Small pilot runs will check the pipeline and measure cost. They change execution order only. The full dataset, method, architecture, ratio, and five-seed requirements remain. Any resource shortfall will be recorded and addressed through scheduling and resource planning. The final review retains the original three-day code freeze.

## 6. Expected Number of Students

The project will be completed by two students.

| Responsibility | Student 1 | Student 2 |
|---|---|---|
| Original method ownership | Conventional augmentation, StyleGAN2-ADA, small DDPM, FID/precision-recall, generator registry | Pretrained diffusion, textual inversion, LoRA/DreamBooth-style adaptation, four-classifier benchmarking, master result table |
| Task extensions | COCO detection pipeline, box verification, and AP evaluation | COCO captioning pipeline, image-caption verification, and caption evaluation |
| Shared work | Feature/hybrid study, split checks, statistical cross-checks, guidelines, writing, documentation, and presentation | Feature/hybrid study, split checks, statistical cross-checks, guidelines, writing, documentation, and presentation |

Hold weekly integration meetings so every method uses the same recorded training and evaluation protocol. Both students should be able to reproduce and explain the complete pipeline.

## 7. Possible Issues

| Issue | Planned response |
|---|---|
| Single-GPU memory and total runtime | Measure representative jobs; use supported memory optimizations and restartable runs; include generation, adaptation, evaluation, and storage in the budget |
| From-scratch generation cost | Start with the original low-resolution Tier A/B protocol, ADA, and DDIM sampling; retain unresolved tier applicability until clarified |
| GAN collapse or diffusion divergence | Monitor training and sample diversity, save checkpoints, and use predefined stopping criteria; retain failure records |
| Medical domain mismatch | Audit generated lesions and compare pretrained/adapted conditions without assuming a favorable result |
| Full experiment count | Enumerate required coverage before the main study and reuse generated sets where scientifically valid; do not remove named requirements to fit a pilot budget |
| FID reliability with small samples | Use consistent preprocessing, report reference/generated counts and uncertainty, and avoid treating small-sample scores as definitive |
| Incorrect boxes, captions, or class labels | Preserve or regenerate annotations, audit samples, and record rejected examples and review costs |
| COCO split or reference leakage | Keep train2017 training/tuning pools disjoint; reserve val2017 for final evaluation; keep all annotations for an image together |
| Leakage and pretraining overlap | Split before fitting or generation, track source groups and duplicates, investigate CUB/ImageNet overlap, and document generator, task-model, and auxiliary-encoder pretraining |
| Unfair comparisons with published COCO results | Match dataset release, image IDs, training-data access, model initialization, and scorer settings before comparing reported scores |
| Caption metrics miss factual errors | Pair fixed automatic metrics with a predefined blinded image-caption review and report the review's limitations |
| Apparent hybrid gains from extra capacity | Compare against the base model and a capacity-matched head control; report added training and inference cost |
| Uncertain or missing results | Report uncertainty and failures explicitly; keep unrun experiments separate from measured negative outcomes |
| Library and checkpoint changes | Pin tested dependency versions and exact model revisions, and verify reproduction after environment changes |

**Generator applicability to clarify:** The original objectives describe all five families across all four tiers, while the detailed methods restrict from-scratch GAN/DDPM to Tiers A/B. Both statements remain recorded. We will begin with the shared A/B requirement and resolve C/D applicability with the advisor before declaring the full run matrix final.

**Implementation decisions to finish in Phase 1:** COCO subset sizes, task models and checkpoint revisions, task-specific generator settings, the auxiliary encoder and fusion rule, feature-study coverage, exact metric implementations, and measured run costs. These decisions will be documented before the main study. They do not make any original dataset or required feedback task optional.
