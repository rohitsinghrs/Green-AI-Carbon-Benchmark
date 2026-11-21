# 🌱 Green AI Carbon Benchmark
### A Reproducible Experimental Framework for Evaluating Machine Learning Architecture, Computational Efficiency, and Carbon Emissions Across Hardware Platforms

<p align="center">
  <img src="https://img.shields.io/badge/Research-Green%20AI-2E8B57?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Domain-Machine%20Learning-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Carbon%20Tracking-CodeCarbon-228B22?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Research%20Project-yellow?style=for-the-badge" />
</p>

<p align="center">
  <b>Beyond Predictive Accuracy: Towards Environmentally Sustainable Machine Learning</b>
</p>

---

## Abstract

The increasing adoption of Artificial Intelligence (AI) and Machine Learning (ML) has intensified concerns regarding the computational resources and environmental costs associated with model development and deployment. Conventional machine learning evaluation predominantly emphasizes predictive performance, including accuracy, precision, recall, and F1-score, while computational efficiency and environmental sustainability frequently receive comparatively less attention.

This research investigates the relationship between machine learning architecture, computational efficiency, hardware characteristics, and carbon emissions through a controlled experimental benchmarking framework.

The proposed research framework examines computational efficiency as a potential mediating mechanism between model architecture and environmental impact, while considering hardware platform as a potential moderating factor.

An experimental pipeline is implemented using Python and PyTorch, incorporating multiple neural network architectures, standardized preprocessing, controlled training configurations, execution-time measurement, and carbon emission tracking through CodeCarbon.

The implementation includes a linear baseline, Small Multilayer Perceptron (MLP), Deep MLP, and one-dimensional Convolutional Neural Network (1D-CNN). Experiments use the Adult Income classification dataset with reproducible preprocessing and configurable training parameters.

The primary objective is to move beyond accuracy-centric model selection and establish an evaluation approach that considers predictive quality, computational cost, and environmental impact together.

**Keywords:** Green AI, Sustainable Machine Learning, Carbon Emissions, Energy Efficiency, Computational Efficiency, Deep Learning, Model Architecture, Hardware-Aware Computing, CodeCarbon, Environmental Sustainability.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Research Motivation](#2-research-motivation)
3. [Research Problem](#3-research-problem)
4. [Research Gap](#4-research-gap)
5. [Research Objectives](#5-research-objectives)
6. [Research Questions and Hypotheses](#6-research-questions-and-hypotheses)
7. [Conceptual Research Framework](#7-conceptual-research-framework)
8. [Experimental Methodology](#8-experimental-methodology)
9. [Dataset Description](#9-dataset-description)
10. [Model Architectures](#10-model-architectures)
11. [Hardware and Experimental Environment](#11-hardware-and-experimental-environment)
12. [Carbon Emission Measurement](#12-carbon-emission-measurement)
13. [Evaluation Metrics](#13-evaluation-metrics)
14. [Experimental Results and Analysis](#14-experimental-results-and-analysis)
15. [Statistical Analysis](#15-statistical-analysis)
16. [Research Contributions](#16-research-contributions)
17. [Limitations](#17-limitations)
18. [Future Research Directions](#18-future-research-directions)
19. [Repository Structure](#19-repository-structure)
20. [Installation and Reproducibility](#20-installation-and-reproducibility)
21. [Technology Stack](#21-technology-stack)
22. [References](#22-references)
23. [License](#23-license)

---

# 1. Introduction

Artificial Intelligence and Machine Learning have become essential technologies across healthcare, finance, transportation, scientific research, and industrial automation.

The increasing complexity of machine learning architectures has resulted in growing computational requirements, including processor utilization, memory consumption, execution time, and electricity consumption.

Although modern ML research primarily focuses on improving predictive performance, achieving higher accuracy does not necessarily imply better computational or environmental efficiency.

A model with marginally higher predictive performance may require substantially more computational resources than a simpler alternative.

This creates an important research question:

**Can machine learning models be evaluated not only by how accurately they learn, but also by how efficiently and sustainably they perform?**

Green AI addresses this challenge by encouraging researchers to consider computational costs and environmental consequences alongside conventional performance metrics.

This research develops an empirical benchmarking approach to investigate how different machine learning architectures influence computational efficiency and carbon emissions under controlled experimental conditions.

The central idea is to integrate environmental considerations into the machine learning model evaluation process rather than treating sustainability as a separate consideration.

# 2. Research Motivation

Traditional machine learning evaluation commonly relies on:

- Accuracy
- Precision
- Recall
- F1-score
- Training loss

However, these metrics do not independently explain the computational resources required to obtain a particular level of predictive performance.

For example:

- A linear model may require relatively few trainable parameters.
- A small MLP may introduce additional computational requirements.
- A deep MLP may increase model complexity.
- A 1D-CNN may introduce a different computational structure through convolution operations.

The environmental implications of these architectures may also depend on the hardware platform on which they execute.

Consequently, architecture selection should consider multiple dimensions:

1. Predictive performance.
2. Computational complexity.
3. Training and inference time.
4. Energy consumption.
5. Carbon emissions.
6. Hardware characteristics.

This research is motivated by the need for a more comprehensive and environmentally responsible approach to machine learning evaluation.

# 3. Research Problem

The primary research problem addressed in this study is the limited integration of computational efficiency and carbon emission measurements into conventional machine learning architecture comparisons.

Most model selection processes prioritize predictive performance without adequately considering the environmental cost associated with achieving that performance.

Furthermore, computational measurements may vary across hardware platforms, making it necessary to investigate whether architecture-related efficiency patterns remain consistent under different execution environments.

**Problem Statement:**

There is a need for a reproducible experimental framework that evaluates the relationship between machine learning architecture, computational efficiency, hardware execution conditions, and carbon emissions while maintaining predictive performance as an essential evaluation criterion.

# 4. Research Gap

This study focuses on the following research gaps:

| Research gap | Research direction |
|---|---|
| Accuracy-centric evaluation | Combine predictive and environmental metrics |
| Limited carbon awareness | Integrate CodeCarbon-based tracking |
| Architecture-related computational differences | Compare multiple neural architectures |
| Hardware dependency | Investigate CPU and GPU execution environments |
| Limited reproducibility | Use controlled configurations and random seeds |
| Isolated performance measurements | Examine relationships between runtime, architecture, and emissions |

The research aims to contribute an empirical framework for examining these relationships.

These gaps are the motivation and scope of this investigation, rather than a claim that no previous studies have addressed them.

# 5. Research Objectives

The main objective is to investigate how machine learning architecture and hardware execution conditions influence computational efficiency and environmental impact.

Specific objectives:

1. Develop a reproducible benchmarking pipeline for multiple machine learning architectures.
2. Evaluate predictive performance using accuracy, precision, recall, and F1-score.
3. Measure training and inference execution time.
4. Track carbon emissions using CodeCarbon.
5. Compare computational characteristics across model architectures.
6. Investigate the influence of hardware platforms on computational performance and emissions.
7. Examine computational efficiency as a potential mediator between architecture and carbon emissions.
8. Investigate hardware platform as a potential moderator of the architecture–efficiency and efficiency–emissions relationships.
9. Explore sustainability-aware model selection strategies.

# 6. Research Questions and Hypotheses

### Research Questions

**RQ1:** How does machine learning architecture influence computational efficiency?

**RQ2:** What relationship exists between computational efficiency and carbon emissions?

**RQ3:** Does computational efficiency mediate the relationship between model architecture and carbon emissions?

**RQ4:** Does hardware platform moderate the relationship between architecture and computational efficiency?

**RQ5:** Can a model with competitive predictive performance achieve lower computational and environmental costs?

### Research Hypotheses

| Hypothesis | Description |
|---|---|
| H1 | Machine learning architecture significantly influences computational efficiency. |
| H2 | Computational efficiency is associated with carbon emissions. |
| H3 | Computational efficiency mediates the relationship between architecture and carbon emissions. |
| H4 | Hardware platform moderates the relationship between architecture and computational efficiency. |
| H5 | Models with comparable predictive performance may exhibit different environmental footprints. |

These hypotheses require formal statistical testing before they can be accepted or rejected.

# 7. Conceptual Research Framework

The proposed framework examines the interaction between three primary research components and one contextual variable.

### Research Variables

| Variable | Research role | Description |
|---|---|---|
| Model Architecture | Independent Variable (IV) | Type and structural complexity of ML architecture |
| Computational Efficiency | Mediating Variable | Training time, inference time, and computational cost |
| Carbon Emissions | Dependent Variable (DV) | Estimated emissions associated with model execution |
| Hardware Platform | Moderating Variable | CPU or GPU execution environment |

### Conceptual Relationship

```text
                  HARDWARE PLATFORM
                    (Moderator)
                         |
                         v
MODEL ARCHITECTURE ---> COMPUTATIONAL EFFICIENCY
       (IV)                  (Mediator)
                                  |
                                  v
                         CARBON EMISSIONS
                               (DV)

Additional relationship:
Model Architecture ----------------> Carbon Emissions
                 Direct Effect
```

### Theoretical Interpretation

The framework proposes that model architecture may influence carbon emissions through differences in computational requirements.

Computational efficiency is therefore examined as a potential explanatory mechanism.

Hardware characteristics may further influence execution time, energy consumption, and the environmental impact of a given architecture.

The framework is intended to support empirical investigation rather than assume that mediation or moderation has already been statistically established.

# 8. Experimental Methodology

The research follows a quantitative, experimental benchmarking approach.

The experimental workflow consists of the following stages.

### Stage 1: Dataset Preparation

The Adult Income dataset is prepared for supervised binary classification.

The target variable is `income`.

### Stage 2: Data Preprocessing

The preprocessing pipeline includes:

- Separation of numerical and categorical features.
- StandardScaler for numerical features.
- OneHotEncoder for categorical features.
- Training-only fitting of preprocessing transformations.
- Application of fitted transformations to the test dataset.

This approach avoids fitting preprocessing transformations on test data.

### Stage 3: Model Construction

Four neural network architecture configurations are implemented:

- Linear baseline.
- Small MLP.
- Deep MLP.
- 1D-CNN.

### Stage 4: Controlled Training

The training pipeline defines:

| Parameter | Configuration |
|---|---|
| Random seed | 42 |
| Training epochs | 20 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Configured repetitions | 3 |
| Learning framework | PyTorch |

### Stage 5: Performance Evaluation

Each model is evaluated using standard classification metrics and computational measurements.

### Stage 6: Carbon Tracking

CodeCarbon is integrated into the CPU experiment pipeline to estimate emissions associated with model execution.

### Stage 7: Result Collection

Experimental measurements are stored in CSV files for further analysis.

### Stage 8: Comparative Analysis

The collected measurements provide the basis for investigating the relationships between architecture, predictive performance, computational efficiency, and environmental impact.

# 9. Dataset Description

### Adult Income Dataset

The experiments use the Adult Income dataset, commonly associated with the UCI Machine Learning Repository.

The dataset is used for binary classification, where the objective is to predict whether an individual's annual income exceeds a specified threshold.

| Attribute | Description |
|---|---|
| Dataset | Adult Income |
| Learning task | Binary Classification |
| Target variable | `income` |
| Feature types | Numerical and categorical |
| Preprocessing | StandardScaler and OneHotEncoder |
| Training data | `adult_train_clean.csv` |
| Testing data | `adult_test_clean.csv` |

The repository contains prepared training and testing files under `prepared_data/`.

The exact sample counts, class distribution, and final feature dimensions should be reported from the released data files rather than assumed.

# 10. Model Architectures

The study implements different neural network structures to examine architecture-related computational differences.

### 10.1 Linear Baseline

A single linear transformation provides a relatively simple baseline architecture.

Its purpose is to establish a reference point for evaluating more complex neural networks.

### 10.2 Small Multilayer Perceptron

The Small MLP uses a feed-forward architecture with two hidden layers.

Its structure is:

```text
Input Layer
     |
Dense (128)
     |
ReLU
     |
Dense (64)
     |
ReLU
     |
Output Layer
```

### 10.3 Deep Multilayer Perceptron

The Deep MLP introduces a deeper feed-forward structure to investigate the computational implications of increased network depth.

### 10.4 One-Dimensional Convolutional Neural Network

The 1D-CNN uses convolution-based feature processing to investigate how a different computational architecture behaves compared with fully connected networks.

### Architecture Comparison

| Architecture | Category | Experimental purpose |
|---|---|---|
| Linear | Linear baseline | Reference computational performance |
| Small MLP | Shallow neural network | Evaluate moderate architectural complexity |
| Deep MLP | Deep neural network | Examine increased network depth |
| 1D-CNN | Convolutional neural network | Examine convolution-based computation |

# 11. Hardware and Experimental Environment

Hardware is an important experimental consideration because execution time and energy consumption can depend on the underlying computing environment.

The repository includes hardware inspection utilities and configuration for local AMD CPU and Colab GPU experiments.

### Hardware Configuration

| Component | Configuration |
|---|---|
| Local execution | AMD CPU |
| GPU experiment configuration | Google Colab GPU |
| Deep learning framework | PyTorch |
| Hardware inspection | psutil |
| Device selection | PyTorch CUDA availability |
| Carbon tracking | CodeCarbon |

The preprocessing pipeline automatically detects CUDA availability and selects the corresponding execution device.

For a valid cross-hardware comparison, experiments should use documented hardware specifications and equivalent data, model, and training configurations.

Exact processor, GPU, memory, CUDA, and software versions should accompany final comparative results.

# 12. Carbon Emission Measurement

Carbon emission measurement is an essential component of this research.

The project integrates CodeCarbon's `EmissionsTracker` to estimate emissions associated with machine learning experiments.

### Carbon Emission Estimation

The conceptual relationship between energy consumption and carbon emissions is:

\[
CO_2e = E \times CI
\]

Where:

- \(CO_2e\) = Carbon dioxide equivalent emissions.
- \(E\) = Energy consumed in kWh.
- \(CI\) = Electricity carbon intensity in kgCO₂e/kWh.

CodeCarbon estimates emissions using available hardware telemetry and configured or inferred electricity carbon intensity information.

### Why CodeCarbon?

CodeCarbon provides a practical mechanism for estimating the environmental footprint of computational workloads.

It helps researchers move beyond runtime-only comparisons by introducing an environmental measurement dimension.

### Measurement Considerations

For reliable experiments, the following should be documented:

- Tracking start and stop boundaries.
- Hardware specifications.
- Electricity carbon intensity source.
- Tracking configuration.
- Repeated measurement variability.
- Background system activity.
- Whether measurements include preprocessing, training, inference, or the complete pipeline.

Carbon emission estimates should be interpreted according to the actual tracking boundaries and measurement configuration.

# 13. Evaluation Metrics

The framework considers predictive performance, computational efficiency, and environmental measurements.

### 13.1 Predictive Performance

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Correct positive predictions |
| Recall | Identification of actual positive samples |
| F1-score | Harmonic mean of precision and recall |

### 13.2 Computational Efficiency

| Metric | Purpose |
|---|---|
| Training Time | Time required to train the model |
| Inference Time | Time required for prediction |
| Trainable Parameters | Model size and structural complexity |

### 13.3 Environmental Metrics

| Metric | Purpose |
|---|---|
| Carbon Emissions (gCO₂e) | Estimated environmental impact |
| Carbon Emissions (kgCO₂e) | Emissions in kilogram units |

### Sustainability-Aware Evaluation

The broader objective is to examine model quality alongside environmental cost.

A model should not be considered environmentally superior based solely on its execution time or parameter count. Actual energy and emissions measurements are necessary.

# 14. Experimental Results and Analysis

The repository provides CSV-based experimental output structures for recording architecture-level measurements.

The final research analysis should distinguish between preliminary execution results and fully validated experimental findings.

### Recorded Result Fields

The CPU experimental results include:

| Field | Description |
|---|---|
| architecture | Evaluated architecture |
| hardware | Execution platform |
| repeat | Experiment repetition |
| seed | Random seed |
| accuracy | Classification accuracy |
| precision | Precision score |
| recall | Recall score |
| f1 | F1-score |
| training_time_sec | Training duration |
| inference_time_sec | Inference duration |
| parameters | Trainable parameter count |
| emissions_kg | Estimated carbon emissions in kg |
| emissions_g | Estimated carbon emissions in grams |

### Result Interpretation

The following comparisons are intended for the final analysis:

1. Predictive performance across architectures.
2. Training time across architectures.
3. Inference time across architectures.
4. Trainable parameter counts.
5. Carbon emissions across architectures.
6. Hardware-dependent differences.
7. Accuracy–efficiency trade-offs.
8. Relationships between computational efficiency and emissions.

**Research integrity note:** Numerical findings, statistical significance, percentage improvements, and architecture rankings should only be reported after validating the corresponding raw experimental logs. No unverified performance or carbon reduction values are presented as established results in this README.

# 15. Statistical Analysis

Statistical analysis is required to determine whether observed differences between architectures are meaningful.

The research framework can incorporate the following methods:

### 15.1 Descriptive Statistics

Calculate:

- Mean
- Standard deviation
- Minimum
- Maximum
- Confidence intervals

### 15.2 Analysis of Variance (ANOVA)

ANOVA can be used to investigate whether mean computational or environmental measurements differ across architectures, subject to the assumptions of the selected statistical design.

### 15.3 Non-parametric Analysis

Kruskal–Wallis testing can be considered when distributional assumptions for conventional parametric comparisons are not appropriate.

### 15.4 Correlation Analysis

Correlation analysis can investigate associations between:

- Training time and emissions.
- Model parameters and execution time.
- Computational efficiency and emissions.
- Predictive performance and computational cost.

### 15.5 Mediation Analysis

Mediation analysis can investigate whether computational efficiency explains part of the relationship between architecture and carbon emissions.

### 15.6 Moderation Analysis

Moderation analysis can investigate whether hardware platform changes the relationship between model architecture and computational efficiency.

These analyses represent the intended analytical framework. Only completed and validated statistical tests should be reported as empirical findings.

# 16. Research Contributions

The expected contributions of this research are:

**1. Multi-dimensional model evaluation**

An evaluation approach that considers predictive performance, computational efficiency, and carbon emissions together.

**2. Architecture-level benchmarking**

A comparative experimental setup for examining multiple neural network architectures under controlled training configurations.

**3. Carbon-aware experimental measurement**

Integration of CodeCarbon into the machine learning experimental pipeline.

**4. Hardware-aware investigation**

An experimental direction for examining the role of hardware platform in computational and environmental measurements.

**5. Reproducible experimentation**

Use of configurable experiment parameters, fixed random seeds, preprocessing pipelines, and structured CSV outputs.

**6. Sustainability-aware model selection**

A research direction that encourages selecting machine learning models based on predictive quality and computational-environmental trade-offs.

# 17. Limitations

The findings should be interpreted in the context of the following limitations.

1. **Dataset dependency:** Results obtained from the Adult Income dataset may not generalize to image, text, time-series, or other learning tasks.

2. **Hardware dependency:** Computational and environmental measurements can vary across processor and GPU configurations.

3. **Carbon estimation uncertainty:** CodeCarbon provides estimates rather than direct, laboratory-grade measurement of total facility emissions.

4. **Background activity:** Other processes running on the machine may influence energy measurements.

5. **Limited architecture coverage:** The current implementation focuses on four architecture configurations.

6. **Experimental repetitions:** The configured repetition count should be expanded where necessary to obtain stronger estimates of experimental variability.

7. **GPU comparison:** A valid CPU–GPU comparison requires completed and consistently controlled runs on both platforms.

8. **Environmental scope:** The current research focuses on operational computational emissions and does not independently quantify embodied hardware carbon, manufacturing, or full lifecycle environmental impact.

# 18. Future Research Directions

Future development can extend the framework through:

- Larger and more diverse benchmark datasets.
- Additional classical ML and deep learning architectures.
- More comprehensive CPU and GPU comparisons.
- Increased independent experimental repetitions.
- Confidence intervals and effect-size reporting.
- Improved energy measurement validation.
- Regional electricity carbon-intensity comparisons.
- Quantization and pruning experiments.
- Early stopping and efficient training strategies.
- Sustainability-aware model selection.
- A standardized benchmarking and visualization dashboard.

These directions can improve the generalizability, reliability, and practical usefulness of the research.

# 19. Repository Structure

The repository is organized around experimental configurations, preprocessing, architecture definitions, model training, hardware inspection, and carbon measurement.

```text
Green-AI-Carbon-Benchmark/
│
├── adult/
│
├── prepared_data/
│   ├── adult_train_clean.csv
│   └── adult_test_clean.csv
│
├── codecarbon_cpu/
├── codecarbon_final/
├── codecarbon_repeated/
├── codecarbon_research_cpu/
├── codecarbon_test/
│
├── experiment_config.py
├── models.py
├── pipeline.py
│
├── train_models.py
├── test_models.py
│
├── cpu_carbon_experiment.py
├── cpu_final_experiment.py
├── cpu_repeated_experiment.py
├── cpu_research_experiment.py
│
├── device_check.py
├── hardware_check.py
├── codecarbon_test.py
├── prepare_adult.py
│
├── cpu_carbon_results.csv
├── cpu_final_results.csv
├── cpu_performance_results.csv
├── cpu_repeated_results.csv
├── cpu_research_raw.csv
├── cpu_research_summary.csv
├── cpu_summary.csv
│
├── README.md
└── .gitignore
```

# 20. Installation and Reproducibility

### Prerequisites

- Python 3.10 or later.
- pip package manager.
- PyTorch-compatible execution environment.
- CPU or supported NVIDIA GPU for GPU experiments.

### Step 1: Clone the Repository

```bash
git clone https://github.com/rohitsinghrs/Green-AI-Carbon-Benchmark.git

cd Green-AI-Carbon-Benchmark
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

Activate on Linux/macOS:

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip

pip install numpy pandas scikit-learn torch codecarbon psutil
```

Install a PyTorch build compatible with your hardware and CUDA environment where applicable.

### Step 4: Verify Hardware

```bash
python hardware_check.py
```

For device detection:

```bash
python device_check.py
```

### Step 5: Verify Model Architectures

```bash
python test_models.py
```

This checks model construction and output dimensions using a sample input batch.

### Step 6: Run the Preprocessing Pipeline

```bash
python pipeline.py
```

This loads the prepared dataset, applies preprocessing, and identifies the available execution device.

### Step 7: Execute CPU Carbon Experiments

```bash
python cpu_final_experiment.py
```

The experiment integrates CodeCarbon tracking and writes structured experimental results.

### Step 8: Inspect Experimental Results

Review the generated CSV files:

```text
cpu_final_results.csv
cpu_research_raw.csv
cpu_research_summary.csv
```

**Important:** Run scripts according to their configured datasets and hardware assumptions. Do not interpret separate experiment outputs as directly comparable unless their configurations and measurement boundaries are aligned.

# 21. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| PyTorch | Neural network construction and training |
| NumPy | Numerical computation |
| Pandas | Dataset processing and result storage |
| Scikit-learn | Preprocessing and evaluation metrics |
| CodeCarbon | Carbon emission estimation |
| psutil | Hardware and system information |
| CSV | Experimental result storage |
| Git and GitHub | Version control and research reproducibility |

# 22. References

The following foundational literature provides context for Green AI and the measurement of computational environmental impact.

1. Schwartz, R., Dodge, J., Smith, N. A., & Etzioni, O. (2020). Green AI. *Communications of the ACM*, 63(12), 54–63.  
   https://doi.org/10.1145/3381831

2. Strubell, E., Ganesh, A., & McCallum, A. (2019). Energy and Policy Considerations for Deep Learning in NLP. *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*.  
   https://doi.org/10.18653/v1/P19-1355

3. Henderson, P., Hu, J., Romoff, J., Brunskill, E., Jurafsky, D., & Pineau, J. (2020). Towards the Systematic Reporting of the Energy and Carbon Footprints of Machine Learning. *Journal of Machine Learning Research*, 21(248), 1–43.  
   https://jmlr.org/papers/v21/20-312.html

4. CodeCarbon. Carbon Emissions Tracking for Computing.  
   https://github.com/mlco2/codecarbon

5. UCI Machine Learning Repository. Adult Dataset.  
   https://archive.ics.uci.edu/dataset/2/adult

# 23. License

This repository is intended for academic research, experimentation, and educational purposes.

A formal open-source license should be added to the repository to define the terms under which the source code and research artifacts may be reused and distributed.

---

## Author

**Rohit Singh**

Research Interest: Green AI | Sustainable Machine Learning | AI Energy Efficiency | Carbon-Aware Computing | Machine Learning Systems

GitHub: https://github.com/rohitsinghrs

Project Repository: https://github.com/rohitsinghrs/Green-AI-Carbon-Benchmark

---

<p align="center">
  <b>Building AI Systems That Are Not Only Intelligent, But Also Sustainable.</b>
</p># Green-AI-Carbon-Benchmark
Reproducible benchmarking of machine learning architectures, energy consumption, computational efficiency, and carbon emissions across hardware platforms.
