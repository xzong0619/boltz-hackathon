# Improving Boltz2 Predictions for Allosteric Ligands

## Challenge and Motivation

## Our Approach
We explored two complementary strategies to improve Boltz2's performance on allosteric ligand prediction:

### 1. Parameter Optimization

Investigating optimal Boltz2 parameters to enhance allosteric pose generation. The questions we've explored include:

- What parameter configurations improve prediction accuracy?
- Do different parameter sets generate more diverse allosteric poses?
- Which parameters have the most significant impact on allosteric site prediction?

### 2. Post-Processing and Reranking
Developing ML models to rerank and select the most promising allosteric compounds from generated poses. For developing the ML models, we took Boltz2 predictions including confidence scores and predicted protein-ligand structures,
then we extracted physics-based features out of the poses and built ML models to predict RMSD values (calculated from predicted poses vs allosteric ground truth).
To train and validate the ML model, we use the provided public set including 40 ligands. To do the post-processing and reranking, we use our pre-trained model and predict the test set.


## Approach 1: Parameter Optimization

Parameters Explored
- Parameter 1: --diffusion_samples
- Parameter 2: --step_scale
- Parameter 3: --use_potentials
- Parameter 4: --recycling_steps

## Approach 2: ML Model for Reranking




## Results

## Key Findings
- Finding 1
- Finding 2
- Finding 3

## Future Directions


## Team Members
- Song Yin
- Jingqian Liu
- Xue Zong
