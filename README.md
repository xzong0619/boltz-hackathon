# Improving Boltz2 Predictions for Allosteric Ligands

## Challenge and Motivation

## Our Approach
We explored two complementary strategies to improve Boltz2's performance on allosteric ligand prediction

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

Features we included in the ML model:
- Confidence scores from Boltz2 outputs
- Physics-based features extracted from predicted pdb file: number of contacting residues, potential hydrogen bonds, binding site size, estimated binding pocket volume, chemical properties ligands etc

We think including these features would capture the physics between protein-ligand binding and serve well for our ML model.


## Key Findings
- Increasing the number of diffusion_samples and decreasing the step_scale both helps Boltz2 generate a more diverse set of poses, which leads to a higher probability of capturing the allosteric poses. We observe an improvement in RMSD for allosteric ligands after we changed these parameters.
- Finding 2
- Finding 3

## Future Directions



## Team Members
- Song Yin
- Jingqian Liu
- Xue Zong
