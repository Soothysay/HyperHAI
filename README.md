# This repo contains the Appendix and code to replicate the experiments of our method for MIMIC-IV Dataset for our SDM 2025 paper "Domain Knowledge Augmented Contrastive Learning on Dynamic Hypergraphs for Improved Health Risk Prediction"

## Instructions
1. P2id_mimic.csv is the mapping file that maps pid to an index
2. feat_create_mimic.py creates the dynamic patient features
3. create_interactions_mimic.py creates the interactions between patients and hospital entities
4. contrastive_views_mimic.py creates the 2 contrastive views of the interactions
5. create_hyperedges.py creates the hyperedges from the interaction data
6. train_mimic_my.py is the training and evaluation script
