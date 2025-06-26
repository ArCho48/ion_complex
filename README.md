# Automated Learning of GNN Ensembles for Predicting Redox Potentials with Uncertainty
PyTorch implementation of Automated Learning of GNN Ensembles for Predicting Redox Potentials with Uncertainty (https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/683e0487c1cb1ecda0ce5640/original/automated-learning-of-gnn-ensembles-for-predicting-redox-potentials-with-uncertainty.pdf)

## Overview
This library contains a PyTorch implementation of Automated Learning of GNN Ensembles for Predicting Redox Potentials with Uncertainty as presented in [[1]](#citation)(https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/683e0487c1cb1ecda0ce5640/original/automated-learning-of-gnn-ensembles-for-predicting-redox-potentials-with-uncertainty.pdf).

## Dependencies

* **python>=3.6**
<!-- * **tensorflow>=1.14.0**: https://tensorflow.org -->
* **numpy**
* **matplotlib**

## Structure
<!-- * [datagen](https://github.com/ArCho48/Unrolled-WMMSE/blob/master/datagen.py): Code to generate dataset. Generates A.pkl ( Geometric graph ), H.pkl ( Dictionary containing train_H and test_H ) and coordinates.pkl ( node position coordinates ).  Run as *python3 datagen.py* \[dataset ID\]. User chosen \[dataset ID\] will be used as the foldername to store dataset. Eg., to generate dataset with ID *set3*, run *python3 datagen.py set3*.
* [data](https://github.com/ArCho48/Unrolled-WMMSE/tree/master/data): should contain your dataset in folder \[dataset ID\]. 
* [main](https://github.com/ArCho48/Unrolled-WMMSE/blob/master/main.py): Main code for running the experiments in the paper. Run as *python3 main.py* \[dataset ID\] \[exp ID\] \[mode\]. Eg., to train UWMMSE on dataset with ID set3, run *python3 main.py set3 uwmmse train*.
* [model](https://github.com/ArCho48/Unrolled-WMMSE/blob/master/model.py): Defines the UWMMSE model.
* [models](https://github.com/ArCho48/Unrolled-WMMSE/tree/master/models): Stores trained models in a folder with same name as \[dataset ID\]. -->

## Usage


Please cite [[1](#citation)] in your work when using this library in your experiments.

## Feedback
For questions and comments, feel free to contact [Arindam Chowdhury](mailto:chowdhurya1@ornl.gov).

## Citation
```
[1] Chowdhury A, Harb H, Alves C, Doan HA, Egele R, Assary RS, et al. Automated Learning of GNN Ensembles for Predicting Redox Potentials with Uncertainty. ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-0tq7j 
```

BibTeX format:
```
@article{chowdhury2025automated,
  title={Automated Learning of GNN Ensembles for Predicting Redox Potentials with Uncertainty},
  author={Chowdhury, Arindam and Harb, Hassan and Alves, Caio and Doan, Hieu Anh and Egele, Romain and Assary, Rajeev Surendran, Balaprakash, Prasanna},
  journal={chemarXiv preprint 10.26434/chemrxiv-2025-0tq7j},
  year={2025}
}

```