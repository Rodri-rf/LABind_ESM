# LABind-ESM: Protein Binding Site Prediction with ESM2-3B

[![DOI](https://zenodo.org/badge/879762327.svg)](https://doi.org/10.5281/zenodo.15871907)

## Overview

This repository is a **fork of the original LABind model** [[Zhang et al., 2025]](https://doi.org/10.1038/s41467-025-62899-0) that replaces the Ankh protein language model with **ESM2-3B** from Meta AI's Evolutionary Scale Modeling family. The modified architecture, **LABind-ESM**, maintains the original LABind framework while leveraging ESM2's richer 2560-dimensional per-residue embeddings for protein-ligand binding site prediction.

### Key Modifications

The original LABind architecture uses three types of inputs:
- **Ligand sequences** (SMILES format) → embedded via MolFormer
- **Protein sequences** → embedded via Ankh (1536-dim)
- **Protein structures** → encoded via DSSP (20-dim)

**LABind-ESM** makes the following changes:
- ✅ **Replaced Ankh with ESM2-3B** (2560-dim embeddings)
- ✅ **Adjusted combined feature dimension** from 1556 to 2580 (ESM2 + DSSP)
- ✅ **All other architecture components unchanged** (graph transformer layers, attention mechanisms, MLP classifier)

This implementation explores whether ESM2's evolutionary and structural patterns learned from larger-scale training data can improve binding site prediction accuracy.

### Rationale

Protein Language Models (PLMs) like Ankh and ESM2 are transformer-based networks trained on vast protein sequence databases. ESM2-3B offers:
- **Richer embeddings**: 2560 dimensions vs. Ankh's 1536
- **Larger training corpus**: Multiple sequence alignments capturing evolutionary patterns
- **Strong transfer learning**: Proven performance on downstream structural prediction tasks

By integrating ESM2 into LABind's graph-based interaction learner, we aim to capture more detailed residue-level features relevant to binding site prediction while maintaining the original architecture's proven structural reasoning capabilities.

![LABind Architecture](./architecture.png)

## Original LABind

LABind is a structure-based method to predict protein binding sites for ions and small molecules in a ligand-aware manner. This fork builds upon their published work:

**Citation:**
```bibtex
@article{zhangLABindIdentifyingProtein2025,
  title = {{{LABind}}: Identifying Protein Binding Ligand-Aware Sites via Learning Interactions between Ligand and Protein},
  author = {Zhang, Zhijun and Quan, Lijun and Wang, Junkai and Peng, Liangchen and Chen, Qiufeng and Zhang, Bei and Cao, Lexin and Jiang, Yelu and Li, Geng and Nie, Liangpeng and Wu, Tingfang and Lyu, Qiang},
  year = {2025},
  journal = {Nat. Commun.},
  volume = {16},
  number = {1},
  pages = {7712},
  issn = {2041-1723},
  doi = {10.1038/s41467-025-62899-0}
}
```

**Original Repository:** [https://github.com/ljquanlab/LABind](https://github.com/ljquanlab/LABind)

## Preparation
Clone this repository by `git clone https://github.com/ljquanlab/LABind.git` or download the code in ZIP archive.

LABind is tested on the Linux system, and the system information is as follows:

```sh
$lsb_release -a
Distributor ID: Ubuntu
Description:    Ubuntu 20.04.4 LTS
Release:        20.04
Codename:       focal
$ldd --version
ldd (Ubuntu GLIBC 2.31-0ubuntu9.7) 2.31
$nvidia-smi
NVIDIA-SMI 535.113.01
Driver Version: 535.113.01
$nvcc -V
Cuda compilation tools, release 10.1, V10.1.243
```

LABind primarily relies on the following Python packages:
- python=3.8.19
- cuda=11.8
- torch=2.1.1
- biopython=1.83
- transformers=4.39.3
- scikit-learn=1.3.2
- panydas=2.0.3
- numpy=1.24.3
- scipy=1.10.1
- tqdm=4.66.2
- lxml=5.2.1
- periodictable=1.7.0
- accelerate=0.30.1

In case you want to use conda for your own installation please create a new LABind environment.
We showed an example of creating an environment.
```sh
conda create -n LABind python=3.8.19
conda activate LABind
conda install pytorch==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install biopython=1.83 transformers=4.39.3 scikit-learn=1.3.2 pandas=2.0.3 numpy=1.24.3 scipy=1.10.1 -c conda-forge
pip install lxml==5.2.1 periodictable==1.7.0 accelerate==0.30.1
```

Or you can use the provided [environment.yml](./environment.yml) to create all the required dependency packages.
```sh
conda env create -f environment.yml
```
Similarly, you can build a Docker image to run LABind.
```sh
docker build -t labind .
docker run -it --gpus all --name=LABind labind  /bin/bash
```
> This will take a few minutes (5-10 minutes) to deploy.

### Pre-trained Model Requirements

**LABind-ESM** requires the following pre-trained models:
- **[ESM2-3B](https://huggingface.co/facebook/esm2_t36_3B_UR50D)** - Replaces Ankh for protein sequence embedding
- **[ESMFold_v1](https://huggingface.co/facebook/esmfold_v1)** - For structure prediction (if PDB files not provided)
- **[MolFormer-XL-both-10pct](https://huggingface.co/ibm/MoLFormer-XL-both-10pct)** - For ligand SMILES embedding

Download these models from HuggingFace and update the paths in `scripts/config.py`. You can also use `python ./scripts/download_weights.py -o <path>` to download model weights automatically.

**Note:** The original LABind uses Ankh-large. This fork **does not require Ankh** and instead uses ESM2-3B.

Then add permission to execute for DSSP and MSMS by `chmod +x ./tools/mkdssp ./tools/msms`

## Architectural Details

### Model Architecture Comparison

| **Component** | **Original LABind (Ankh)** | **LABind-ESM** |
|---------------|---------------------------|----------------|
| Protein Language Model | Ankh-large | **ESM2-3B** |
| Sequence Embedding Dim | 1536 | **2560** |
| Combined Feature Dim (w/ DSSP) | 1556 | **2580** |
| Hidden Dimension | 256 | 256 |
| Attention Heads | 4 | 4 |
| Transformer Layers | 4 | 4 |
| Graph Neighbors (top-k) | 30 | 30 |

### Feature Processing Pipeline


1. **Ligand Representation**: SMILES → MolFormer → 768-dim embedding
2. **Protein Sequence**: Amino acids → ESM2-3B → 2560-dim per-residue embeddings
3. **Protein Structure**: PDB → DSSP → 20-dim secondary structure features
4. **Combined Features**: [ESM2 ∥ DSSP] → 2580-dim per-residue features
5. **Graph Construction**: Residues as nodes, top-30 nearest neighbors as edges
6. **Interaction Learning**: 4-layer graph transformer with cross-attention to ligand
7. **Classification**: MLP → per-residue binding probabilities


## Validation
Checkpoints trained on the DS3 dataset are provided in `./model/Unseen/`. If you need to validate our results, you can modify the checkpoint used by LABind in the `./scripts/config.py`.
We have uploaded all the data files and checkpoints to [Zenodo](https://zenodo.org/records/15692081).

### Prediction
We provide examples in `./example`. First, you must provide a FASTA file. If `smiles.txt` is not available, LABind will automatically retrieve it, so please ensure your device can connect to the RCSB database.

An example of a FASTA file is shown below:
```fasta
>8iw5B ZN
SLRALHLVEDLRGLLEMMETDEKEGLRCQIPDSTAEVLIEWLQN
>8oerI CA
CVREVCRWSSWYNGHRPEPGLGGGDFETFENLRQRGYQVCPVLADIECRAAQLPDMPLEELGQQVDCDRMRGLMCANSQQSPPLCHDYELRVLCCEYVPC
>5ldpA ATP
SEPQRLFFAIDLPAEIREQIIHWRAKHFPPEAGRPVAADNLHLTLAFLGEVSAEKEKALSLLAGRIRQPGFTLTLDDAGQWLRSRVVWLGMRQPPRGLIQLANMLRSQAARSGCFQSNRPFHPHITLLRDASEAVTIPPPGFNWSYAVTEFTLYASSFARGRTRYTPLKRWALTQ
>8ppzB 0AN
MDPEFMEMWHEGLEEASRLYFGERNVKGMFEVLEPLHAMMERGPQTLKETSFNQAYGRDLMEAQEWCRKYMKSGNVKDLTQAWDLYYHVFRRIS
```
> Please make sure to use the same format as the example. Deviations may result in unexpected errors or issues during processing.

Then run:
```bash
cd scripts/
python prediction.py -op ../example/out/ -if ../example/protein.fa -ip ../example/pdb/ -il ../example/smiles.txt
```
And the prediction results will be saved in `./example/out/RESULT.csv`. 

Other parameters:
```
--cluster           Whether to cluster the binding residues, default=False
-b  BATCH           Batch size for LABind prediction, default=1
-g  GPU_LIST        Gpu list for LABind prediction, default=0
```
> For instance, `-b 4 -g 0 1`

### Retrain
We have provided a detailed workflow to reproduce the results presented in the paper, using training and validation based on DS3 as an illustrative example. (`./scripts/reproduction.ipynb`) If you need to retrain the model, you can try modifying `./scripts/wandb.py` for training, which requires installing the wandb package. `pip install wandb`  

If you need to create a dataset based on the latest version of the [BioLiP2](https://zhanggroup.org/BioLiP/index.cgi) database, you can use `./scripts/create_ds.py` to generate a new dataset.
```bash
python create_ds.py -o <out_path> -m <max_length>
```


## Dataset

**LABind-ESM** is trained and evaluated on the **DS1 (LigBind)** dataset from the original LABind paper:
- **10,341** non-redundant proteins
- **11,121** protein-ligand binding samples
- **72,029** binding sites vs. **3,734,243** non-binding sites (~1.9% positive class)
- **19 different ligands** (metal ions and small molecules)

Data files and checkpoints are available at: [Zenodo](https://zenodo.org/records/15692081)

## Training Configuration

### 5-Fold Cross-Validation

All reported metrics use stratified 5-fold cross-validation with `random_state=42` for reproducibility.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.0004 (Adam) |
| Batch Size | 1 |
| Training Epochs | 70 |
| Early Stopping Patience | 10 |
| Dropout | 0.1 |
| Attention Dropout | 0.1 |
| Coordinate Augmentation (ε) | 0.05 |

## Contacts

**LABind-ESM Fork:**
- Rodrigo Adolfo Reyes Feregrino - rodrigo.reyesferegrino@utoronto.ca
- Xavier Alejandro Ibanez-Padron - xavier.ibanezpadron@mail.utoronto.ca
- Leon Alexander Nitsch - leon.nitsch@mail.utoronto.ca
- Zixin Zeng - zxin.zeng@mail.utoronto.ca

**Original LABind:**
- zjzhang21@stu.suda.edu.cn
- ljquan@suda.edu.cn

## License

This project is licensed under the terms of the MIT license. See [LICENSE](./LICENSE) for additional details.

## Source Code and Data

- **LABind-ESM Repository:** [https://github.com/Rodri-rf/LABind_ESM](https://github.com/Rodri-rf/LABind_ESM)
- **Project Data (Zenodo):** [https://zenodo.org/records/15875006](https://zenodo.org/records/15875006)
- **Original LABind Repository:** [https://github.com/ljquanlab/LABind](https://github.com/ljquanlab/LABind)

## Citation

If you use LABind-ESM in your research, please cite both this implementation and the original LABind paper:

**Original LABind:**
```bibtex
@article{zhangLABindIdentifyingProtein2025,
  title = {{{LABind}}: Identifying Protein Binding Ligand-Aware Sites via Learning Interactions between Ligand and Protein},
  author = {Zhang, Zhijun and Quan, Lijun and Wang, Junkai and Peng, Liangchen and Chen, Qiufeng and Zhang, Bei and Cao, Lexin and Jiang, Yelu and Li, Geng and Nie, Liangpeng and Wu, Tingfang and Lyu, Qiang},
  year = {2025},
  journal = {Nat. Commun.},
  volume = {16},
  number = {1},
  pages = {7712},
  issn = {2041-1723},
  doi = {10.1038/s41467-025-62899-0}
}
```

**ESM2:**
```bibtex
@article{lin2022esm2,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and Verkuil, Robert and Kabeli, Ori and Shmueli, Yair and others},
  journal={Science},
  year={2023}
}
```

## Acknowledgments

This work was completed as part of CSC413 (Neural Networks and Deep Learning) at the University of Toronto Mississauga. We thank the original LABind team for their excellent open-source implementation and the Meta AI team for the ESM2 models.
