# GradPSPStain
Implementation of the paper **"GradPSPStain: Learning Grad-CAM-Guided Attention for
H&E-to-IHC Virtual Staining"** (ICVGIP 2026)

## Requirements
```bash
conda env create -f environment.yml
```


## Dataset
- Breast Cancer Immunohistochemical (BCI) challenge dataset: HER2 biomarker
- Multi-IHC Stain Translation (MIST) dataset: HER2 biomarker

where the dataset structure is 
```bash
dataset/
│
├── trainA/
    ├── HE
├── trainB/
    ├── IHC
├── valA/
├── valB/

```

## Training and Testing 
We use ```experiments/GradPSPStain_launcher.py``` to generate the command line arguments for training and testing. 
```bash
python -m experiments --name GradPSPStain --cmd train --id 0
python -m experiments --name GradPSPStain --cmd test --id 0
```

## Acknowledgement
The repo is built upon https://github.com/ccitachi/PSPStain
