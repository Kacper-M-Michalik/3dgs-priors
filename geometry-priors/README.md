# Generating normal priors
## Model Repository
Inside the geometry-priors folder:
```
git clone https://github.com/baegwangbin/surface_normal_uncertainty.git
```

Your folder should now contain:
```
geometry-priors/
    generate_normals.py
    surface_normal_uncertainty/
```

## Dependencies
Recommended to create a virtual environment. 
```
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```
pip install -r normals_requirements.txt
```

## Checkpoint

Download the pretrained checkpoint from the author’s Google Drive: https://drive.google.com/drive/folders/1Ku25Am69h_HrbtcCptXn4aetjo7sB33F

Download `scannet.pt`

Place it under a new checkpoints folder within the cloned repo.
```
geometry-priors/
    surface_normal_uncertainty/
        checkpoints/
            scannet.pt     
```

## Directory Structure 
```
geometry-priors/
│
├── generate_normals.py
├── normals_requirements.txt
│
├── input_imgs/
│   └── srn_cars/
│       ├── cars_train/
│       ├── cars_test/
│       └── cars_val/
│
├── output_imgs/   (auto-created)
│
└── surface_normal_uncertainty/
    ├── checkpoints/
    │    └── scannet.pt
    └── ...
```

## Generation
Run
```
python generate_normals.py \
    --in_folder input_imgs \
    --out_folder output_imgs
```
