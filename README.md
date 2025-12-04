# 3DGS With Priors
This project investigates augmenting the Splatter Image 3DGS model to accept geometry data additional to the base RGB of an image,
as to improve 3D reconstruction quality.
Additional geometry data is inferred from RGB images using specialized state-of-the-art models, allowing the input to Splatter Image 
to remain single or few RGB images.

This top level repository contains code for the entire project.

Code and notes related to research and generation of appropriate geometry priors can be found in the ```/geometry-priors``` folder.
<br>
A fork of Splatter Image with the appropriate modifications can be found in the ```/splatter-image``` submodule/folder. 
<br>
Test and related code can be found in ```/testing```.
<br>
Notes and latex for the final report can be found in the ```/report``` folder.
<br>
Code used to generate diagrams for the report can be found in the ```stats-code``` folder.

## Datasets
Due to time and compute limitations, the project only precomputes one dataset, namely ```SRN cars```.

#### ShapeNet Cars
The ```SRN cars``` dataset was used to generate ```srn_cars_priors``` containing precomputed priors alongside the base dataset contents, ```srn-cars_priors``` is available on HuggingFace at: https://huggingface.co/datasets/MVP-Group-Project/srn_cars_priors

To generate the priors locally:

1. Download the ```generate_cars_dataset.ipynb``` notebook
2. Fill in the relevant arguments in the top cell (can be left at default)
3. Run all cells

The notebook is an orchestrator notebook that will perform all the appropriate setup and run the models to generate a ready-to-upload modified dataset, like the one found on HuggingFace.
It is recommended to run the notebook on a service such as Google Colab, the existing dataset on HuggingFace was generated using the A100 runtime on Google Colab Pro+.

## Evaluation
An example notebook, that performs appropriate repo cloning, requirement setup and Gaussian splat rasterizer initialisation can be found in ```/eval_example.ipynb```.

Original instructions can be found at: https://github.com/szymanowiczs/splatter-image

### Using Pretrained models
Evaluation can be run on the standard Splatter Image datasets alongside the new ```cars_priors``` dataset option. 
If you are running evaluation on the standard datasets, you will have to manually download the datasets and then update ```/splatter-image/splatter_datasets/dataset_factory.py``` with the correct paths.
Running evaluation on ```cars_priors``` performs automatic download of the dataset, so no extra steps are necessary for this dataset. 

Required packages can be found in ```/splatter-image/requirements.txt```
<br>
After installing packages you need to run the following command to initialise the Gaussian Splat Rasterizer: 
```
!pip install -e submodules/diff-gaussian-rasterization
```

Evaluation can be run with:
```
python eval.py $dataset_name
```
`$dataset_name` is the name of the dataset. Options are:
- `gso` (Google Scanned Objects), 
- `objaverse` (Objaverse-LVIS), 
- `nmr` (multi-category ShapeNet), 
- `hydrants` (CO3D hydrants), 
- `teddybears` (CO3D teddybears), 
- `cars` (ShapeNet cars), 
- `cars_priors` (ShapeNet cars with predicted priors)
- `chairs` (ShapeNet chairs).

The code will automatically download the relevant model for the requested dataset.

### Using Local models
You can also train your own models and evaluate them with: 
```
python eval.py $dataset_name --experiment_path $experiment_path
```
`$experiment_path` should hold a `model_latest.pth` file and a `.hydra` folder with `config.yaml` inside it.

To evaluate on the validation split, use the option `--split val`.

To save renders of the objects with the camera moving in a loop, run evaluation with the option `--split vis`. With this option the quantitative scores are not returned since ground truth images are not available in all datasets.

You can set how many objects to save renders for, using the option `--save_vis`.

You can set where to save the renders, using the option `--out_folder`.

You can set where to save the resulting evaluation ```scores.txt```, using the option `--score_path`.

## Training
Original instructions can be found at: https://github.com/szymanowiczs/splatter-image

Ready weights for models trained on ```SRN Cars``` can be found on HuggingFace at: https://huggingface.co/MVP-Group-Project/splatter-image-priors/tree/main

To train modified models locally:
1. Download the ```training.ipynb`` notebook
2. Fill in the relevant arguments in the args cell (3rd cell from top - can be left at default)
3. Run the first 2 cells manually (repo clone and requirements installation cells), as the requirement installation will require restart of the session/notebook to allow use of the new packages
4. Run all remaining cells

NOTE: The training cell requires further user input to select logging options due to Splatter Image's use of wandb.

The notebook performs relevant set up for Splatter Image followed by training/fine-tuning, the last cells can be run to upload the resulting weights to HuggingFace.
Due to hardware demands, it is recommended to run the notebook on a service such as Google Colab, the existing models on HuggingFace were trained overnight using the A100 runtime on Google Colab Pro+.

## Testing
A variety of testing notebooks can be found in the ```/testing``` folder.

The ```eval_test.ipynb``` notebook performs tests to verify if the modified evaluation script 
successfully performs evaluation both on previous datasets (such as ShapeNet cars) and our new dataset
```cars_priors```.

The ```dataloader_test.ipynb``` notebook tests our custom Dataset and ready generated ```cars_priors``` data.
It does so by loading both the previous reference ```srn.py``` Dataset and our new ```srn_priors.py``` Dataset, walking both, comparing the resulting batches.

The ```graft_test.ipynb``` notebook performs tests checking whether the 
```graft_weights_with_channel_expansion``` function returns successfully, for a variety of model configurations. It additionally verifies whether an automatically grafted model's state dictionary exactly matches the state dictionary of a manually grafted reference model.

## Results
DESCRIBE HOW TO USE GENERATE STATS

ALSO NOTE TO CHANGE DATASET LOADER DIR