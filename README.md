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
Notes and latex for the final report can be found in the ```/report``` folder.

## Datasets
Due to time and compute limitations, the project only modifies one dataset, namely ```SRN cars```.

#### ShapeNet Cars
The ```SRN cars``` dataset containing ready-generated predicted priors is available on HuggingFace at: https://huggingface.co/datasets/MVP-Group-Project/srn_cars_priors

To generate the priors locally:

1. Download the ```generate_cars_dataset.ipynb``` notebook
2. Fill in the relevant arguments in the top cell (can be left at default)
3. Run all cells

The notebook is an orchestrator notebook that will perform all the appropriate setup and run the models to generate a ready-to-upload modified dataset, like the one found on HuggingFace.
It is recommended to run the notebook on a service such as Google Colab, the existing dataset on HuggingFace was generated using the A100 runtime on Google Colab Pro+.

## Training
Ready weights for models trained on ```SRN Cars``` can be found on HuggingFace at: https://huggingface.co/MVP-Group-Project/splatter-image-priors/tree/main

To train modified models locally:

1. Download the ```training.ipynb`` notebook
2. Fill in the relevant arguments in the top cell (can be left at default)
3. Run the first 3 cells manually (args, repo clone and requirements installation cells), as the requirement installation will require restart of the session/notebook to allow use of the new packages
4. Run all remaining cells

The notebook performs relevant set up for Splatter Image followed by training/fine-tuning, the last cells can be run to upload the resulting weights to HuggingFace.
Due to hardware demands, it is recommended to run the notebook on a service such as Google Colab, the existing models on HuggingFace were trained overnight using the A100 runtime on Google Colab Pro+.

## Evaluation
DESCRIBE HOW TO EVAL, WHERE RESUTLS FOUND

## Results
DESCRIBE HOW TO USE GENERATE STATS

ALSO NOTE TO CHANGE DATASET LOADER DIR