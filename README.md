# 3DGS With Priors
## Datasets
Ready-generated predicted priors are available on HuggingFace at: X

To generate the priors locally:
#### ShapeNet Cars
Download the 'generate_cars_dataset.ipynb' notebook and run it on a GPU based Google Colab runtime,
the notebook is an orchestrator that will setup vm's, download the original SplatterImage cars dataset
and then run our selected models, generating the appropriate priors in a file/directory format similar to the original dataset.

## Training
Ready weights can be found on HuggingFace at: X

To train modified models, download the 'training.ipynb' notebook and run it on a GPU based Google Colab runtime.
The notebook will train appropriate models and save the weights to HuggingFace if desired.