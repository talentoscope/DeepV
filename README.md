## Deep Vectorization of Technical Drawings [[Web page](http://adase.group/3ddl/projects/vectorization/)] [[Paper](https://arxiv.org/abs/2003.05471)] [[Video](https://www.youtube.com/watch?v=lnQNzHJOLvE)] [[Slides](https://drive.google.com/file/d/1ZrykQeA2PE4_8yf1JwuEBk9sS4OP8KeM/view?usp=sharing)]
Official Pytorch repository for ECCV 2020 paper [Deep Vectorization of Technical Drawings](https://link.springer.com/chapter/10.1007/978-3-030-58601-0_35)

![alt text](https://drive.google.com/uc?export=view&id=191r0QAaNhOUIaHPOlPWH5H4Jg7qxCMRA)

## Repository Structure

To make the repository user-friendly, we decided to stick with - module-like structure.
The main modules are cleaning, vectorization, refinement, and merging(each module has an according to folder).
Each folder has Readme with more details. Here is the brief content of each folder.

* cleaning - model, script to train and run, script to generate synthetic data 
* vectorization - NN models, script to train
* refinement - refinement module for curves and lines
* merging - merging module for curves and lines
* notebooks - a playground to show some function in action
* utils - loss functions, rendering, metrics
* scripts - scripts to run training and evaluation

## Requirments
Linux system  
Python 3

See requirments.txt and additional packages

cairo==1.14.12  
pycairo==1.19.1  
chamferdist==1.0.0


## Compare 

To compare with us without running code, you can download our results on the full pipeline on the test set
from the project website or contact the authors.


## Notebooks

To show how some of the usability of the functions, there are several notebooks in the notebooks folder.
1) Rendering notebook
2) Dataset loading, model loading, model training, loss function loading
3) Notebook that illustrates  how to work with pretrained model and how to do refinement on lines(without merging)
4) Notebook that illustrates how to work with pretrained model and how to do refinement on curves(without merging)

## Models

Download pretrained models for [curve](https://disk.yandex.ru/d/yOZzCSrd-QSACA)
and for [line](https://disk.yandex.ru/d/FKJuMvNJuy-K9g) .

## How to run
1. Download models.
2. Either use Dockerfile to create docker image with needed environment or just install requirements
3. Run scripts/run_pipeline.sh with correct paths for trained model, data dir and output dir. Don't forget to chose primitive type and primitive count in one patch.

P.s. currently cleaning model not included there.

## Benchmarking and Evaluation

DeepV includes a comprehensive benchmarking pipeline for evaluating vectorization models across multiple datasets and comparing against state-of-the-art baselines.

### Quick Benchmarking

```bash
# Run evaluation on synthetic dataset
python scripts/benchmark_pipeline.py \
  --data-root /path/to/datasets \
  --deepv-model-path /path/to/trained/model \
  --datasets synthetic

# Run comprehensive benchmark across multiple datasets
python scripts/benchmark_pipeline.py \
  --data-root /path/to/datasets \
  --deepv-model-path /path/to/trained/model \
  --datasets dataset1 dataset2 dataset3 \
  --output-dir benchmark_results
```

### Supported Dataset Formats
- PNG + DXF format pairs (image + ground truth)
- SVG vector graphics
- PDF technical drawings
- Any custom dataset following standard directory structure

### Evaluation Metrics
- **Vector Metrics**: F1 Score, IoU, Hausdorff Distance, Chamfer Distance
- **Raster Metrics**: PSNR, MSE, SSIM
- **Comprehensive Reports**: Automated comparison against baselines

See `scripts/README_benchmarking.md` for detailed usage instructions.

Quick developer checks
---------------------

Quick developer checks
---------------------

This repository is maintained as a personal project. There is no CI configured — run all checks and tests locally before pushing.

Before running heavy training or the full pipeline, run the environment validator and the test suite locally:

```bash
# Validate Python and key packages
python scripts/validate_env.py

# Run unit tests (recommended inside a virtualenv or container)
pip install -r requirements.txt
pip install pytest
pytest -q
```

   
## Dockerfile 

Build the docker image:

```bash
docker build -t Dockerfile owner/name:version .
```
example:
```bash
docker build -t vahe1994/deep_vectorization:latest .
```


When running container mount folder with reporitory into code/, folder with datasets in data/ folder with logs in logs/
```bash
docker run --rm -it --shm-size 128G -p 4045:4045 --mount type=bind,source=/home/code,target=/code --mount type=bind,source=/home/data,target=/data --mount type=bind,source=/home/logs,target=/logs  --name=container_name owner/name:version /bin/bash
```

Windows / WSL notes
-------------------

On Windows use WSL2 or Docker Desktop with WSL integration enabled. Example (powershell / WSL):

```powershell
# Build image (run in repo root)
docker build -t deepv:latest .

# Run container (example mounts for Windows paths)
docker run --rm -it --shm-size 128G -p 4045:4045 \
  --mount type=bind,source="C:/path/to/DeepV",target=/code \
  --mount type=bind,source="C:/path/to/data",target=/data \
  --mount type=bind,source="C:/path/to/logs",target=/logs \
  --name deepv-container deepv:latest /bin/bash
```

If using WSL, use the Linux paths from inside WSL (e.g., `/home/username/...`) when mounting.

Activating the packaged environment inside container:

```bash
. /opt/.venv/vect-env/bin/activate/
```

Note on `util_files` helpers
----------------------------

This repository previously included `util_files/os.py` which shadowed the Python standard library `os` module. It has been renamed to `util_files/file_utils.py` to avoid conflicts. When editing or importing utilities, prefer:

- `from util_files import file_utils as fu` (explicit alias), or
- `from util_files.file_utils import require_empty`


Anaconda with packages are installed in follder opt/ . Environement with packages that needed are installed in environment vect-env.
. To activate it run in container
```bash
. /opt/.venv/vect-env/bin/activate/
```

## How to train
Look at vectorization /srcipts/train_vectorizatrion 

### Citing
```
@InProceedings{egiazarian2020deep,
  title="Deep Vectorization of Technical Drawings",
  author="Egiazarian, Vage and Voynov, Oleg and Artemov, Alexey and Volkhonskiy, Denis and Safin, Aleksandr and Taktasheva, Maria and Zorin, Denis and Burnaev, Evgeny",
  booktitle="Computer Vision -- ECCV 2020",
  year="2020",
  publisher="Springer International Publishing",
  address="Cham",
  pages="582--598",
  isbn="978-3-030-58601-0"
}
```
