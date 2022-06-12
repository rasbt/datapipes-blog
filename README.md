## About

This is the complementary code for the article [Taking Datasets, DataLoaders, and PyTorch’s New DataPipes for a Spin](https://sebastianraschka.com/blog/2022/datapipes.html).



To recreate the environment used in this blogpost, you can use 


```bash
conda create --name datapipes python=3.8
conda activate datapipes
pip install -r requirements.txt
```



The code was run using

```
Python version: 3.8.13

torch: 1.11.0
torchdata: 0.3.0
```



## Running the Code



Run the [`0_download-and-prep-data.ipynb`](0_download-and-prep-data.ipynb) notebook first to download the dataset. The data loading scripts are all independent and self-contained and can be run in any order

- [1_dataset-csv.py](1_dataset-csv.py): `python 1_dataset-csv.py`
- [2_imagefolder.py](2_imagefolder.py):  `python 2_imagefolder.py`
- [3_datapipes-csv.py](python 3_datapipes-csv.py): `python 3_datapipes-csv.py`
