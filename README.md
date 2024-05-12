# 1. Requirements

```python
pip3 install torch torchvision
pip3 install -U scikit-learn
pip3 install numpy
pip3 install -U sklearn
pip3 install libmr
```
# 2. Datasets
Please download the dataset ant put it in 'user_dataset_path'

>Guanxiong Shen, Junqing Zhang, Alan Marshall, February 3, 2022, "LoRa_RFFI_dataset", IEEE Dataport, doi: https://dx.doi.org/10.21227/qqt4-kz19.

# 3. Guideline

## 3.1 Sythesize with pa nonlinearity
```bash
python main.py --task synthesis --dataset_path user_dataset_path
```
Assume the synthetic file path is `syn_path`

## 3.2 Train SFCR model with sythesized data
```bash
python main.py --task train --dataset_path user_dataset_path --samples_file 'syn_path'
```
