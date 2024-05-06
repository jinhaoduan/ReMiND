
# ReMiND: Recovery of Missing Neuroimaging using Diffusion Models with Application to Alzheimer's Disease

## Evnironment

```shell
conda create -n remind
conda activate remind
pip install -r requirements.txt
```

## Dataset
### Data Collection and Preparation
Please collect your MRIs from [ADNI](https://adni.loni.usc.edu/data-samples/access-data/). 
ReMiND use the [ANTs Longitudinal-SST](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10204115/) for MRI preprocessing.
### Train/Val/Test splits
ReMiND takes `train/val-list.txt` for model training and `test-list.txt` for generation.
Please organize your splits into the following format:
```
<past-visit-mri-path> <current-visit-mri-path> <future-visit-mri-path> <past-visit-id> <current-visit-id> <future-visit-id> <past-visit-stage> <current-visit-stage> <future-visit-stage>
# stage could be NL, MCI, or AD
...
```

## Model Training
Please refer to the following script for ReMiND-PF training:
```
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/adni_remind_pf.yml --exp remind-pf --ni --data_path <path-to-your-data-root>
```
Training hyperparameters can be tuned by modifying `./configs/adni_remind_pf.yml`:
+ `data.data_root`: path to your dataset
+ `data.train_list_path`: path to the training split file
+ `data.val_list_path`: path to the validation split file
+ `data.test_list_path`: path to the test split file
+ `data.channels`: number of segments
+ `data.cont_local`: number of slices in a local clip

### Pretrained Model 
The pretrained ReMiND-PF is provided in [here](https://drive.google.com/drive/folders/1pvRs7RkwTQckf2gAhrL8e7dbBuiHHVkt?usp=sharing).
## Model Inference
Please refer to the following script for missing MRI interpolation:
```shell
python generation.py \
--test-listpath <path-to-the-test-list> \
--checkpoint-path <path-to-your-checkpoint> \ # the config.yml should be in the same folder as the checkpoint
--save-as-nii 
```
The interpolated MRI will be saved at `./interpolated-<current-visit-name>.nii`
# Acknowledgement
This codebase is built on top of [mcvd-pytorch](https://github.com/voletiv/mcvd-pytorch). Thanks for their excellent contribution.

# Reference

Please cite our paper as
```
@article{yuan2023remind,
  title={Remind: Recovery of missing neuroimaging using diffusion models with application to alzheimer’s disease},
  author={Yuan, Chenxi and Duan, Jinhao and Tustison, Nicholas J and Xu, Kaidi and Hubbard, Rebecca A and Linn, Kristin A},
  journal={medRxiv},
  year={2023},
  publisher={Cold Spring Harbor Laboratory Preprints}
}
```