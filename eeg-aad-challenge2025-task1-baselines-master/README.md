<div align="center">

# Baselines for EEG-AAD 2025: EEG Auditory Attention Decoding Challenge

</div>

# Overview
Baseline implementation for the <a href='https://fchest.github.io/nips-aad/'>EEG-AAD Auditory Attention Decoding Challenge</a>.

# Data
We provide a multi-modal AAD (MM-AAD) dataset[1] consisting of EEG data collected from 40 subjects in two settings: audio-only and audio-visual scenes. 
Each subject was instructed to focus on one of two competing voices from the 90° left or right for an average of 55 minutes, resulting in a total of approximately 73.3 hours of data.
In the EEG auditory attention decoding challenge, the team will compete to build the best deep learning model to analyze the spatial orientation (left/right, 0/1) of the subject's attention to the speaker from EEG signals.

## Preprocessing
We used preprocessed data as starting point and we applied minimal preprocessing involving:
- 0.1 Hz and 50 Hz bandpass filter
- Remove 50 Hz noise 
- Ocular artifact removal via ICA
- Downsampled to 128 Hz 

## Architectures
We chose DARNet[2] as baselines for both tasks:
- w/o CSP in Task 1
- w CSP in Task 2
No significant changes were applied to the original architectures.

## Training
-Task1 Cross-subject: we provided 30 subjects for training and validation. We selected 4 subjects to serve as a separate held-out-subjects validation set (val_subject). To encourage participants to thoroughly explore model generalization and the robustness of validation strategies, this task allows and recommends that participants customize their validation set partitioning schemes.
Models were trained using the Adam optimizer for 100 epochs. For each subject of tasks, we get 6,580 decision windows. The model was then fed all the windows. Additionally, we set the batch size to 128 and utilized the Adam optimizer with a learning rate of 5e-4 and weight decay of 3e-4 to train the model.

-Task2 Cross-Session: This track requires participants to bulid AAD models capable of decoding auditory attention categories from EEG signals of scenarios that subjects have never seen before. Data from 30 subjects in two scenarios are provided, where the audio-visual scenario data from the same subject is used for training, and the audio-only scenario data is used for validation. 

## Inference
-Task1 Cross-subject: we provided 10 unseen subjects for test. For inference, same as for validation, each unseen test subject EEG data was first segmented into 6,580 decision windows. The optimal model saved from the validation set was used for inference.

-Task2 Cross-Session: For the testing phase, we provided 10 unseen subjects for test, each unseen test subject EEG data was first segmented into 6,580 decision windows, following the same process as validation: models trained on the audio-visual scenario are tested on data from the audio-only scenario. 

# Results
Our strategy yields the following results that serve as baseline

| DARNet            | Cross-subject | Cross-session |
|-------------------|---------------|---------------|
| Val Acc           |    53.1       |      54.9     |
                    

# How to run

### **Requirements**
-Python3.9
 pip install -r requirements.txt

All the baselies were tested on a single NVIDIA RTX 4090 GPU.


### **Train a model**

```
python train.py 
```

Model weights at best validation accuracy will be saved at exps/cross-subject


### **Inference**

This script will generate te required file for the final submission.
Always specify:
- the task 
- the model architectures
- the path (absolute or relative) to the folder with .pth file

As an example you can run:

``` 
python inference.py --model DARNet --resume exps/cross-subject/DARNet/baseline_2025-02-28-01-39-14
```

Running inference on **cross subject** will create a csv file named *results_cross_subject_test_subject.csv* for the held-out-subjects test set.

Each csv has only two columns:
- **id**: the id of the sample
- **prediction**: the predicted class


# References
[1] Cunhang Fan, Hongyu Zhang, Qinke Ni, Jingjing Zhang, Jianhua Tao, Jian Zhou, Jiangyan Yi, Zhao Lv, and Xiaopei Wu. eeing helps hearing: A multi-modal dataset and a mamba-based dual branch parallel network for auditory attention decoding. Information Fusion, page 102946, 2025.

[2] Sheng Yan, Cunhang Fan, Hongyu Zhang, Xiaoke Yang, Jianhua Tao, and Zhao Lv. Darnet: Dual attention refinement network with spatiotemporal construction for auditory attention detection. Advances in Neural Information Processing Systems,37:31688–31707, 2024. 
