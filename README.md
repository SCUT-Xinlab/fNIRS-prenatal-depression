<center><h1>Attention-Based-Features-Fusion
Emotion-Guided fNIRS Classification Network
for Prenatal Depression Recognition 🤰</h1></center>

South China University of Technology</br>
The Third Affiliated Hospital, Sun Yat-sen University

> **Abstract.** Using functional Near-Infrared Spectroscopy (fNIRS) for prenatal depression recognition can improve accuracy compared to relying solely on rating scales. However, due to the low signal-to-noise ratio (SNR) and limited data in fNIRS, deep learning based classification methods still face challenges. Previous works primarily exhibit two main limitations: they lack manually extracted statistical features to guide temporal models, and they only use one single task fNIRS as classification features. Researchers have found that the emotional responses of individuals with depression differ from those of healthy individuals, and that women become more emotionally sensitive during pregnancy. Based on these observations, we design three different tasks for the collection of fNIRS signals: happy stimulation, sad stimulation, and resting status. Both the temporal and statistical features of fNIRS are utilized as inputs for the model. We propose an Attention-Based-Features-Fusion Emotion-guided network, which integrates distinct features from various tasks to obtain the classification results. The proposed model achieves the best results on a dataset with 27 subjects, verifying its effectiveness.

### 1. Enviroment

The version of `torch` is `1.8.1+cu111`.

### 2. Data

Our data has not been released. However, in this project, we provide sample data and retain the structure of the data files.  We hope this makes it easier for you to understand how the code works :-) 

### 3. Acknowledgement

> This work is funded by Key-Area Research and Development Program of Guang- dong Province (2023B0303040001), Guangdong Basic and Applied Basic Re- search Foundation (2024A1515010180), National Natural Science Foundation of China (No. 82070606 to Chengfang Xu), Natural Science Foundation of Guang- dong (No. 2021A1515011441 to Chengfang Xu), Guangdong Provincial Key Lab- oratory of Human Digital Twin (2022B1212010004), and Natural Science Foun- dation of Guangdong (No. 2024A1515010538 to Chengfang Xu).

### 4. BibTeX

```latex
inproceedings{yu2024attention,
  title={Attention-Based-Features-Fusion Emotion-Guided fNIRS Classification Network for Prenatal Depression Recognition},
  author={Yu, Sijin and Li, Xuejiao and Lei, Huirong and Yao, Yingxue and Chen, Zhaojin and Zheng, Zicong and Liang, Guodong and Xing, Xiaofen and Zhang, Xin and Xu, Chengfang},
  booktitle={International Workshop on PRedictive Intelligence In MEdicine},
  pages={12--23},
  year={2024},
  organization={Springer}
}
```

