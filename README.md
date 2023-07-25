## NF1 dataset 

**[Our current paper.](https://www.medrxiv.org/content/10.1101/2021.04.08.21255123v1)**

**[What is NF1?](https://www.cancer.net/cancer-types/neurofibromatosis-type-1)**

**[See some examples of NF1 images.](https://dermnetnz.org/topics/neurofibromatosis-images/)**

**_[Click to see an example of our survey to pediatricians and geneticists.](https://ncidccpssurveys.gov1.qualtrics.com/jfe/form/SV_2icqumxXrn2x2iG)_**


Please contact us for the data. 

## SIIM-ISIC dataset (you need to install [Kaggle API](https://github.com/Kaggle/kaggle-api))
Download the 2020 and 2019 data (which already had 2018 data) by Chris Deotte.

```
mkdir ./data
cd ./data
for input_size in 512 
do
  kaggle datasets download -d cdeotte/jpeg-isic2019-${input_size}x${input_size}
  kaggle datasets download -d cdeotte/jpeg-melanoma-${input_size}x${input_size}
  unzip -q jpeg-melanoma-${input_size}x${input_size}.zip -d jpeg-melanoma-${input_size}x${input_size}
  unzip -q jpeg-isic2019-${input_size}x${input_size}.zip -d jpeg-isic2019-${input_size}x${input_size}
  rm jpeg-melanoma-${input_size}x${input_size}.zip jpeg-isic2019-${input_size}x${input_size}.zip
done
```

Detail at: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175412
 
## Train, test, see pixel attribution
Use script in `Experiment/MakeScript.py`. 

## Ensemble classifier
We run a 5-fold cross-validation, and then ensemble these 5 models. Edit `ensemble_our_classifier.py` if needed. 

## Pre-trained models
Please contact us. 

