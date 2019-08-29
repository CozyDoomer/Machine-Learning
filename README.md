# Machine Learning

This is a repo for different machine learning tasks, most of them come from Kaggle competitions.

I wrote everything in Python3 using Jupyter Notebooks in an Anaconda3 environment.

Examples contain Code for both structured and unstructured data and are mostly for showcasing the code because the data I used was stored locally.

### Notebooks that are somewhat cleaned

[Whale Identification (one-shot learning, siamese network)](https://github.com/DollofCuty/Deep-Learning/blob/master/unstructured-data/image/whale-identification/siamese_network_cropped.ipynb)

[Protein Classification (Multilabel classification, resnet50)](https://github.com/DollofCuty/Deep-Learning/blob/master/unstructured-data/image/protein-detection/resnet_strat_validation.ipynb)

[NLP example](https://github.com/DollofCuty/Deep-Learning/blob/master/unstructured-data/nlp/quora-insincere-questions/fastai_text.ipynb)

[structured data example and feature engineering](https://github.com/DollofCuty/Deep-Learning/blob/master/structured-data/google-analytics-customer-revenue/lgb_feature_score.ipynb)

[plotting shenanigans](https://github.com/DollofCuty/Deep-Learning/blob/master/structured-data/plotting_baseline.ipynb)

[Style Transfer (for fun :D )](https://github.com/DollofCuty/Deep-Learning/blob/master/unstructured-data/image/style-transfer/style_transfer_fastai.ipynb)

## Most interesting competitions so far were 

## Unstructured Data

### Protein Classification Challenge

In this competition, you will develop models capable of classifying mixed patterns of proteins in microscope images. The Human Protein Atlas will use these models to build a tool integrated with their smart-microscopy system to identify a protein's location(s) from a high-throughput image.

| Dataformat   |      Metric      |  Prediction |
|----------|:-------------:|------:|
| 4 channel image | macro F1 Score | multi label classification |

Score: 
public 0.591, 118/2172

Focalloss worked way better than binary cross entropy

Started with resnet34 using fastai for multilabel-classification
resnet50 worked even better (by about 0.05 macro F1-score)

### Humpback Whale Identification

In this competition, you’re challenged to build an algorithm to identify individual whales in images. You’ll analyze Happywhale’s database of over 25,000 images, gathered from research institutions and public contributors. By contributing, you’ll help to open rich fields of understanding for marine mammal population dynamics around the globe.

| Dataformat   |      Metric      |  Prediction |
|----------|:-------------:|------:|
| 3 channel image | Mean Average Precision @ 5 | single label classification (@ 5)|

Score: 
0.86838, 555/2131

The greatest challenge for this competition was the lack of images for each label of humpback whale (1-20 different images)
So I tried different kinds of one-shot learning algorithms like siamese networks with LAP matching of positive and negative examples.

In the end it turned out metric learning and siamese networks were indeed good approaches to the problem but time was running short.

### Jigsaw Unintended Bias in Toxicity Classification: 

In this competition, you're challenged to build a model that recognizes toxicity and minimizes this type of unintended bias with respect to mentions of identities. You'll be using a dataset labeled for identity mentions and optimizing a metric designed to measure unintended bias.

| Dataformat   |      Metric      |  Prediction |
|----------|:-------------:|------:|
| text |  generalized mean of bias AUCs | classification |

First NLP competition I have joined and I still feel like I have to learn a lot in this space.

Used [GloVe](https://nlp.stanford.edu/projects/glove/) combined with a lstm + word embedding neural network.

Score:
0.93568, 718/2646 placement

### Ship Detection Challenge:

In this competition, you are required to locate ships in images, and put an aligned bounding box segment around the ships you locate. Many images do not contain ships, and those that do may contain multiple ships. Ships within and across images may differ in size (sometimes significantly) and be located in open sea, at docks, marinas, etc.

| Dataformat   |      Metric      |  Prediction |
|----------|:-------------:|------:|
| 3 channel image | F2 Score | binary segmentation |

Score: 

public 0.70823, 208/884 placement

private 0.82704, 524/884 placement 

Used fastai with resnet34 for image segmentation.

Big dropoff on private test set because I tried to select a part of the train set to reduce computing time but my selection method was lacking. 
Definitely will keep this mistake in mind for the future

## Structured Data

### Store Item Demand Forecasting Challenge

You are given 5 years of store-item sales data, and asked to predict 3 months of sales for 50 different items at 10 different stores.

| Dataformat   |      Metric      |  Prediction |
|----------|:-------------:|------:|
| time series data | symmetric mean absolute percentage error | regression |

Score: 
because of a leak the competition was reset in the last few weeks and I did not have the time to submit again.


### Predicting Molecular Properties

#### Codebase

[seperate github repo](https://github.com/DollofCuty/Predicting-Molecular-Properties)

In this competition, you will develop an algorithm that can predict the magnetic interaction between two atoms in a molecule (i.e., the scalar coupling constant).

| Dataformat   |      Metric      |  Prediction |
|----------|:-------------:|------:|
| structured data (__graph based__) | log mean average error | regression |

A lot of additional data that is not usable directly because it's not contained in the test set.
Also domain knowledge about atom interaction in molecules seems really important.

Solved using lightgbm and a [message passing neural network](https://arxiv.org/pdf/1704.01212.pdf)

#### Score 
__top 2%__

| leaderboard   | score | placement |
|----------|:-------------:|---------:|
| public | -2.37190 | __43/2757__ |
| private | -2.36477 | __42/2757__ |
