# Machine Learning

Trying different things, mostly Kaggle competitions

I wrote everything in Python using Jupyter Notebooks in an Anaconda3 environment

Examples contain Code for both structured and unstructured data and are mostly for showcasing the code because the data I used was stored locally

## Most interesting competitions so far were 

## Unstructured Data

### Pneumia Detection: 

In this competition, you’re challenged to build an algorithm to detect a visual signal for pneumonia in medical images. Specifically, your algorithm needs to automatically locate lung opacities on chest radiographs.

| Dataformat   |      Metric      |  Prediciton |
|----------|:-------------:|------:|
| DICOM image |  IoU | multi bounding box |

### Ship Detection Challenge:

In this competition, you are required to locate ships in images, and put an aligned bounding box segment around the ships you locate. Many images do not contain ships, and those that do may contain multiple ships. Ships within and across images may differ in size (sometimes significantly) and be located in open sea, at docks, marinas, etc.

| Dataformat   |      Metric      |  Prediciton |
|----------|:-------------:|------:|
| 3 channel image | F2 Score | binary segmentation |

Score: 
public 0.70823, 208/884 placement
private 0.82704, 524/884 placement

Using fastai with resnet34 for image segmentation
I tried to select a part of the train set to reduce computing time but my selection method was lacking. 
Definitly will keep this mistake in mind for the future

### Protein Classification Challenge

In this competition, you will develop models capable of classifying mixed patterns of proteins in microscope images. The Human Protein Atlas will use these models to build a tool integrated with their smart-microscopy system to identify a protein's location(s) from a high-throughput image.

| Dataformat   |      Metric      |  Prediciton |
|----------|:-------------:|------:|
| 4 channel image | macro F1 Score | multi label classification |

Score: 
public 0.591, 8/1732

Focalloss worked way better than binary cross entropy

Started with resnet34 using fastai for multilabel-classification
resnet50 worked even better (by about 0.05 macro F1-score)
currently testing Se-ResNext50

## Structured Data

### Store Item Demand Forecasting Challenge

You are given 5 years of store-item sales data, and asked to predict 3 months of sales for 50 different items at 10 different stores.

| Dataformat   |      Metric      |  Prediciton |
|----------|:-------------:|------:|
| time series data | symmetric mean absolute percentage error | regression |

Score: 
public 12.61028, 44/44

### Google Analytics Customer Revenue Prediction

In this competition, you’re challenged to analyze a Google Merchandise Store (also known as GStore, where Google swag is sold) customer dataset to predict revenue per customer. 

| Dataformat   |      Metric      |  Prediciton |
|----------|:-------------:|------:|
| time series data | root mean squared error | regression |

Score: 
because of a leak the score was invalidated and the competition restarted with the whole data
