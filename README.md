# Machine Learning

Trying different things, mostly Kaggle competitions

I wrote everything in Python using Jupyter Notebooks in an Anaconda3 environment

Examples contain Code for both structured and unstructured data and are mostly for showcasing the code because the data I used was stored locally

## Most interesting competitions so far were 

### Pneumia Detection: 

In this competition, you’re challenged to build an algorithm to detect a visual signal for pneumonia in medical images. Specifically, your algorithm needs to automatically locate lung opacities on chest radiographs.

Dataformat: DICOM image

Metric: IoU

Prediciton: multi bounding box

### Ship Detection Challenge:

In this competition, you are required to locate ships in images, and put an aligned bounding box segment around the ships you locate. Many images do not contain ships, and those that do may contain multiple ships. Ships within and across images may differ in size (sometimes significantly) and be located in open sea, at docks, marinas, etc.

Dataformat: 3 channel image

Metric: F2 Score

Prediciton: binary segmentation

Score: 
public 0.70823, 208/884 placement
private 0.82704, 524/884 placement

Using fastai with resnet34 for image segmentation
I tried to select a part of the train set to reduce computing time but my selection method was lacking. 
Definitly will keep this mistake in mind for the future

### Protein Classification Challenge

In this competition, you will develop models capable of classifying mixed patterns of proteins in microscope images. The Human Protein Atlas will use these models to build a tool integrated with their smart-microscopy system to identify a protein's location(s) from a high-throughput image.

Dataformat: 4 channel image

Metric: macro F1 Score

Prediciton: multi label classification

Score: 
public 0.493, 60/1182

Using resnet34 with fastai for multilabel-classification
Trying resnet50 next
