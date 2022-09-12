---
layout: post
title: Why Human Activity Recognition Using Wearables Is Far From Being Solved  
date: 2022-08-02
comments: true
usemathjax: true
---
Authors: [Shing Chan](https://scholar.google.co.uk/citations?user=-FqhzRcAAAAJ&hl=en) & Hang Yuan

Keywords: Human activity recognition, IMU, wearables, machine learning


## HAR intro

Human activity recognition (HAR) is one of the most popular applications for wearable devices. HAR describes the class of methods that we use to identify human activities from time series signals using one or more data modalities such as images and accelerometers/Inertial measurement units(IMUs).

Many devices that we use nowadays have lots of HAR applications. HAR is used in smartwatches to track fitness levels and sleep quality, VR devices for human-computer interaction and health applications for clinical monitoring. Despite the irresistible marketing slogans from the device manufacturers about how good their products are. HAR algorithms used nowadays suffer from numerous limitations.

In fact, **we'd argue that HAR is far from being solved** because of the following reasons:

I. [Difficult to define what is an activity. Temporal definition](#i-hard-to-define-what-is-an-activity)

II. [Hetergenous benchmark baselines](#ii-hetergenous-benchmark-baselines)

III. Getting groud-truth data for HAR is both expensive and difficult

IV. Activity profile could differ based different characeterstics of different individuals such as weight/height/race/culture

V. Lacking standarslization in data storage, processing and analytics 

In this post, we will entail each aspect that makes HAR difficult and mitigations to these challenges. Hopefully, by reading this post, more people become wary of the limitations of current technology and points out prompsing directions if people want to engage with HAR research.

## I. Hard to define what is an activity 
For us humans, it is obvious when someone is running or doing dishes. However, it is much harder for the machines to know what constitues an activity. Take walking for an example. Despite its perceived simplicity, designing a step counter is non-trivial. 

Below you can find differnt types of gait patterns

Depending on a person's age and the environment, we could see a a perfect gait cycle on the one end, or something very sporadic on the other end. The `duration`, `trajectory`, and `geometric characteritiscs` of even the same activity class can be totally different. Even getting walking right is not easy, let alone other activities that require more complex coordination of body movement such as dancing. So it is really challenging to define what is an activity.

[The Compendium of Physical Activities](https://sites.google.com/site/compendiumofphysicalactivities/home?authuser=0) is one of the major initiaties that tries to standardsie the type of physical activities and their corresponding energy expediture. The compedium is one of the common reference in depimiological studies. Whereas, more recently, 4d2go also proposed to do something similar by having over 200+ activitu taxonmoy to label its ego-centric video stream for VR applications. Depending on the type of applications, one might opt to use a different activity vocabulary. The gist is there are several standard ways to representing these categories and given one's specific use, we might wish to adaopt or create our own definitions. 

Except for the common activities, there are many activities special in HCI but haven't been widely recgonised yet. For example, on the older apple watches whose battery doesn't last very long. One way Apple tries to perseve the battery is that only flash the screen when they detect a user has a motion to lift the watch up and facing the user himsefl/herself. This type of motion might be specific to the type of device and the purpose of the interaction, therefore, that adds more complexit to have a general HAR model that can detect every single activity that we have


Regular walk            |  Irregular walk    | Edler Strolling |
:-------------------------:|:-------------------------:|:----------:|
 <img src="/assets/gifs/walk1.gif" width="300"> | <img src="/assets/gifs/walk2.gif" width="300">| <img src="/assets/gifs/walk3.gif" width="300">|

Source: GIPHY

## II. Hetergenous benchmark baselines
In other popular domains in machine learning such as vision or language, there are many well-recognised benchmarks that everyone understands and uses such as CIFAR-10 and ImageNet. To be the `state-of-the-art` model, one will need to develop some new methods that out-performnce existings baselines on some dimensions. **Unforuatenly, in HAR, we don't have a well-recognised HAR benchmark that allows us to directly compare different model performances.**



<img src="/assets/images/baseline_har.png" alt="Source: Yuan et al. 2022 Self-supervised Learning for Human Activity Recognition Using 700,000 Person-days of Wearable Data " style="width: 90%; display:block; margin: 0 auto;"/>


The above shows a list of common benchmark datasets in HAR. Most datasets are small-scaled in nature with a very limited number of activity classes labelled. This is partly due to the difficulty in collecting labelled HAR dataset which will be cover in the next section. If you look closer to the performance that existings techniques have on those benchmarks, you will find that many methods already achieve almost a perfect score on those benchmarks. However, this doesn't mean that we have perfect HAR model at all. The reason being that 

* Simple macro-actiivty classes instead of more fine-grainded labels
* Lab envrionment datasets which is not representative of the free living envrionment 

Anothe reason why it is very difficult to compare different models is that all the benchmarks have totally differnet evaluation characterstisc which make it not possible to apply one model and the evaluation protocol on all the datasets. The two main differences lie in the evaluation protocol and the window length people use to divide up a continuious time series to label for each activity label. 


| Dataset     | #Samples | Evaluation method                                                      | Window length          | Evaluation metric |
|-------------|----------|------------------------------------------------------------------------|------------------------|-------------------|
| Capture24   | 573K     | Held-one-subject_out                                                   | 30 sec                 | F-measure/Kappa   |
| Rowlands    | 36K      | Tested proprietary algorithms with all subjects being in one test set. | 1 min                  | ROC               |
| WISDM       | 28K      | 10-fold CV                                                             | 5 sec/10 sec           | Accuracy          |
| REALWORLD   | 12K      | 10-Fold CV                                                             | 1 sec                  | F-measure         |
| Opportunity | 3.9K     | Fixed train/test split                                                 | 500 ms                 | F-measure/AUC     |
| PAMAP2      | 2.9K     | 9-fold CV                                                              | 5.12 sec               | F-score/Accuracy  |
| ADL         | .6k      | None specified                                                         | No fixed window length | Accuracy          |



The sampling frequency and data modality is also different. People end up spending lots of time in the data preprocessing stage. One question people often have is what's the ideal sampling frequency that one should use. From expereince, `30hz` is a good threshold between battery consumption without the loss of performance for IMUs. Since most of human activties have a frequency rate below `15hz`, it would be safe to run up to `30hz`. Some devices like to have a frequency of `100hz`+, we didn't really benefit from having that much more  data.

## III. Getting groud-truth data for HAR is both expensive and difficult
One of the key reasons why existing benchmark datasets is small is that it is rather challenging to annotate ground truth for IMUs. To annotate HAR datasets, we will require concurrent ACC and video data. The difficulties are:
* We sync the timestamps on both the wearable and video recording devices whoses timestamps might not be perfect syncrhnoty.
* It might be easy to obtain the video stream of human activity in a lab envrionment. However, data collected in a labl doesn't relfect the data distribution in a free-living envrionment. However, getting the concurrent video stream in a free-living environment is much harder because we would require the participants to wear an ego-centraic camera or install many camceras in the participants' living envrionments. Neither of which is ideal and both impose privacy concerns.


<img src="/assets/images/cameras.png" alt="TODO:  add references; https://cvit.iiit.ac.in/blog/ego4d-an-achievement-to-celebrate"  style="width: 60%; display:block; margin: 0 auto;"/>

### Impossible to annotate the video as a high frame rate
The below is a list of images that were taken when I worn one of these cameras before. A annoatator will need to select an activity from 200+ activity classes for every frame that is taken. Depending on how many pictures are taken per second, the sheer volumne of the task becomes extremely large. If 1 image is taken per second, then one will need to annotate 1 * 60 * 60 * 24 = 86400 images just for one day of data per person. We are no way near to have the capacity to have high-quality free-living data at the moment that relfects the real events without much loss of signals. The best are aware of at the moment is the capture-24 which only takens one image every 30 seconds. On the other extreme, we have ego4do, which have very high frame rates but much shorter duration. 

<img src="/assets/images/sample_view.png" alt="Egocentric view"  style="width: 80%; display:block; margin: 0 auto;"/>





## IV. Activity profile could differ based different characeterstics of different individuals such as weight/height/race/culture
For the same type of activity, we shall expect to a large differences in differnet groups of people. That aspect also contributes to te fact why an activity is hard to define. Even after defining what an activity is, we will have issues with the data. The vast diverse activity chaoractersitics often mean that the model trained is likey to overfit to the trianing set. When during prediction on a different population groups, we should expect to see a performance reduction because of the out-of-domain prediction. 

There are a myriad of aspects might me exacerbsite this issue:
* Age
* Weight
* Height
* Culture
* Wrist length
Are some of the key axes that make HAR harder than it always is.


The easist solution is really to have more data in your model development, which we sadly don't, especially for labelled data. The alternative is to use better methods that can make use of unlabelled data which we have plenty of. Relevant works includes using transfer learning to to personalise the prediction trained on a large pool of subjects to a specific subject for which we just have very limited data for (Add ref). Or we can use self-supervised-learning to learn useful embedding from unlabelled data such that the eventual model simply only needs much smaller labelled dataset. (to add ref)



## V. Lacking standarslization in data storage, processing and analytics
The last aspect which makes HAR so challenging is because in the wearabels domain, this is still an early stage. Every single vendo is like a tiger trying to get as much market share as possible utilising their propertiery solutions which means we have bespoke data storage format, processing tools, and eventually analtics. As compared to a more standard image modality such as images, we have format such as JPG and PNG which every signle hardware and sofware will support making it much easier for the industry to move forward. It will also make the study outcomes from different devices more comparable. 

Whilst it is understandable that no one has the motivation to make a standardlised format for everyone to use because each would like to have their own thing. But as the field we will be saving a lot by introducint these standards. In order to do this, however, would require buy in from all sides, the industries, reserach labs and the devices users to design the best all-purpose future-proof soltuions that everyone can benefit from. Althought there has been some effrot some the industry, the field is still not picking it up yet (cite the meta data format).

* Standardlisation
    * Add some open source data storage format that Meta shared 



# Future direction
In summary, the wearables field is a rapidly growing field as examplied by the recenet lauches of Apple Watch ultra which even incoporates temprature into daily consumer product. Maybe by incoporating more data modality is something that will greatly  enhance the applicability of wearables for HAR especially.  There are still a lot of work to be done around just just the hardware itself anymore but moving towards a more unifying field of data, analytics and inferences which could eventually translate into helping the device users have a better quality of life. We'd like to close this post with the figure below. It seems that there are still a lots of innotivation happening on the sensors and hardware side of things to make these sensors more accurate and more portable. We are at the stage where we shall gradually start to understand and standarlise the way that we do things so that the downstream applicaitons can be properly implemented (TODO: add a figure about the development of wearable tech compare vision with wearable tech)




## References 
[1] [Yuan, H., Chan, S., Creagh, A. P., Tong, C., Clifton, D. A., & Doherty, A. (2022). Self-supervised Learning for Human Activity Recognition Using 700,000 Person-days of Wearable Data. arXiv preprint arXiv:2206.02909.
Chicago](https://arxiv.org/abs/2206.02909)


