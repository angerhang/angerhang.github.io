---
layout: post
title: Why Human Activity Recognition Using Wearables Is Far From Being Solved  
date: 2022-08-02
comments: true
usemathjax: true
---
Authors: [Rosemary Walmsley](https://www.bdi.ox.ac.uk/Team/rosemary-walmsley-1) &  [Shing Chan](https://scholar.google.co.uk/citations?user=-FqhzRcAAAAJ&hl=en) & Hang Yuan

Keywords: Human activity recognition, IMU, wearables, machine learning


## HAR intro

Human activity recognition (HAR) is a popular application for wearable devices. HAR describes the methods we use to identify human activities from time series signals using one or more data modalities such as images and accelerometers/Inertial measurement units (IMUs). 

Many devices in our everyday lives use HAR: fitness levels and sleep quality tracking in smartwatches, human-computer interaction support in VR devices, and patient monitoring in clinical applications. Whatever the marketing slogans from device manufacturers about the strengths of their products, current HAR algorithms have important limitations.

In fact, **we'd argue that HAR is far from being solved** because of the following reasons:

I. [Difficult to define what is an activity](#i-hard-to-define-what-is-an-activity)

II. [Hetergenous benchmark baselines](#ii-hetergenous-benchmark-baselines)

III. [Getting groud-truth data for HAR is both expensive and difficult](#iii-getting-groud-truth-data-for-har-is-both-expensive-and-difficult)

IV. [Diverse characteristics contribute to different activity profiles](#iv-diverse-characteristics-contribute-to-different-activity-profiles)

V. [Lacking standardization in data storage, processing and analytics](#v-lacking-standarslization-in-data-storage-processing-and-analytics) 


## I. Hard to define what is an activity 
For us humans, it is obvious when someone is running or doing dishes. However, it is much harder for machines to know what constitutes an activity. Take walking, an apparently simple behaviour, as an example. Despite its perceived simplicity, designing a step counter is non-trivial. 

Below you can find three different gait patterns:


Regular walk            |  Irregular walk    | Edler Strolling |
:-------------------------:|:-------------------------:|:----------:|
 <img src="/assets/gifs/walk1.gif" width="300"> | <img src="/assets/gifs/walk2.gif" width="300">| <img src="/assets/gifs/walk3.gif" width="300">|

Source: GIPHY

Depending on someone's age and context, we could see a regular gait cycle in young adults, however, in older individuals, we might see a much more gradual and perhaps irregular gait cycle. The `duration`, `trajectory`, and `geometric` characteristics of even the same activity type can be vastly different.

Well, perhaps rather than having an activity classed called "walk", we could have many "walk" types to account for the differences in how people walk in different contexts. What a brilliant idea!  [The Compendium of Physical Activities](https://sites.google.com/site/compendiumofphysicalactivities/home?authuser=0) (Ainsworth, et al., 2000) is one of the major initiatives that aim to have a universal activity classification. The compendium is widely used in epidemiological studies. More recently, [ego4d](https://ego4d-data.org) (Grauman, et al., 2022) also proposed something similar by having over 200+ activity labels to apply to its ego-centric video stream for VR. Depending on the application, we might choose a different activity dictionary.


Nonetheless, it is important to note that none of the activity compendiums is perfect. In an ideal world, having a single model that can classify every kind of activity type in our activity dictionary.  More often, it suffices to develop a classifier using a much simpler definition. For example, if we only want a rough idea of how active in general, we might be happy with a mere differentiation between sleep, sedentary behaviour, light physical activity, and moderate-to-vigorous physical activity like [Walmsley, et al., 2022](https://bjsm.bmj.com/content/56/18/1008.abstract).


Another reason why HAR can be challenging is that we need an evolving activity definition to account for everything that we want to capture.  As new hardware surfaces, we need to have novel gesture recognition to improve the human-computer interaction (HCI) process. For example, Apple Watch's battery doesn't last very long, so the Watch tries to preserve the battery by keeping the screen dim until a specific up lift motion is detected. The lift detector is designed to capture when the user looks at the watch. This motion is rather unique and specific to the type of device. A general HAR model won't be able to capture this type of motion, hence, adding more complexity to the HAR model development.


## II. Hetergenous benchmark baselines
**The benchmark datasets for HAR are so heterogeneous that as a field we don't know how to do an apple-to-apple comparison for different modelling techniques.** In popular machine learning conferences nowadays, there is a big emphasis on beating the `state-of-the-art` performance on existing benchmarks. In the field of computer vision, for example, one can evaluate their methods on [ImageNet](https://www.image-net.org) or [COCO](https://cocodataset.org/#home), if a paper proposes an algorithm that beats the current best model on these benchmark datasets, then that paper becomes the new `state-of-the-art`. How much a contribution a paper largely depends on the performance difference between the proposed algorithm and the existing best method. When a well-recognized benchmark exists, it is easy to do an apple-to-apple comparison between different methods. However, in HAR, we don't have a well-recognized benchmark yet, which makes it much harder to which method is better.


<img src="/assets/images/baseline_har.png" alt="Source: Yuan et al. 2022 Self-supervised Learning for Human Activity Recognition Using 700,000 Person-days of Wearable Data " style="width: 90%; display:block; margin: 0 auto;"/>

There have been many open-source HAR benchmarks for researchers to use (Table 1). However, no existing dataset has become the gold standard to compare algorithms. The reasons are multi-faceted: 

* **Most of the benchmarks are small**. For small datasets, cross-validation is better suited to provide a more robust estimation for the empirical risk as compared to the larger datasets, from which, a subset is held out as the test set. Having a common test set makes it easy to compare different methods at once. In most HAR research, however, people rely on their own way to partition the benchmark datasets, making it impossible to compare results from different papers directly.
* **The limited sizes of the benchmarks also mean that the number of activity classes labelled is also limited.** It is often to see almost perfect performance on some of the smaller datasets but that doesn't mean the method used is perfect for HAR but that it is possible to achieve a perfect score when separating several activities in a small number of subjects.
* We define an activity label over a fixed window length. However, **the current benchmarks have vastly different window length definitions in their evaluation, making it hard to even compare the model performance across different datasets.**
* Lastly, **data collected in a lab environment doesn't truly reflect the model performance in the real world.** Admittedly, it is much easier to set up some mounted cameras in a lab so that we can label the data by looking at the video stream. However, there is a big gap between the types of activities and how people behave in a lab and free-living environment. So to fully appreciate the performance of HAR, we need to test our model on more datasets collected under free-living conditions.


| Dataset     | #Samples | Evaluation method                                                      | Window length          | Evaluation metric |
|-------------|----------|------------------------------------------------------------------------|------------------------|-------------------|
| Capture24   | 573K     | Held-one-subject_out                                                   | 30 sec                 | F-measure/Kappa   |
| Rowlands    | 36K      | Tested proprietary algorithms with all subjects being in one test set. | 1 min                  | ROC               |
| WISDM       | 28K      | 10-fold CV                                                             | 5 sec/10 sec           | Accuracy          |
| REALWORLD   | 12K      | 10-Fold CV                                                             | 1 sec                  | F-measure         |
| Opportunity | 3.9K     | Fixed train/test split                                                 | 500 ms                 | F-measure/AUC     |
| PAMAP2      | 2.9K     | 9-fold CV                                                              | 5.12 sec               | F-score/Accuracy  |
| ADL         | .6k      | None specified                                                         | No fixed window length | Accuracy          |



As for the data-sampling variations shown above, from experience, it doesn't make a big difference for analysis with deep learning models. `30 Hz` is usually a good threshold between battery consumption without the loss of performance for IMUs. Since most human activities have a frequency rate below `15 Hz` (send us a reference if you have one!), it would be safe to run up to `30 Hz`. Some devices like to have a frequency of `100 Hz`+. We don't really benefit from having that much more data in theory. However, if you are using statistical learning methods, some differences in the derived measurements exist if you compare data collected using `25 Hz` and `100 Hz` as suggested by [Small, et al., 2021](https://journals.humankinetics.com/view/journals/jmpb/4/4/article-p298.xml). In this case, choose your sampling methods carefully.


## III. Getting groud-truth data for HAR is both expensive and difficult
One of the key reasons why existing benchmark datasets is small is that it is challenging to annotate ground truth for IMUs. To annotate HAR datasets, we will require concurrent ACC and video data. The difficulties are:
* We sync the timestamps on both the wearable and video recording devices, for which the timestamps might not be in perfect synchrony.
* It might be easy to obtain a video stream of human activity in a lab environment. However, data collected in a lab doesn't reflect the data distribution in a free-living environment. However, getting the concurrent video stream in a free-living environment is much harder because we would require the participants to wear an ego-centric camera or install many cameras in the participants' living environments. Neither of which is ideal and both cause privacy concerns.


<img src="/assets/images/cameras.png" alt="TODO:  add references; https://cvit.iiit.ac.in/blog/ego4d-an-achievement-to-celebrate"  style="width: 60%; display:block; margin: 0 auto;"/>

### Impossible to annotate the video at a high frame rate
Below is a list of images that were taken when I wore one of these cameras before. An annotator will need to select an activity from 200+ activity classes for every frame that is taken. Depending on how many pictures are taken per second, the sheer volume of the task becomes extremely large. If 1 image is taken per second, then one will need to annotate 1 * 60 * 60 * 24 = 86400 images just for one day of data per person. We are no way near to having the capacity to have high-quality free-living data at the moment that reflects the real events without much loss of signals. The best are aware of at the moment is the capture-24 which only takes one image every 30 seconds. On the other extreme, we have ego4d, which has a very high frame rate but a much shorter duration. 

Capture-24 and ego4d decided to use very different approaches to annotate human activity from a video stream. Exactly how much signal is lost when doing the annotation with a lower sampling rate high depends on the sort of behaviour that we try to capture. We could also argue that activity variations over too short a time, for example, <1s or <5s, just shouldn't represent a separate behaviour because humans don't really shift behaviour that quickly. How quickly behaviour changes is also population specific.  Activity transitions are likely to happen more quickly for kids than adults.




<img src="/assets/images/sample_view.png" alt="Egocentric view"  style="width: 80%; display:block; margin: 0 auto;"/>





## IV. Diverse characteristics contribute to different activity profiles 
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
[1] Ainsworth, B. E., Haskell, W. L., Whitt, M. C., Irwin, M. L., Swartz, A. M., Strath, S. J., ... & Leon, A. S. (2000). [Compendium of physical activities: an update of activity codes and MET intensities. Medicine and science in sports and exercise](https://www.researchgate.net/profile/Ann-Swartz-2/publication/12330586_Compendium_of_Physical_Activities_an_Update_of_Activity_Codes_and_MET_Intensities/links/0912f51407bee1e3a6000000/Compendium-of-Physical-Activities-an-Update-of-Activity-Codes-and-MET-Intensities.pdf), 32(9; SUPP/1), S498-S504.

[2] Grauman, K., Westbury, A., Byrne, E., Chavis, Z., Furnari, A., Girdhar, R., ... & Malik, J. (2022). [Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition](https://openaccess.thecvf.com/content/CVPR2022/html/Grauman_Ego4D_Around_the_World_in_3000_Hours_of_Egocentric_Video_CVPR_2022_paper.html) (pp. 18995-19012).
	

[3] [Yuan, H., Chan, S., Creagh, A. P., Tong, C., Clifton, D. A., & Doherty, A. (2022). Self-supervised Learning for Human Activity Recognition Using 700,000 Person-days of Wearable Data](https://arxiv.org/abs/2206.02909). arXiv preprint arXiv:2206.02909.

[4]Walmsley, R., Chan, S., Smith-Byrne, K., Ramakrishnan, R., Woodward, M., Rahimi, K., ... & Doherty, A. (2022). [Reallocation of time between device-measured movement behaviours and risk of incident cardiovascular disease](https://bjsm.bmj.com/content/56/18/1008.abstract). British journal of sports medicine, 56(18), 1008-1017.
	

[5] Small, S., Khalid, S., Dhiman, P., Chan, S., Jackson, D., Doherty, A., & Price, A. (2021). [Impact of reduced sampling rate on accelerometer-based physical activity monitoring and machine learning activity classification](https://journals.humankinetics.com/view/journals/jmpb/4/4/article-p298.xml). Journal for the Measurement of Physical Behaviour, 4(4), 298-310.
Chicago	