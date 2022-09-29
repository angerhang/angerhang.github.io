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

Human activity recognition (HAR) is one of the most popular applications for wearable devices. HAR describes the class of methods that we use to identify human activities from time series signals using one or more data modalities such as images and accelerometers/Inertial measurement units (IMUs).

Many devices that we use nowadays have lots of HAR applications. HAR is used in smartwatches to track fitness levels and sleep quality, VR devices for human-computer interaction, and health applications for clinical monitoring. Despite the irresistible marketing slogans from the device manufacturers about how good their products are. HAR algorithms used nowadays suffer from numerous limitations.

In fact, **we'd argue that HAR is far from being solved** because of the following reasons:

I. [Difficult to define what is an activity. Temporal definition](#i-hard-to-define-what-is-an-activity)

II. [Hetergenous benchmark baselines](#ii-hetergenous-benchmark-baselines)

III. [Getting groud-truth data for HAR is both expensive and difficult](#iii-getting-groud-truth-data-for-har-is-both-expensive-and-difficult)

IV. [Diverse characteristics contribute to different activity profiles](#iv-diverse-characteristics-contribute-to-different-activity-profiles)

V. [Lacking standardization in data storage, processing and analytics](#v-lacking-standarslization-in-data-storage-processing-and-analytics) 

## I. Hard to define what is an activity 
For us humans, it is obvious when someone is running or doing dishes. However, it is much harder for machines to know what constitutes an activity. Take walking for an example. Despite its perceived simplicity, designing a step counter is non-trivial. 

Below you can find three different gait patterns:


Regular walk            |  Irregular walk    | Edler Strolling |
:-------------------------:|:-------------------------:|:----------:|
 <img src="/assets/gifs/walk1.gif" width="300"> | <img src="/assets/gifs/walk2.gif" width="300">| <img src="/assets/gifs/walk3.gif" width="300">|

Source: GIPHY

Depending on a person's age and the environment, we could see a perfect gait cycle in young adults, however, in older individuals, we might see a much more gradual and perhaps irregular gait cycle. The `duration`, `trajectory`, and `geometric` characteristics of even the same activity type can be vastly different, not even to mention other activities that require more complex body coordination. 

Well, some might suggest that instead of having an activity classed called "walk", let's have many "walk" types to account for the differences in how people. What a brilliant idea!  [The Compendium of Physical Activities](https://sites.google.com/site/compendiumofphysicalactivities/home?authuser=0) (Ainsworth, et al., 2000)
 is one of the major initiatives that aim to have a universal activity classification. The compendium is widely used in epidemiological studies. Whereas, more recently, [Ego4d](https://ego4d-data.org) (Grauman, et al., 2022) also proposed to do something similar by having over 200+ activity taxonomy to label its ego-centric video stream for VR applications. Depending on the type of applications, we might opt to use a different activity vocabulary.

Nonetheless, it is important to note that none of the activity compendiums is perfect. Sometimes, we will need to develop a variant of our own. For example, if you are only interested in knowing roughly how active you are per week. Perhaps, you don't need to develop a model for hundreds of activity types. Three classes might be all you need: vigorous, moderate, and sedentary.  


Most activity type definition needs careful design. There is one exception to the rule which is when we would like to detect a specific gesture signal in HCI. For example,  Apple Watch's battery doesn't last very long. Hence, the Watch tries to preserve the battery by having the screen dimed until when a specific up lift motion is detected. The lift detector is designed to capture the user's desire to look at the watch. This type of motion is rather unique and specific to the type of device. A general HAR model won't be able to capture this type of motion, hence, adding more complexity to the HAR model development.


## II. Hetergenous benchmark baselines
**The benchmark datasets for HAR are so heterogeneous that as a field we don't know how to do an apple-to-apple comparison for different modeling techniques.** In popular machine learning conferences nowadays, there is a big emphasis on beating the `state-of-the-art` performance on existing benchmarks. In the field of computer vision, for example, one can evaluate their methods on [ImageNet](https://www.image-net.org) or [COCO](https://cocodataset.org/#home), if a paper proposes an algorithm that beats the current best model on these benchmark datasets, then that paper becomes the new `state-of-the-art`. How much a contribution a paper largely depends on the performance difference between the proposed algorithm and the existing best method. When a well-recognized benchmark exists, it is easy to do an apple-to-apple comparison between different methods. However, in HAR, we don't have a well-recognized benchmark yet, which makes it much harder to which method is better.


<img src="/assets/images/baseline_har.png" alt="Source: Yuan et al. 2022 Self-supervised Learning for Human Activity Recognition Using 700,000 Person-days of Wearable Data " style="width: 90%; display:block; margin: 0 auto;"/>

There have been many open-sourced HAR benchmarks for researchers to use (Table 1). However, none of the existing datasets became the gold standard to compare different algorithms. The reasons are multi-faceted: 

* **Most of the benchmarks are small**. For small datasets, cross-validation is better suited to provide a more robust estimation for the empirical risk as compared to the larger datasets, from which, a subset is held out as the test set. Having a common test set makes it easy to compare different methods at once. In most HAR research, however, people rely on their own way to partition the benchmark datasets, making it impossible to compare results from different papers directly.
* **The limited sizes of the benchmarks also mean that the number of activity classes labelled is also limited.** It is often to see almost perfect performance on some of the smaller datasets but that doesn't mean the method used is perfect for HAR but that it is possible to achieve a perfect score when separating several activities in a small number of subjects.
* We define an activity label over a fixed window length. However, **the current benchmarks have vastly different window length definitions in their evaluation, making it hard to even compare the model performance across different datasets.**
* Lastly, **data collected in a lab environment doesn't truly reflect the model performance in the real world.** Admittedly, it is much easier to set up some mounted cameras in a lab so that we can label the data by looking at the video stream. However, there is a big gap between the types of activities and how people behave in a lab and real-living environment. So to fully appreciate the performance of HAR, we need to test our model on more datasets collected under free-living conditions.


| Dataset     | #Samples | Evaluation method                                                      | Window length          | Evaluation metric |
|-------------|----------|------------------------------------------------------------------------|------------------------|-------------------|
| Capture24   | 573K     | Held-one-subject_out                                                   | 30 sec                 | F-measure/Kappa   |
| Rowlands    | 36K      | Tested proprietary algorithms with all subjects being in one test set. | 1 min                  | ROC               |
| WISDM       | 28K      | 10-fold CV                                                             | 5 sec/10 sec           | Accuracy          |
| REALWORLD   | 12K      | 10-Fold CV                                                             | 1 sec                  | F-measure         |
| Opportunity | 3.9K     | Fixed train/test split                                                 | 500 ms                 | F-measure/AUC     |
| PAMAP2      | 2.9K     | 9-fold CV                                                              | 5.12 sec               | F-score/Accuracy  |
| ADL         | .6k      | None specified                                                         | No fixed window length | Accuracy          |



As for the data-sampling variations shown above, from experience, it doesn't make a big difference for analysis. `30hz` is usually a good threshold between battery consumption without the loss of performance for IMUs. Since most human activities have a frequency rate below `15hz` (send us a reference if you have one!), it would be safe to run up to `30hz`. Some devices like to have a frequency of `100hz`+. We don't really benefit from having that much more data.


## III. Getting groud-truth data for HAR is both expensive and difficult
One of the key reasons why existing benchmark datasets is small is that it is rather challenging to annotate ground truth for IMUs. To annotate HAR datasets, we will require concurrent ACC and video data. The difficulties are:
* We sync the timestamps on both the wearable and video recording devices whoses timestamps might not be perfect syncrhnoty.
* It might be easy to obtain the video stream of human activity in a lab envrionment. However, data collected in a labl doesn't relfect the data distribution in a free-living envrionment. However, getting the concurrent video stream in a free-living environment is much harder because we would require the participants to wear an ego-centraic camera or install many camceras in the participants' living envrionments. Neither of which is ideal and both impose privacy concerns.


<img src="/assets/images/cameras.png" alt="TODO:  add references; https://cvit.iiit.ac.in/blog/ego4d-an-achievement-to-celebrate"  style="width: 60%; display:block; margin: 0 auto;"/>

### Impossible to annotate the video as a high frame rate
The below is a list of images that were taken when I worn one of these cameras before. A annoatator will need to select an activity from 200+ activity classes for every frame that is taken. Depending on how many pictures are taken per second, the sheer volumne of the task becomes extremely large. If 1 image is taken per second, then one will need to annotate 1 * 60 * 60 * 24 = 86400 images just for one day of data per person. We are no way near to have the capacity to have high-quality free-living data at the moment that relfects the real events without much loss of signals. The best are aware of at the moment is the capture-24 which only takens one image every 30 seconds. On the other extreme, we have ego4do, which have very high frame rates but much shorter duration. 

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
Chicago	

[3] [Yuan, H., Chan, S., Creagh, A. P., Tong, C., Clifton, D. A., & Doherty, A. (2022). Self-supervised Learning for Human Activity Recognition Using 700,000 Person-days of Wearable Data. arXiv preprint arXiv:2206.02909.
Chicago](https://arxiv.org/abs/2206.02909)


