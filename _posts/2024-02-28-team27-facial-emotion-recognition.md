---
layout: post
comments: true
title: Facial Expression Recognition (FER)
author: Wei Jun Ong, Matthew Teo, Mingyang Li, Pierce Chong
date: 2024-02-28
---


> Facial Expression Recognition is the automated process of detecting and analyzing facial expressions in images or videos using computer vision techniques. This project aims to evaluate and compare different approaches to FER. We will also be running existing models and training a new class for expression recognition.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

## Different Approaches to FER
### 1. Poster V2



### 2. TO BE DETERMINED


### 3. YOLOv5

Lastly, another approach to FER is the use of the YOLOv5 (You Only Look Once) architecture, which is a popular object detection algorithm that builds upon the previous versions of the YOLO family of models. The architecture of YOLOv5 consists of a backbone network, neck network, and head network as shown in Fig 1.

![]({{'/assets/images/team27/yolo_architecture.png'|relative_url}}) 
*Fig 1. The default inference flowchart of YOLOv5 [1]*

**Backbone Network:** The backbone network is responsible for extracting features from the input image. In YOLOv5, the [CSPDarknet53](https://paperswithcode.com/method/cspdarknet53) architecture is used as the backbone, which is a deep CNN with residual connections. It consists of multiple convolutional layers followed by residual blocks, which help in capturing both low-level and high-level features from the image.

**Neck Network:** The neck network of YOLOv5 employs a [PANet](https://paperswithcode.com/method/panet)  (Path Aggregation Network) module. The PANet module fuses features from different scales to enhance the model's ability to detect objects of various sizes. It consists of bottom-up and top-down pathways that connect features of different resolutions, to generate a multi-scale feature map that aids in detecting objects of different sizes.

**Head Network:** The head network of YOLOv5 is responsible for generating the final predictions It consists of several convolutional layers followed by a global average pooling layer and fully connected layers. These detection layers predict the bounding box coordinates, class probabilities, and other attributes for the detected objects. YOLOv5 uses anchor boxes to assist in predicting accurate bounding boxes for objects of different sizes.

![]({{'/assets/images/team27/darknet53.png'|relative_url}}) 
*Fig 2. Darknet-53 architechture [2]*

**Data Augmentation**

YOLOv5 employs various data augmentation techniques to improve the model's ability to generalize and reduce overfitting. These techniques include: Mosaic Augmentation, Copy-Paste Augmentation, Random Affine Transformations, MixUp Augmentation, Albumentations and Random Horizontal Flip.

**Computing Loss**

The YOLOv5 loss is determined by aggregating three distinct components:
- Classification Loss (BCE Loss): This evaluates the classification task's error using Binary Cross-Entropy loss.
- Objectness Loss (BCE Loss): Utilizing Binary Cross-Entropy loss again, this component assesses the accuracy in determining object presence within a given grid cell.
- Localization Loss (CIoU Loss): Complete IoU loss is employed here to gauge the precision in localizing the object within its respective grid cell.

$$
\text{loss} = \lambda_1 \cdot L_{\text{cls}} + \lambda_2 \cdot L_{\text{obj}} + \lambda_1 \cdot L_{\text{loc}}
$$

**FER Application**

Due to YOLOv5's efficiency and speed, many papers have adapted it for facial expression recognition by training it on a dataset that includes facial images labeled with expression categories. Evaluating YOLOv5 on the [RAF-DB](https://www.v7labs.com/open-datasets/raf-db) dataset gives us an accuracy of 73.6% and mAP@0.5 (%) of 81.8%, most notably, the inference time was only 15ms.[3] 

![]({{'/assets/images/team27/yolo_performance.png'|relative_url}}) 
*Fig 3. Different models experiment on RAF-DB Dataset [3]*

## Comparison of the Approaches

## Bonus:

### 1. Recognizing our own expressions

### 2. Training our own "awake" and "sleep" class

To supplement our project, we wanted to explore and train a model with two new custom classes for facial expression recognition. We collated our own dataset of 40 images (20 awake, 20 sleep) and annotated them using RoboFlow. Subsequently, we used the yolov5 architecture to train our own custom model.

![]({{'/assets/images/team27/roboflow_images.png'|relative_url}}) 
*Fig 4. Image Annotations on Roboflow*

We used transfer learning from yolov5s.pt and trained our model for 150 epochs using a single Google Colab T4 GPU. 

```
!python train.py --img 416 --batch 16 --epochs 150 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache
```

Model summary: 157 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs

|   Class   | Images | Instances |   P   |   R   | mAP50 | mAP50-95 |
|:---------:|:------:|:---------:|:-----:|:-----:|:-----:|:--------:|
|    all    |    8   |     8     | 0.76  | 0.597 | 0.781 |  0.696   |
|   awake   |    8   |     2     | 0.714 |  0.5  | 0.638 |   0.56   |
|   sleep   |    8   |     6     | 0.806 | 0.695 | 0.924 |  0.831   |

**Run Inference on Trained Weights:**

![]({{'/assets/images/team27/test_images.png'|relative_url}}) 
*Fig 5. Test Images with Annotations*


## Reference

[1] Liu H, Sun F, Gu J, Deng L. SF-YOLOv5: A Lightweight Small Object Detection Algorithm Based on Improved Feature Fusion Mode. Sensors. 2022; 22(15):5817. https://doi.org/10.3390/s22155817

[2] Lu, Z., Lu, J., Ge, Q., Zhan, T.: Multi-object detection method based on Yolo and ResNet Hybrid Networks. In: 2019 IEEE 4th International Conference on Advanced Robotics and Mechatronics (ICARM). (2019) https://doi.org/10.1109/icarm.2019.8833671

[3] Zhong, H., Han, T., Xia, W. et al. Research on real-time teachers’ facial expression recognition based on YOLOv5 and attention mechanisms. EURASIP J. Adv. Signal Process. 2023, 55 (2023). https://doi.org/10.1186/s13634-023-01019-w

--------------------------- DELETE LATER ----------

Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work.

Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |

```
# This is a sample code block
import torch
print (torch.__version__)
```

You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

---
