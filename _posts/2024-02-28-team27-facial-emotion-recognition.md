---
layout: post
comments: true
title: Exploring different techniques for Facial Expression Recognition (FER)
author: Wei Jun Ong, Matthew Teo, Mingyang Li, Pierce Chong
date: 2024-02-28
---


> Facial expression recognition (FER) is a pivotal task in computer vision with applications spanning from human-computer interaction to affective computing. In this project, we conduct a comparative analysis of three prominent model architectures for FER: a convolutional neural network (CNN), POSTERV2 [1] a cross-fusion transformer-based network, and YOLOv5. These architectures represent diverse approaches in leveraging deep learning techniques for facial expression analysis. We evaluate the performance of each model architecture on RAF-DB, which encompass a wide range of facial expressions under various contexts. The evaluation metrics include accuracy and number of parameters, which give comprehensive insights into the models' capabilities in recognizing facial expressions accurately across different datasets. Our evaluations have shown that the POSTERV2 model outperforms the other models in terms of accuracy. We also present a demonstration of the YOLOv5 model running on a webcam and training a custom model to recognize "awake" and "sleep" expressions. Our findings provide valuable insights into the strengths and limitations of different model architectures for FER, which can guide the selection of appropriate models for specific applications. 


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction


Facial expression recognition (FER) is a pivotal area of computer vision and artificial intelligence focused on identifying human emotions from facial expressions captured in images or videos. It holds significant implications for various fields, including human-computer interaction, safety, and entertainment. Over the years, researchers have explored different approaches to tackle this task, with deep learning approaches becoming more popular in recent years. Classical approaches to facial expression recognition typically relied on handcrafted features and traditional machine learning algorithms. These methods involved extracting facial features such as edges, corners, or texture descriptors from images, followed by training classifiers to recognize specific emotional states based on these features. Techniques like Support Vector Machines (SVMs), Decision Trees, or k-Nearest Neighbors (k-NN) were commonly employed for classification tasks. While classical approaches laid the foundation for facial expression recognition, they often encountered limitations. One major challenge was the reliance on manually engineered features, which might not fully capture the complexity and variability of facial expressions across different individuals, lighting conditions, and facial poses. Additionally, these methods struggled with handling variations in facial expressions caused by factors like occlusions, facial hair, or cultural differences, leading to reduced accuracy and robustness.

Deep learning methods have emerged as a more successful paradigm in facial expression recognition, offering significant improvements over classical techniques. Deep neural networks excel at learning representations directly from large-scale datasets via powerful computational resources, thus removing the need for handcrafted features. A key advantage of this is the ability to automatically capture important features within facial images, enabling more robust and accurate recognition of facial expressions under various conditions.

In this project, we explore three prominent deep neural network architectures for FER, including a convolutional neural network (CNN), POSTERV2, and YOLOv5. We show that PosterV2 outperforms the other models in terms of accuracy due to its cross-fusion transformer-based architecture. Each model was evaluated on RAF-DB, a benchmark dataset, and we compare their accuracies. Finally, we compare between the three models and discuss their advantages and limitations in the context of FER, before ending with a demonstration of the YOLOv5 model running on a webcam and training a custom model to recognize "awake" and "sleep" expressions.


## Different Approaches to FER

### 1. TO BE DETERMINED


### 2. POSTER V2: A simpler and stronger facial expression recognition network

**Overview** An alternative approach to FER is the POSTER V2 model, an enhanced version of the original POSTER model. The original POSTER model has 4 main features, namely a landmark detector, an image backbone, cross-fusion transformer encoders and a pyramid network. Given an input image, the landmark detector extracts detailed facial landmark features while the image backbone extracts generic image features. Following that, the landmark and image features are concatenated and scaled to different sizes, before interacting via cross-attention in separate cross-fusion transformer encoders for each scale. Finally, the model extracts and integrates the outputs of each encoder into multi-scale features of images and landmarks. Despite achieving state-of-the-art performance in FER, the architecture of the original POSTER model is highly complicated, resulting in expensive computational costs. Hence, POSTER V2 implements 3 key improvements on top of the original POSTER model architecture to not only reduce computational costs, but also enhance model performance.

![]({{'/assets/images/team27/original_POSTER.png'|relative_url}}) 
*Fig 1. Original POSTER model architecture [1]*

**Remove Image-to-Landmark Branch** A primary characteristic of the original POSTER model is its two-stream design that consists of both an image-to-landmark branch, and a landmark-to-image branch. To reduce computational cost, the POSTER V2 research team conducted an ablation study to determine which branch plays a more decisive role in model performance, so as to remove the less decisive branch. As seen from the results below, the landmark-to-image branch proved to be the more decisive one. Let us find an intuitive explanation for this trend. In the landmark-to-image branch, image features are guided by landmark features, which are the query vectors in the cross-attention mechanism. Since landmark features highlight the most important regions of the face, it reduces discrepancies within the same class, which are emotions for FER tasks. Furthermore, it reduces focus on face-prevalent regions, thus reducing similarities among different classes. Therefore, by retaining the landmark-to-image branch, POSTER V2 ensures that key FER issues such as intra-class discrepancy and inter-class similarity are mitigated. At the same time, removing the image-to-landmark branch reduces 

### 3. YOLOv5

Lastly, another approach to FER is the use of the YOLOv5 (You Only Look Once) architecture, which is a popular object detection algorithm that builds upon the previous versions of the YOLO family of models. The architecture of YOLOv5 consists of a backbone network, neck network, and head network as shown in Fig 1.

![]({{'/assets/images/team27/yolo_architecture.png'|relative_url}}) 
*Fig 1. The default inference flowchart of YOLOv5 [1]*

**Backbone Network:** The backbone network is responsible for extracting features from the input image. In YOLOv5, the [CSPDarknet53](https://paperswithcode.com/method/cspdarknet53) architecture is used as the backbone, which is a deep CNN with residual connections. It consists of multiple convolutional layers followed by residual blocks, which help in capturing both low-level and high-level features from the image.

**Neck Network:** The neck network of YOLOv5 employs a [PANet](https://paperswithcode.com/method/panet)  (Path Aggregation Network) module. The PANet module fuses features from different scales to enhance the model's ability to detect objects of various sizes. It consists of bottom-up and top-down pathways that connect features of different resolutions, to generate a multi-scale feature map that aids in detecting objects of different sizes.

**Head Network:** The head network of YOLOv5 is responsible for generating the final predictions It consists of several convolutional layers followed by a global average pooling layer and fully connected layers. These detection layers predict the bounding box coordinates, class probabilities, and other attributes for the detected objects. YOLOv5 uses anchor boxes to assist in predicting accurate bounding boxes for objects of different sizes.

![]({{'/assets/images/team27/darknet53.png'|relative_url}}) 
*Fig 2. Darknet-53 architecture [2]*

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

Due to YOLOv5's efficiency and speed, many papers have adapted it for facial expression recognition by training it on a dataset that includes facial images labeled with expression categories. With its small size and fast inference time, YOLOv5 is suitable for real-time applications such as emotion recognition in video calls, driver monitoring systems, and emotion-aware advertising. It can also be used in applications on low-powered edge devices such as smartphones and IoT devices.

## Comparison of the Approaches

### Performance Evaluation

We compare the performance of the three model architectures on [RAF-DB](https://www.v7labs.com/open-datasets/raf-db), a popular dataset for benchmarking FER. 

**Undetermined model**

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed etiam laoreet, nunc nec lacinia tincidunt, nunc nunc

**YoloV5**

![]({{'/assets/images/team27/yolo_performance.png'|relative_url}}) 
*Fig 3. Different models experiment on RAF-DB Dataset [3]*

Evaluating YOLOv5 on the RAF-DB dataset gives us an accuracy of 73.6% and mAP@0.5 (%) of 81.8%, most notably, the inference time was only 15ms [3].

**Poster V2**

PosterV2, also known as Poster++, exhibits state-of-the-art performance on the FER task, outperforming the other models in terms of mean accuracy. Out of the three models, Poster++ achieved the highest accuracy on the RAF-DB dataset with an accuracy of 92.21% across all classes.

![PosterV2]({{'/assets/images/team27/posterv2_params.png'|relative_url}})
*Fig 4. Performance, parameters and FLOPs of Poster V2 [5]*

Despite acheiving SOTA results on FER, Poster++ maintains a number of parameters (43.7M) comparable to YoloV5 (46.2M). Thus, Poster++ is a much more memory efficient model for the FER task.

### Advantages and Limitations of each approach

**Undetermined model**

- Advantages:
    -

- Limitations:
    -

**YoloV5**

YoloV5 is a lightweight and efficient model for FER, with a fast inference time of 15ms. It is also capable of detecting multiple objects in an image, making it suitable for real-time applications. However, YoloV5 may not achieve the same level of accuracy as other models such as PosterV2, and might not be as robust in detecting facial expressions under various contexts like occulusion or low-light conditions.

**Poster V2**

PosterV2 is a state-of-the-art model for FER, achieving the highest accuracy on the RAF-DB dataset. It is also a lightweight model with a relatively small number of parameters similar to YoloV5, making it efficient for deployment on edge devices. However, PosterV2 may have a longer inference time compared to YoloV5 due to the higher computational burden of the transformer architecture, and may not be as suitable for real-time applications.

## Conclusion

In this project, we conducted a comparative analysis of three prominent model architectures for FER, including a convolutional neural network (CNN), POSTERV2, and YOLOv5. Our evaluations have shown that the POSTERV2 model outperforms the other models in terms of accuracy, but might be the most computationally intensive. 

Below, we present a demonstration of the YOLOv5 model running on a webcam and training a custom model to recognize "awake" and "sleep" expressions. This demonstration shows a potential application of YOLOv5 in real-time FER, and the process of training custom models for specific facial expressions.

## Bonus:

### 1. Recognizing our own expressions

On top of studying the approaches to FER on paper, we also wanted to run an existing codebase to try out one of the models on our own. We found a YOLOv5 pre-trained model and ran it on our own webcam. This model was trained on the [AffectNet](http://mohammadmahoor.com/affectnet/) dataset, which has 420,299 facial expressions. It also detects 8 basic facial expressions: anger, contempt, disgust, fear, happy, neutral, sad, surprise.

![]({{'/assets/images/team27/yolo_infer.gif'|relative_url}}) 
*Fig 4. YOLOv5-FER inference on our Webcam*

### 2. Training our own "awake" and "sleep" class

To supplement our project, we wanted to explore and train a model with two new custom classes for facial expression recognition. We collated our own dataset of 40 images (20 awake, 20 sleep) and annotated them using RoboFlow. Subsequently, we used the yolov5 architecture to train our own custom model.

![]({{'/assets/images/team27/roboflow_images.png'|relative_url}}) 
*Fig 5. Image Annotations on Roboflow*

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
*Fig 6. Test Images with Annotations*


## Reference

[1] Liu H, Sun F, Gu J, Deng L. SF-YOLOv5: A Lightweight Small Object Detection Algorithm Based on Improved Feature Fusion Mode. Sensors. 2022; 22(15):5817. https://doi.org/10.3390/s22155817

[2] Lu, Z., Lu, J., Ge, Q., Zhan, T.: Multi-object detection method based on Yolo and ResNet Hybrid Networks. In: 2019 IEEE 4th International Conference on Advanced Robotics and Mechatronics (ICARM). (2019) https://doi.org/10.1109/icarm.2019.8833671

[3] Zhong, H., Han, T., Xia, W. et al. Research on real-time teachers’ facial expression recognition based on YOLOv5 and attention mechanisms. EURASIP J. Adv. Signal Process. 2023, 55 (2023). https://doi.org/10.1186/s13634-023-01019-w

[4] Ultralytics. YOLOV5. PyTorch Hub. https://pytorch.org/hub/ultralytics_yolov5/

[5] Mao, Jiawei and Xu, Rui and Yin, Xuesong and Chang, Yuanqi and Nie, Binling and Huang, Aibin. POSTER V2: A simpler and stronger facial expression recognition network. arXiv preprint arXiv:2301.12149. (2023) https://arxiv.org/pdf/2301.12149

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
