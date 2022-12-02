# OT2 - Machine Learning and Data Analytics, INSA Lyon

![descarga](https://user-images.githubusercontent.com/53874772/205325067-f993b168-918e-408e-9bb4-8d956d65b250.png)

## Deep Learning for face recognition project

This repository contains the face recognition project made by Mohamed Abderrhamne Taleb Mohamed, Stefania Curila, Alfredo Mahns, Omar Ormachea Hermoza and Anna Wachtel.

### Files composition and details

:warning: The main file is ``` sliding_window.py ```. Run this file to see the execution of the complete project.

* The ``` ./0_final_results ``` folder contains the final results of the pictures tested.

* The ``` ./cropped ``` folder contains the detected faces and the final image with the bounding boxes before and after applying NMS.

* The ``` ./fam_pictures ``` folder contains the images used to test the model.

* The ``` ./models ``` folder contains the trained models of each try.

* The ``` ./torchsample ``` and ``` ./imbalanced-dataset-sampler-master ```folder contains the used libraries.

* The ``` ./texture_imgs ``` folder contains the texture images used for the bootstrapping. They can be downloaded from this [link](https://1drv.ms/u/s!AhXBXVYMvnktm_wB8OKGA-bC50NH3g?e=knhLK2)

* The ```./train_images``` folder containes the training images

* The ```./test_images``` folder contains the test images

| File | Description |
| --- | --- |
| ``` bootstrapping.py ``` | Bootstrapping algorithm is defined |
| ``` load_data.py ``` | Database is read and saved, and the training is done |
| ``` net.py ``` | The architecture of the model is defined |
| ``` nms.py ``` | The non-maximum suppression algorithm is defined |
| ``` sliding_window.py ``` | Sliding window algorithm. The rescaling factor can be changed. The path of the images and other global varibales can be changed. |
| ``` test.py ``` | Testing the model and its performance |
