================================================
Image Recognition - Python-OpenCV Implementation
================================================

* Image Recognition using SIFT and BRISK feature detectors

Installation
------------
* We are using the latest `opencv` library version `3.0.0rc1`
* Run `pip install -r requirements.txt`
* Test running it by `python beyondbagsoffeatures.py`

Dependencies
------------
* Scipy
* Numpy
* scikit-learn
* matplotlib

Test Data Set
-------------
* The image data set need to run the code is provided in the `images` folder. (User may change it as per requirement.)
* Currently there are 3 datasets namely
 -  `training` and `testing` form the 1st pair
 -  `caltech_train` and `caltect_test` form the 2nd pair
 -  `c1_train` and `c1_test` form the last pair
* User can change the dataset one wants to run by commenting the `line#30-37` of `beyondbagsoffeatures.py`.
* User can switch between SIFT and BRISK by setting `sift=False` on `line#28`

Results
-------
![alt tag](https://github.com/bvnayak/image_recognition/blob/master/results/result1.png)
![alt tag](https://github.com/bvnayak/image_recognition/blob/master/results/result2.png)

References
----------

* S. Lazebnik, C. Schmid, and J. Ponce, “Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories”, IEEE CVPR, 2006
* https://github.com/wihoho/Image-Recognition
