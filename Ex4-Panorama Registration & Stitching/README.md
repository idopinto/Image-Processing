# Panorama Registration & Stitching
## Exercise 4| Image Processing @ HUJI

## Image Pair Registration
### Original Image Pair
![image](https://user-images.githubusercontent.com/71530633/236032140-7cf49266-e1aa-4951-8eda-fa8622a16a8d.png)

### Finding Key Points (Harris Corner Detector) and Sampling Descriptors (MOPs-like) in both images
![image](https://user-images.githubusercontent.com/71530633/236032256-5cd832c8-68c3-45f5-b497-d0f0e952fcc9.png)

### Matching Features with Descriptors
![image](https://user-images.githubusercontent.com/71530633/236032406-616d3f1d-b683-4675-af9b-9be43abb2bb3.png)

### Computing homography between images with RANSAC. (Blue - Outliers, Yellow - Inliers)
![image](https://user-images.githubusercontent.com/71530633/236032538-7459663e-4780-415a-ac50-2f10cb818938.png)

### Warping image with that homography 
![image](https://user-images.githubusercontent.com/71530633/236032582-c0f8ccf0-a7ce-4c73-a644-3c3cfbe60b66.png)
