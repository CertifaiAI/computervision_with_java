# Eclipse Deeplearning4j Computer Vision Applications  

Each program structured to tackle a commonly seek for computer vision problem with Bytedeco JavaCV and Eclipse DeepLearning4j  

Below shows a high level overview of the applications in each modules

#### cv  
- [Color2GrayScale](https://github.com/CertifaiAI/computervision_with_java/blob/master/cv/src/main/java/ai/certifai/Color2GrayScale.java): Convert a single chosen RGB image into grayscale image

#### advanced-cv  
- [FrontCamFaceDetection](https://github.com/CertifaiAI/computervision_with_java/blob/master/advanced-cv/src/main/java/ai/certifai/FrontCamFaceDetection.java): Front face detection with Haar Cascade algorithm
<p align="center">
  <img align="middle" src="metadata/haarcascade.gif" width="400" height="255"/>
</p>

- [YOLOObjDetection](https://github.com/CertifaiAI/computervision_with_java/blob/master/advanced-cv/src/main/java/ai/certifai/YOLOObjDetection.java): Object detection with pretrained TinyYOLO models from Eclipse Deeplearning. 
<p align="center">
  <img align="middle" src="metadata/tinyyolo.gif" width="400" height="255"/>
</p> 
As the model was trained with on Pascal VOC dataset, it can detect these 20 labels as listed below.  

    1. Person  
    2. Car  
    3. Bicycle  
    4. Bus  
    5. Motorbike  
    6. Train  
    7. Aeroplane  
    8. Chair  
    9. Bottle  
    10. Dining Table  
    11. Potted Plant  
    12. TV/Monitor  
    13. Sofa  
    14. Bird  
    15. Cat  
    16. Cow  
    17. Dog  
    18. Horse  
    19. Sheep  
    20. Boat   


The programs had been fully tested running on the backend of CPU.  

## Getting Started ##

### Install Java ###

Download Java JDK
[here](https://adoptopenjdk.net/).  
(Note: Use Java 8 for full support of DL4J operations)

Check the version of Java using: 
```sh
java -version
```

Make sure that 64-Bit version of Java is installed.

### Install IntelliJ IDEA Community Edition ###
Download and install 
[IntelliJ IDEA](https://www.jetbrains.com/idea/download/).

### Install Apache Maven  *Optional* ###
IntelliJ provides a default Maven that is bundled with the installer.  
Follow these [instructions](https://maven.apache.org/install.html) to install Apache Maven.
