# Travel-Mode Inference Based On GPS-Trajectory Data Through Multi-Scale Mixed Attention Mechanism

## ABSTRACT
Identifying travel modes is essential for contemporary urban transportation planning and management. Innovative data collection methods, particularly those utilizing Global Positioning Systems (GPS), offer new prospects for quickly and rationally inferring users’ travel modes. This study introduces a novel approach to travel mode inference based on GPS trajectory data. The method integrates multi-scale convolutional technology to extract both temporal and spatial information from GPS trajectory data, thereby discerning the inherent spatiotemporal correlations in user motion behavior and movement patterns. Additionally, the incorporation of an attention mechanism enables the model to autonomously learn, facilitating the identification and prioritization of relevant information across various time periods and spatial coordinates. Consequently, this enhances the focus on salient features while mitigating sensitivity to external noise. Empirical evaluation conducted on the open-source GPS trajectory data set, GeoLife, shows that the proposed method achieved an accuracy of 83.3%. These findings confirm the potential of this approach, wherein a judicious combination of multi-scale convolutional technology and attention mechanisms enables the model to comprehend and predict travel modes more accurately.

![An illustration of four motion signals extracted from GPS trajectories](./figs/An%20illustration%20of%20four%20motion%20signals%20extracted%20from%20GPS%20trajectories.png)

## Running environment

* Ubuntu 20.04.6 LTS
* NVIDIA GeForce RTX 3080
* Python 3.9
* PyTorch 2.0.1

## Execution steps

1. Download the dataset from https://drive.google.com/file/d/17rESi-JG3arz3vLMBTtV8RvB4LAamIu9/view?usp=drive_link
2. Unzip the dataset in the data folder
3. Execute code in numbered order

## Cite

If our work is helpful to you, please cite:
(the work is under review)


