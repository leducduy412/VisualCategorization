# Visual Categorization with Bag of Keypoints - Group 8

## Overview
Our project is an in-depth exploration into visual categorization using Bag of Keypoints (BoK) and Spatial Pyramid Matching (SPM). We've based our research and implementation on two key papers, extending their methodologies to optimize and compare their performances on standard datasets.


## Our based papers
1. **Visual Categorization with Bags of Keypoints**: This foundational paper discusses categorizing visuals using a bag of keypoints approach, a method that quantizes local descriptors into a collective histogram.
2. **Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories**: This paper extends the BoK concept, introducing Spatial Pyramid Matching (SPM) to incorporate spatial information into the categorization process.

## Our work
- **Implementation**: We've implemented the BoK and SPM algorithms, ensuring adherence to the methodologies proposed in the papers.
- **Optimization**: By tuning the 'k' (number of clusters in k-means) and 'C' (regularization parameter in SVM) parameters, we've optimized the model's performance.
- **Testing**: Our models have been tested on the Caltech-101 and another custom dataset, providing a broad basis for performance evaluation.
- **Comparison**: A comparative analysis between the results of BoK and SPM provides insights into the strengths and weaknesses of each method.

### Datasets Used
- **Caltech-101**: A widely used dataset for visual object recognition tasks.
- **Dataset 2**: An additional dataset to further test and validate our models.

### Key Performance Metrics
- Confusion Matrix
- Error Rate
- Accuracy

## Results Overview

We found that the choice of 'k' and 'C' significantly influences the model's performance. Our experiments suggest that a 'k' value of 1000 and a 'C' value of 0.0001 provides a good balance between accuracy and computational efficiency.

Below are the accuracy results for each dataset:

| Dataset     | Accuracy  |
|-------------|-----------|
| Caltech-101 | 31.35%    |
| Dataset 2   | 60.29%    |

These results demonstrate the effectiveness of our models and optimization strategies. The detailed confusion matrices and further analysis are available in our [comprehensive report](https://drive.google.com/file/d/1nLKfMoRdf1AGfQUSnw6BhtmRKCzZXevJ/view?usp=sharing).

### Comparative Analysis
The BoK method, while effective, ignores the spatial location of descriptors, leading to a loss of spatial information in the final feature vector. On the other hand, SPM, which incorporates spatial information, showed a marked improvement in categorization accuracy.

| Method      | Caltech-101 Accuracy | Dataset 2 Accuracy |
|-------------|----------------------|--------------------|
| BoK         | 31.35%               | 60.29%             |
| SPM L=2     | 40.45%               | 64.59%             |
| SPM L=3     | --                   | 67.46%             |

For Caltech-101, the accuracy also increased from 31.35% to 40.45% when using SPM with L=2, demonstrating the effectiveness of including spatial information.
file:///home/duc/Pictures/Screenshots/Screenshot%20from%202023-12-31%2007-52-50.png
file:///home/duc/Pictures/Screenshots/Screenshot%20from%202023-12-31%2007-53-04.png
file:///home/duc/Pictures/Screenshots/Screenshot%20from%202023-12-31%2007-53-22.png

## Our report
[Visual Categorization Group 8](https://drive.google.com/file/d/1nLKfMoRdf1AGfQUSnw6BhtmRKCzZXevJ/view?usp=sharing)

## Getting Started

### Installation
Clone the repository and navigate to the project directory. Install the necessary requirements to get started with the experiments.

```bash
git clone https://github.com/your-github/visual-categorization-group-8.git
cd visual-categorization-group-8
pip install -r requirements.txt

## Usage
Navigate to the 'experiment' folder to run the experiments. Execute the scripts to find optimal parameters and test the datasets.
- Cloning this project to your computer.
- Opeing the project folder, and also the cmd.
- First, installing all the requirements of the project:
```
pip install requirements.txt
```
- Then you can move to the folder "experiment" to run our experiment:
```
# To find the k parameter
py find_k.py

# To find the C parameter
py find_C.py

# Run with dataset 1
py test_data1.py

# Run with dataset 2
py test_data2.py
```
