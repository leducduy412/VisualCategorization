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

![BOK results](https://i.ibb.co/ngxTBVP/Screenshot-from-2023-12-31-08-06-54.png)
![SPM L=2 results](https://i.ibb.co/LCHYymK/Screenshot-from-2023-12-31-08-07-10.png)
![SPM L=3 results](https://i.ibb.co/jMxpjSB/Screenshot-from-2023-12-31-08-07-23.png)

## Usage
To use this project:
- Clone the project, install requirements:
    ```bash
    git clone https://github.com/your-github/visual-categorization-group-8.git
    cd visual-categorization-group-8/src
    pip install -r requirements.txt
    ```
- Navigate to the respective script directory and run the desired script. For example:
    ```bash
    # To find the optimal 'k' parameter:
    python models/nb/find_k.py

    # To find the optimal 'C' parameter:
    python models/nb/find_C.py

    # To test with dataset 1:
    python data/test/test_data1.py

    # To test with dataset 2:
    python data/test/test_data2.py
    ```

## Our report
For an in-depth understanding of our methodologies, results, and analyses, refer to our [detailed report](https://drive.google.com/file/d/1nLKfMoRdf1AGfQUSnw6BhtmRKCzZXevJ/view?usp=sharing)

## Contributing
We welcome contributions, suggestions, and issues. Please read the contributing guide before making any pull request.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## References
This project is based on the methodologies and concepts presented in the following papers:
```
@article{article,
  author = {Csurka, Gabriela and Dance, Christopher and Fan, Lixin and Willamowski, Jutta and Bray, Cédric},
  year = {2004},
  month = {01},
  pages = {},
  title = {Visual categorization with bags of keypoints},
  volume = {Vol. 1},
  journal = {Work Stat Learn Comput Vision, ECCV}
}

@INPROCEEDINGS{1641019,
  author={Lazebnik, S. and Schmid, C. and Ponce, J.},
  booktitle={2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06)}, 
  title={Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories}, 
  year={2006},
  volume={2},
  number={},
  pages={2169-2178},
  doi={10.1109/CVPR.2006.68}}
}

```
