## Face Classifier and Verifier Using Pre-trained FaceNet Model

### Project Overview
In this project, we designed a Face Classifier and a Face Verifier system using pre-trained weights from the FaceNet model. The implementation leverages PyTorch and the `pytorch-facenet` library.

### Architecture
The deep architecture used in this project combines GoogLeNet (Inception modules) and ResNet. We utilize the `pytorch-facenet` library, which provides an implementation of the InceptionResnet architecture and includes pre-trained models on Webface-CASIA and VGGFace2 datasets. For this exercise, we use weights trained on the VGGFace2 dataset.

### Dataset
The dataset consists of face images of 14 actors, each with approximately 100 images sized 160 × 160 pixels.

### Implementation Steps

1. **Model Initialization**:
   - Create the InceptionResnetV1 model and load the pre-trained weights from the VGGFace2 dataset using the following code:
     ```python
     model = InceptionResnetV1(pretrained='vggface2').eval()
     ```

2. **Model Modification**:
   - Add a small MLP network to the end of the model. This MLP network will take embeddings as input and output class predictions for the dataset. The number of layers, neurons per layer, and optimal configurations are determined using training techniques such as dropout and batch normalization.

3. **Training**:
   - Evaluate the impact of freezing the layers of InceptionResnetV1 versus fine-tuning its final layers during training on the dataset.

### Face Verifier System
Using the learned embeddings, we will create a Face Verifier system. The similarity between embedding vectors can be calculated using cosine similarity or Euclidean distance.

### Reporting
- **Classifier Evaluation**:
  - Document the model architecture, training process, and evaluation metrics.
- **Verifier Evaluation**:
  - Divide the test data into known and unknown images. Each unknown image should be compared with all known images to determine the identity based on the highest similarity. If the similarity is below a certain threshold, report a mismatch.
  - Compare and discuss the performance of cosine similarity and Euclidean distance for this task.

This project involves fine-tuning deep learning models and experimenting with different training techniques and similarity measures to achieve accurate face classification and verification.
