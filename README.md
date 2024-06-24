# Alzheimer-s-Disease-Classification-using-InceptionV3
Alzheimer’s Disease Classification using InceptionV3 employs a deep learning model to classify brain MRI scans into various stages of Alzheimer's Disease. Utilizing TensorFlow and Keras, the project aims to aid in early diagnosis and treatment planning, leveraging advanced image processing and transfer learning techniques for accurate detection.

### Inspiration
The inspiration for this project stems from the significant global challenge posed by Alzheimer’s Disease (AD). Witnessing the debilitating effects of AD on patients and their families, we were motivated to leverage the advancements in deep learning to aid in early diagnosis. The potential to improve patient outcomes by detecting the disease early inspired us to use a cutting-edge deep learning model, InceptionV3, for this purpose.

### What it Does
This project classifies brain MRI scans into four categories: Non Demented, Mild Demented, Moderate Demented, and Very Mild Demented. Using the InceptionV3 model, the system analyzes MRI images to distinguish between healthy brains and those affected by various stages of Alzheimer’s Disease. This classification helps in early detection and better management of the disease.

### How We Built It
1. **Data Collection and Preprocessing:**
   - Collected a dataset of 6400 MRI images, categorized into Non Demented, Mild Demented, Moderate Demented, and Very Mild Demented.
   - Resized all images to 128x128 pixels.
   - Addressed class imbalance using the SMOTE (Synthetic Minority Over-Sampling Technique) to generate synthetic samples for minority classes.

2. **Model Selection and Architecture:**
   - Employed the InceptionV3 model, pre-trained on ImageNet, for feature extraction.
   - Removed the top layer of the InceptionV3 model and added a new sequential model with several dense layers, batch normalization, dropout regularization, and a softmax output layer for classification.
   - Implemented other models like VGG19, ResNet50, and EfficientNet for comparative analysis.

3. **Training and Validation:**
   - Trained the model using the processed dataset, with a split for training and validation to monitor performance.
   - Used dropout regularization and global average pooling to prevent overfitting and reduce spatial dimensions, respectively.

### Challenges We Ran Into
- **Class Imbalance:** The dataset was imbalanced, with fewer images in some categories, which we addressed using SMOTE.
- **Overfitting:** Preventing the model from overfitting the training data was challenging, which we mitigated using dropout regularization and careful monitoring of validation metrics.
- **Computational Resources:** Training deep learning models on large datasets required significant computational power and time, necessitating the use of efficient coding practices and optimization techniques.

### Accomplishments That We're Proud Of
- Successfully implemented and fine-tuned the InceptionV3 model for multi-class classification of Alzheimer’s Disease stages.
- Achieved high accuracy in distinguishing between different stages of Alzheimer’s Disease, demonstrating the potential of deep learning in medical diagnosis.
- Addressed and mitigated class imbalance effectively, ensuring the model performs well across all categories.

### What We Learned
- Gained deep insights into the preprocessing and augmentation of medical imaging data.
- Learned to implement and fine-tune complex deep learning architectures for specific tasks.
- Understood the importance of addressing class imbalance and preventing overfitting in deep learning models.
- Developed skills in using various deep learning frameworks and libraries for model building and evaluation.

### What's Next for Alzheimer’s Disease Classification using InceptionV3
- **Model Enhancement:** Explore the integration of more advanced architectures and ensemble methods to further improve classification accuracy.
- **Dataset Expansion:** Incorporate more diverse and larger datasets to enhance the model's robustness and generalizability.
- **Clinical Trials:** Collaborate with medical professionals to validate the model's effectiveness in clinical settings and gather real-world feedback.
- **User Interface Development:** Create a user-friendly interface for healthcare professionals to easily utilize the model for early diagnosis and treatment planning.
- **Continuous Learning:** Implement a system for continuous learning where the model can be periodically updated with new data to maintain its accuracy and relevance.
