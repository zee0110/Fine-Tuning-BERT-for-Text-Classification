# Fine-Tuning-BERT-for-Text-Classification

### README

#### Fine-Tuning BERT for Text Classification

This code demonstrates the process of fine-tuning a BERT (Bidirectional Encoder Representations from Transformers) model for text classification using TensorFlow. The model is trained on a dataset containing labeled text samples to classify them into one of three categories: Hate Speech, Offensive Language, or Neither. 

#### Prerequisites

Ensure you have Python installed on your system along with the necessary libraries:
- datasets
- transformers
- TensorFlow
- NumPy
- matplotlib
- pandas
- seaborn

You can install the required libraries using pip:

```
pip install transformers datasets tensorflow numpy matplotlib pandas seaborn
```

#### Usage

1. Clone the repository or download the code files.
2. Ensure you have the dataset file "labeled_data.csv" in the same directory as the code.
3. Run the code using a Python interpreter.
4. The code loads the dataset, preprocesses it, and formats it for fine-tuning the BERT model.
5. It tokenizes the text data using the BERT tokenizer and prepares the dataset for training, validation, and testing.
6. The BERT model is initialized and compiled with an optimizer, loss function, and evaluation metric.
7. Training is performed on the training dataset, and the model's performance is evaluated on the validation dataset.
8. The training progress is visualized using matplotlib.
9. After training, the model is evaluated on the test dataset, and its accuracy is printed.
10. Finally, the model is used to make predictions on new text samples, and the predicted classes are displayed.

#### Features

- Loads and preprocesses text classification dataset.
- Tokenizes text data using BERT tokenizer.
- Fine-tunes a pre-trained BERT model for text classification.
- Compiles the model with appropriate optimizer, loss function, and evaluation metric.
- Trains the model on the training dataset and evaluates its performance on the validation dataset.
- Visualizes training progress using matplotlib.
- Evaluates the model's accuracy on the test dataset.
- Makes predictions on new text samples using the trained model.

