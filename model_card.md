# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Sudarshana Sharma created this Model. It is a Random Forest Classifier model trained on census data.

## Intended Use
The model can be used to predict the salary level (<=50k or >=50k) of an individual based on the following attributes:
age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week and native-country.

## Training Data
The data ,also known as Adult dataset, was <a href='https://archive.ics.uci.edu/dataset/20/census+income'> extracted </a> from the 1994 Census database by Ronny Kohavi and Barry Becker. 
Training data is used from <a href='https://github.com/udacity/nd0821-c3-starter-code/blob/master/starter/data/census.csv'> here </a> <br>
A 80-20 split was used to break this dataset into a train and test set. Stratification on target label "salary" was applied. To use the data for training a One Hot Encoder was used on the categorical features and a label binarizer was used on the target.

## Evaluation Data
Evaluation data is the 20% of full downloaded data by using `train_test_split` function from scikit-learn.
Model has been evaluated by *fbeta*, *precision* and *recall* scores. And the score on Evaluation data is below:
`Precision: 0.7352281226626777, Recall: 0.6269132653061225, Fbeta: 0.6767641996557661`

## Ethical Considerations
Fairness and bias: There is a risk of bias in the dataset due to the way it was collected. For example, some groups of people may be underrepresented in the dataset, leading to biased predictions. This could result in unfair treatment of certain individuals or groups.

Privacy and confidentiality: The dataset contains sensitive information about individuals, such as their income, occupation, and education level. It is important to ensure that this information is not misused or shared in a way that could harm individuals.

Data quality and accuracy: There may be errors or inaccuracies in the dataset, which could lead to incorrect predictions. It is important to validate the data and ensure that it is of sufficient quality for the intended use.

Transparency and explainability: It is important to be transparent about how the dataset was collected, processed, and used. This can help build trust with users and stakeholders, and ensure that the predictions are fair and accurate.

Algorithmic accountability: The use of machine learning algorithms to make predictions based on the dataset raises questions of accountability. It is important to ensure that the algorithms are fair and transparent, and that their decisions can be explained and challenged if necessary.

## Caveats and Recommendations
This model is not suitable for real-time predictions. It is suitable for batch predictions.