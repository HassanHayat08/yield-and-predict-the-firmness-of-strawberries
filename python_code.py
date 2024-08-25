"""
This script develops a Machine Learning model using multilabel classification to maximize yield 
and predict the firmness of strawberries.

The script performs the following steps:
1. Load and preprocess the data.
2. One-hot encode the multilabel target variable.
3. Select and standardize the features.
4. Split the data into training and test sets.
5. Train a multilabel classification model.
6. Evaluate the model using a confusion matrix and classification report.
7. Save the trained model to a file.
8. Load the saved model and evaluate it on unseen data.
9. Perform linear interpolation to find y-values based on given x-values.
"""

# importing libraries
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d	
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
import pickle

# function for linear interpolation
def interpolation(X, Y, interpolate_x):
    """
    This function performs linear interpolation on a given set of X and Y values,
    and returns the interpolated Y values for the specified X values.

    Parameters:
    X (list): A list of X values (independent variable) on which the interpolation is performed.
    Y (list): A list of Y values (dependent variable) corresponding to the X values.
    interpolate_x (list): A list of X values for which the interpolation needs to be performed.

    Returns:
    list: A list of interpolated Y values corresponding to the specified X values.
    """
    a = []
    for x in interpolate_x:
        # Perform linear interpolation using scipy's interp1d function
        y_interp = interp1d(X, Y)
        a.append(float(y_interp(x)))
    return a

# function for conditions 
# apply multiple condition to get the class of fairness values:
# class C: less than 3.2 N; class D: 3.2 to 5.5 N; class E: 5.6 to 7.9 N; class F: 8.0 to 10.4 N
def condition(x):
  """
  This function assigns a class label to a given strawberry firmness value based on specific conditions.

  Parameters:
  x (float): The strawberry firmness value to be classified.

  Returns:
  str: The class label ('C', 'D', 'E', or 'F') corresponding to the given firmness value.
  """
  if x < 56.00:
    return 'C'
  elif x >= 56.00 and x <= 65.99:
    return 'D'
  elif x >= 66.00 and x <= 75.99:
    return 'E'
  elif x >= 76.00 and x <= 86:
    return 'F'

# function for model output interpretation
def output_interpret(results, X, Y, values_interpolate):
  """
  This function interprets the output of a machine learning model for predicting strawberry yield and firmness.
  It takes the model's prediction results, X and Y values for interpolation, and specific values for interpolation.
  It then prints the interpretation based on the predicted results and the interpolated values.

  Parameters:
  results (list): The model's prediction results, where each element represents a class label (0 or 1).
  X (list): A list of X values (independent variable) on which the interpolation is performed.
  Y (list): A list of Y values (dependent variable) corresponding to the X values.
  values_interpolate (list): A list of X values for which the interpolation needs to be performed.

  Returns:
  None. The function prints the interpretation based on the input parameters.
  """
  fairness_vales = interpolation(X, Y, values_interpolate)
  A = 'high yield'
  B = 'low yield'
  C = '<3.2N'
  D = '3.2 to ' + str(fairness_vales[0]) + 'N'
  E = str(fairness_vales[1]) + ' to ' + str(fairness_vales[2]) + 'N'
  F = str(fairness_vales[3]) + ' to ' + '10.4N'
  array = []
  for i in range(len(results)):
      if i == 0 and results[i] == 1:
          array.append('A')
      elif i == 1 and results[i] == 1:
          array.append('B')
      elif i == 2 and results[i] == 1:
          array.append('C')
      elif i == 3 and results[i] == 1:
          array.append('D')
      elif i == 4 and results[i] == 1:
          array.append('E')
      elif i == 5 and results[i] == 1:
          array.append('F')
  for x in array:
      if x == 'A':
          print(f'Based on given data, the farmer will get {A} and')
      elif x == 'B':
          print(f'Based on given data, The farmer will get {B} and')
      elif x == 'C':
          print(f'the strawbery firmness values are between {C}')
      elif x == 'D':
          print(f'the strawbery firmness values are between {D}')
      elif x == 'E':
          print(f'the strawbery firmness values are between {E}')
      elif x == 'F':
          print(f'the strawbery firmness values are between {F}')


# Data Processing class
class DataProcessing:
  """
  This class is responsible for processing and preparing the data for the machine learning model.
  It includes methods for finding missing values, labeling yield and fairness, replacing missing values,
  and generating final labels for the dataset.

  Parameters:
  PATH (str): The file path of the dataset to be processed.

  Attributes:
  path (str): The file path of the dataset.
  df (DataFrame): The pandas DataFrame containing the dataset.
  """

  def __init__(self, PATH):
      """
      Initializes the DataProcessing class with the given file path.

      Parameters:
      PATH (str): The file path of the dataset to be processed.
      """
      self.path = PATH

  def find_missing_vales(self):
      """
      Finds and returns the count of missing values in each column of the dataset.

      Returns:
      Series: A pandas Series containing the count of missing values in each column.
      """
      return self.df.isna().sum()

  def yield_label(self):
      """
      Adds a new column 'yield_labels' to the dataset, indicating whether the yield is maximum (Class A)
      or not (Class B) based on the air humidity level.

      Returns:
      DataFrame: The updated DataFrame with the 'yield_labels' column added.
      """
      self.df['yield_labels'] = self.df['airhumidity'].apply(lambda x: 'A' if x >= 65.00 and x <= 75.00 else 'B')
      return self.df

  def get_multilabels(self):
      """
      Adds a new column 'fairness_labels' to the dataset, indicating the fairness class based on the air humidity level.

      Returns:
      DataFrame: The updated DataFrame with the 'fairness_labels' column added.
      """
      self.df['fairness_labels'] = self.df['airhumidity'].apply(condition)
      return self.df

  def replace_missing_values(self):
      """
      Replaces missing values in the dataset with the mean of the respective column.

      Returns:
      DataFrame: The updated DataFrame with missing values replaced.
      """
      features_list = ['airtemp', 'airhumidity', 'irtemp', 'dewpoint']
      for feature in features_list:
          self.df[feature] = self.df[feature].fillna(self.df[feature].mean())
      return self.df

  def get_final_labels(self):
      """
      Combines the 'yield_labels' and 'fairness_labels' columns to create a new column 'final_labels'.

      Returns:
      DataFrame: The updated DataFrame with the 'final_labels' column added.
      """
      self.df['final_labels'] =  self.df['yield_labels'] + self.df['fairness_labels']
      return self.df

  def get_data(self):
      """
      Performs all the data processing steps (finding missing values, labeling yield and fairness, replacing missing values,
      and generating final labels) and returns the processed DataFrame.

      Returns:
      DataFrame: The processed DataFrame ready for machine learning model training.
      """
      self.df = pd.read_csv(self.path)
      self.df = self.replace_missing_values()
      self.df = self.yield_label()
      self.df = self.get_multilabels()
      self.df = self.get_final_labels()
      return self.df
  
def load_and_evaluate_model(model_save_path, X_test, Y_test, label_names):
  """
  Load a saved model and evaluate it on the test data.

  Parameters:
  model_save_path (str): Path to the saved model file.
  X_test (numpy.ndarray): Test features.
  Y_test (numpy.ndarray): True labels for the test data.
  label_names (list of str): Names of the labels for the classification report.

  Returns:
  None
  """
  # Load the saved model
  loaded_model = pickle.load(open(model_save_path, 'rb'))
  
  # Predict the labels for the test data
  Y_pred = loaded_model.predict(X_test)
  
  # Print the classification report
  print(classification_report(Y_test, Y_pred, target_names=label_names))
  return None
          
def process_and_train_model(data_path, model_save_path):
  """
  Process the data, train the model, and save the model parameters.

  Parameters:
  data_path (str): Path to the CSV file containing the data.
  model_save_path (str): Path to save the trained model.

  Returns:
  None
  """
  # Load and process the data
  data = DataProcessing(data_path)
  df = data.get_data()

  # One-hot encode the multilabel target variable
  one_hot = MultiLabelBinarizer()
  lab_array = np.array(df["final_labels"])
  Y = one_hot.fit_transform(lab_array)

  # Select the features for the model
  new_df = df[['airtemp', 'airhumidity', 'irtemp', 'dewpoint']]

  # Standardize the feature data
  X = new_df.to_numpy()
  X = stats.zscore(X)

  # Split the data into training and test sets
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

  # Initialize and train the classification model
  base_lr = LogisticRegression()
  model = ClassifierChain(base_lr, order='random', cv=5, random_state=0)
  model.fit(X_train, Y_train)

  # Predict the labels for the test data
  y_pred = model.predict(X_test)

  # Generate and print the multilabel confusion matrix
  print(multilabel_confusion_matrix(Y_test, y_pred))

  # Define the label names for the classification report
  label_names = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E', 'Class F']

  # Print the classification report
  print(classification_report(Y_test, y_pred, target_names=label_names))

  # Interpret the model output for a single sample
  print("Interpretation for a single sample:")    
  single_sample = X_test[10]
  single_sample = single_sample.reshape(1, -1)
  y_pred_single = model.predict(single_sample).reshape(-1)

  X = [56.00,86.00] # x-axis humidity values
  Y = [3.2,10.4] # y-axis fairness values
  # get fainess (y) value based on humidity level (x-axis)
  values_interpolate = [65.99,66.00,75.99,76.00] # different humidity levels
  output_interpret(y_pred_single, X, Y, values_interpolate)

  # Save the trained model parameters to a file
  pickle.dump(model, open(model_save_path, 'wb'))

  # Load the saved model and evaluate it on the test data
  load_and_evaluate_model('./strawfirmnessmodel.sav', X_test, Y_test, 
                        ['Class A', 'Class B', 'Class C', 'Class D', 'Class E', 'Class F'])
  return None

# Process and train the model
process_and_train_model('./FargroSample.csv', './strawfirmnessmodel.sav')


