# AER-850 Intro to Machine Learning (Mehtab Singh 500960754)
import pandas as pd
# from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, log_loss, mean_squared_error, mean_absolute_error, r2_score

# Step 1: Data Processing
data_csv_file = "Project 1 Data.csv"
# making dataframe  
dataframe = pd.read_csv(data_csv_file) 

#______________________________________________________________________________

# Step 2: Data Visualization
# Converting dataframe to an array
data_array = dataframe.to_numpy()
#print(data_array)

# 3D Plot of x, y, z
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("3D Plot of x,y,z coordinates for Motion Simulator")
x = data_array[:, 0]
y = data_array[:, 1]
z = data_array[:, 2]
step = data_array[:, 3]
ax.scatter(x, y, z)
plt.show()

# Create a step and x relationship
plt.plot(step, x)
# Set labels and title
plt.xlabel('Step')
plt.ylabel('X-axis')
plt.title('Step to X Plot for Motion Simulator')
# Show the plot
plt.show()

# Create a step and y relationship
plt.plot(step, y)
# Set labels and title
plt.xlabel('Step')
plt.ylabel('Y-axis')
plt.title('Step to Y Plot for Motion Simulator')
# Show the plot
plt.show()

# Create a step and z relationship
plt.plot(step, z)
# Set labels and title
plt.xlabel('Step')
plt.ylabel('Z-axis')
plt.title('Step to Z Plot for Motion Simulator')
# Show the plot
plt.show()
#______________________________________________________________________________

# Step 3: Correlation Analysis

# Presenting columns and making correlation matrix of data
columns_rows = ['x', 'y', 'z', 'Step']
corr_matrix_data = dataframe.corr()
corr_matrix_data = corr_matrix_data.to_numpy()
# check to see correlation matrix
#print(corr_matrix_data)
# Plot the correlation martix as a Heatmap diagram using seaborn library
ax = plt.axes()
sns.heatmap(corr_matrix_data, annot=True, xticklabels=columns_rows, yticklabels=columns_rows)
ax.set_title("Heatmap Correlation Martix for Motion Simulator")
plt.show()

# Correlation calculation & stratified sampling (Step is the Stratify variable)
X_train, X_test, Y_train, Y_test, Z_train, Z_test, step_train, step_test = train_test_split(data_array[:,0], data_array[:,1], data_array[:,2], data_array[:,3], test_size=0.2, stratify=data_array[:,3], random_state=42)

# Calculate correlations
correlation_X = np.corrcoef(step_train, X_train)[0, 1]
correlation_Y = np.corrcoef(step_train, Y_train)[0, 1]
correlation_Z = np.corrcoef(step_train, Z_train)[0, 1]

#print(step_train)

# Creating the DataFrame from X, Y, Z, Step training data
df_train = pd.DataFrame({'Y': Y_train, 'X': X_train, 'Z': Z_train, 'Step': step_train})

# Calculate correlations
correlation_matrix = df_train.corr()

# Access the correlations for Step
correlation_step = correlation_matrix['Step']

print(f"Correlation with Step:{correlation_step}")
#______________________________________________________________________________

# Step 4: Classification Model

# Classification Model 1: Logistic Regression
x = dataframe.drop(columns=["Step"])
y = dataframe['Step']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Print sample confusion matrix and classification report
cm_log = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_rep}")

# Classification Model 2: Random Forest Regressor
# Create the model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)
y_pred_reg = rf_regressor.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred_reg)
mae = mean_absolute_error(y_test, y_pred_reg)
r2 = r2_score(y_test, y_pred_reg)

# Print MAE
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared (R2): {r2}")

# Classification Model 3: Decision Tree
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred_tree = dt_classifier.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)

# Print sample confusion matrix and classification report
report = classification_report(y_test, y_pred_tree)
cm_tree = confusion_matrix(y_test, y_pred_tree)
print(f"Accuracy: {accuracy_tree}")
print(f"Classification Report:{report}")
#______________________________________________________________________________

# Step 5: Model Analysis
# Creating a Confusion Matrix plot
# Plot the Logistic Regression confusion matrix as a heatmap
columns_rows_step = np.arange(1, 14)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_log, annot=True, fmt="d", cmap="Blues", xticklabels=columns_rows_step, yticklabels=columns_rows_step)
plt.xlabel('Predicted (Step)')
plt.ylabel('Actual (Step)')
plt.title('Logistic Regression Confusion Matrix for Motion Simulator')
plt.show()

# Plot the Decision Tree confusion matrix as a heatmap
columns_rows_step = np.arange(1, 14)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_tree, annot=True, fmt="d", cmap="Blues", xticklabels=columns_rows_step, yticklabels=columns_rows_step)
plt.xlabel('Predicted (Step)')
plt.ylabel('Actual (Step)')
plt.title('Decision Tree Confusion Matrix for Motion Simulator')
plt.show()
#______________________________________________________________________________

# Step 6: Model Evaluation
# Evaluation data for each array presented
features_array1 = [9.375, 3.0625, 1.51]
features_array2 = [6.995, 5.125, 0.3875]
features_array3 = [0, 3.0625, 1.93]
features_array4 = [9.4, 3, 1.8]
features_array5 = [9.4, 3, 1.3]
feature_arrays = [features_array1, features_array2, features_array3, features_array4, features_array5]

predictions = dt_classifier.predict(feature_arrays)

print(predictions)
#______________________________________________________________________________

# Write Outputs to file
with open('Project1_Results_MehtabSingh.txt', 'w') as file:
    # All values required for the steps are placed here
    file.write("AER850-Project 1 Results (Mehtab Singh 500960754)\nStep 3 Results:\n")
    file.write(f"Correlation with Step: {correlation_step}")
    file.write("\n\nStep 4 Results")
    file.write("\nLogistic Regression Results:\n")
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"Classification Report:\n{classification_rep}\n")
    file.write("\nRandom Forest Regressor Results:\n")
    file.write(f"Mean Squared Error: {mse}\n")
    file.write(f"Mean Absolute Error: {mae}\n")
    file.write(f"R-squared (R2): {r2}\n")
    file.write("\nDecision Tree Results:\n")
    file.write(f"Accuracy: {accuracy_tree}\n")
    file.write(f"Classification Report: {report}\n")
    file.write(f"\n\nStep 6 Results:\n {predictions}")
file.close()