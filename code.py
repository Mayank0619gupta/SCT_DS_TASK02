import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
uploaded = files.upload()
data = pd.read_csv("titanic.csv")
data.columns = data.columns.str.strip()
print(data.head())
print("\nMissing values in each column:\n", data.isnull().sum())

if 'Age' in data.columns:
    data['Age'].fillna(data['Age'].median(), inplace=True)

if 'Embarked' in data.columns:
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

if 'Cabin' in data.columns:
    data.drop('Cabin', axis=1, inplace=True)

data.drop_duplicates(inplace=True)

# Plot 1:Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=data, palette='Set2')
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Plot 2: Gender vs Survival
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=data, palette='Set1')
plt.title('Survival by Gender')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Plot 3: Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(data['Age'], bins=30, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Plot 4: Age vs Survival
plt.figure(figsize=(8,5))
sns.histplot(data=data, x='Age', hue='Survived', bins=30, kde=True, palette='coolwarm', alpha=0.6)
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Plot 5: Passenger Class vs Survival
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=data, palette='viridis')
plt.title('Survival by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()

# Plot 6: Correlation Heatmap
plt.figure(figsize=(10,6))
corr = data[['Survived', 'Pclass', 'Age','Fare']].corr()
sns.heatmap(corr, annot=True, cmap='magma', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Plot 7: Fare Distribution by Survival
plt.figure(figsize=(8,5))
sns.boxplot(x='Survived', y='Fare', data=data, palette='pastel')
plt.title('Fare Distribution by Survival')
plt.xlabel('Survived')
plt.ylabel('Fare')
plt.show()

# Plot 8: Embarked vs Survival — Only if column exists
if 'Embarked' in data.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x='Embarked', hue='Survived', data=data, palette='Accent')
    plt.title('Survival by Embarkation Port')
    plt.xlabel('Embarked')
    plt.ylabel('Count')
    plt.show()

# Plot 9: Pairplot for numerical features
sns.pairplot(data[['Survived', 'Pclass', 'Age', 'Fare']], hue='Survived', palette='cool', diag_kind='kde')
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()
