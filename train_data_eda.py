
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Data Cleaning

df_result = pd.read_csv('/content/drive/MyDrive/cnc-project/data_v1_train/train.csv')

df_result.head()

df_result.info()

df_result.isnull().sum()

df_result['passed_visual_inspection']

df_result['passed_visual_inspection'] = df_result['passed_visual_inspection'].fillna('no')

df_result.isnull().sum()

---

# 2. EDA

df_result

### **i. Univariate Analysis**

df_result['material'].nunique()

# Setting up a figure
plt.figure(figsize=(15, 4))
sns.set(style="whitegrid")

# Plot 1: tool_condition
plt.subplot(1, 3, 1)
sns.countplot(data=df_result, x='tool_condition', hue = 'tool_condition', palette='dark')
plt.title('Tool Condition')
plt.xlabel('')
plt.ylabel('Count')

# Plot 2: machining_finalized
plt.subplot(1, 3, 2)
sns.countplot(data=df_result, x='machining_finalized', palette='colorblind', hue = 'machining_finalized')
plt.title('Machining Finalized')
plt.xlabel('')
plt.ylabel('')

# Plot 3: passed_visual_inspection
plt.subplot(1, 3, 3)
sns.countplot(data=df_result, x='passed_visual_inspection', palette='bright' , hue = 'passed_visual_inspection')
plt.title('Visual Inspection')
plt.xlabel('')
plt.ylabel('')

plt.tight_layout()
plt.show()

# Plot 4: Density plot for Feedrate
plt.figure(figsize=(8, 4))
sns.kdeplot(data=df_result, x='feedrate', fill=True, color='skyblue', linewidth=2)
plt.title('Density Plot of Feedrate')
plt.xlabel('Feedrate')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

# Plot 5: Density plot for Clamp Pressure
plt.figure(figsize=(8, 4))
sns.kdeplot(data=df_result, x='clamp_pressure', fill=True, color='orange', linewidth=2)
plt.title('Density Plot of Clamp Pressure')
plt.xlabel('Clamp Pressure')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

## **ii. Bivariate**

pd.crosstab(df_result['tool_condition'], df_result['passed_visual_inspection'])

pd.crosstab(df_result['tool_condition'], df_result['machining_finalized'])

sns.kdeplot(data=df_result, x='feedrate', hue='tool_condition', fill=True)

sns.kdeplot(data=df_result, x='clamp_pressure', hue='tool_condition', fill=True)

## **iii. Multivariate**

# Creating a crosstab
ct = pd.crosstab(index=[df_result['tool_condition'], df_result['machining_finalized']],
                 columns=df_result['passed_visual_inspection'])

# Plotting as heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(ct, annot=True, cmap='YlGnBu', fmt='d')
plt.title('Inspection Outcome by Tool Condition & Machining Finalized')
plt.ylabel('Tool Condition & Machining Finalized')
plt.xlabel('Passed Visual Inspection')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot 6: by tool_condition
sns.scatterplot(data=df_result, x='clamp_pressure', y='feedrate', hue='tool_condition', style='tool_condition', s=100, ax=axs[0])
axs[0].set_title('Feed Rate vs Clamp Pressure by Tool Condition')
axs[0].set_xlabel('Feed Rate')
axs[0].set_ylabel('Clamp Pressure')
axs[0].grid(True)

# Plot 7: by machining_finalized
sns.scatterplot(data=df_result, x='feedrate', y='clamp_pressure', hue='machining_finalized', style='machining_finalized', s=100, ax=axs[1])
axs[1].set_title('Feed Rate vs Clamp Pressure by Machining Finalized')
axs[1].set_xlabel('Feed Rate')
axs[1].set_ylabel('Clamp Pressure')
axs[1].grid(True)

plt.tight_layout()
plt.show()

