# Sequential SHAP

In this study, the limitations of the SHAP (Shapley Additive exPlanations) method have been examined within the scope of Explainable Artificial Intelligence (XAI), particularly regarding **sequential multi-class classification problems**. 

Stemming from the observation that the classical SHAP method is restricted to revealing only positive and negative contributions for a single class, a modified alternative approach is proposed to overcome these constraints. This novel method aims to elucidate the model's decision-making mechanism in a more granular manner by categorizing feature effects into distinct semantic groups.

## The Sequential Solution

**Sequential SHAP** addresses standard SHAP restrictions by breaking down the multi-class problem into two distinct binary classification stages based on a provided semantic class hierarchy (order). It categorizes features into three distinct groups:

- **UPPER (Red):** Features driving the prediction towards a higher class.
- **LOWER (Blue):** Features driving the prediction towards a lower class.
- **AMBIGUOUS (Gray):** Features demonstrating inconsistent or uninformative effects across class transitions.

The findings obtained from analyses—such as on an obesity dataset—indicate that this proposed approach offers more profound insights than standard SHAP analysis and enables precise interpretations regarding class transitions. In conclusion, this modification significantly enhances the transparency of the model, presenting a robust analytical framework for complex hierarchical classification scenarios where interpretability is critically important.

## Installation

You can easily install the package via pip:

```bash
pip install sequential-shap-explainer
```

> **Note:** The current version is optimized for tree-based models (e.g. Random Forest) and uses `shap.TreeExplainer` under the hood.

## Usage Example

```python
from sequential_shap import SequentialSHAP

# 1. Initialize the explainer with your trained model, training data, and the semantic order of your classes
# Example: class_order=['Low', 'Medium', 'High']
explainer = SequentialSHAP(
    model=your_trained_model,
    X_train=X_train,
    y_train=y_train,
    class_order=['Low_Level', 'Medium_Level', 'High_Level'] 
)

# 2. Explain a specific instance by its index in the dataset
results_df = explainer.explain_by_index(index=42)

# 3. View the summarized DataFrame
print(results_df)

# 4. Plot the Sequential SHAP chart
explainer.plot()
```

## Technical Features
- **Automatic String Label Conversion:** Safely maps string class labels to internal integers.
- **Custom DataFrame Output:** Embeds the primary prediction into the DataFrame string representation for cleaner console readability.
- **Intuitive Visualization:** Automatically sorts and colors features based on their true directional impact on sequential classes.
