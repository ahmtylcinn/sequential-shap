# Sequential SHAP

The SHAP (Shapley Additive exPlanations) method has established itself as a powerful tool in Explainable Artificial Intelligence (XAI). However, its classical approach is often restricted to revealing only positive and negative contributions for a single class, which can limit interpretabilty in **sequential multi-class classification problems**.

To overcome these constraints, **Sequential SHAP** proposes a modified alternative approach. This novel method elucidates the model's decision-making mechanism in a more granular manner by categorizing feature effects into distinct semantic groups.

## The Sequential Solution

**Sequential SHAP** addresses standard SHAP restrictions by breaking down the multi-class problem into two distinct binary classification stages based on a provided semantic class hierarchy (order). It categorizes features into three distinct groups:

- **UPPER (Red):** Features driving the prediction towards a higher class.
- **LOWER (Blue):** Features driving the prediction towards a lower class.
- **AMBIGUOUS (Gray):** Features demonstrating inconsistent or uninformative effects across class transitions.

The findings obtained from analyses—such as on an obesity dataset—indicate that this proposed approach offers more profound insights than standard SHAP analysis and enables precise interpretations regarding class transitions. In conclusion, this modification significantly enhances the transparency of the model, presenting a robust analytical framework for complex hierarchical classification scenarios where interpretability is critically important.

## Installation

This package intentionally does not auto-install external libraries to avoid altering your environment. Therefore, before installing this package, please ensure the following prerequisite libraries are installed:

```bash
pip install numpy pandas matplotlib scikit-learn shap
```

The library has been developed and tested with the following package versions:
- `pandas == 3.0.1`
- `numpy == 2.4.2`
- `matplotlib == 3.10.8`
- `scikit-learn == 1.8.0`
- `shap == 0.51.0`

Afterward, you can install the Sequential SHAP package via pip:

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

# 4. Plot the Sequential SHAP chart (Default behavior)
explainer.plot()

# 5. Plot BOTH the Classical SHAP Waterfall and Sequential SHAP side-by-side
# explainer.plot(show_classical=True)
```

## Technical Features
- **Automatic String Label Conversion:** Safely maps string class labels to internal integers.
- **Custom DataFrame Output:** Embeds the primary prediction into the DataFrame string representation for cleaner console readability.
- **Intuitive Visualization:** Automatically sorts and colors features based on their true directional impact on sequential classes.
