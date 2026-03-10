import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone

class CustomDataFrame(pd.DataFrame):
    _metadata = ['predicted_class']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return CustomDataFrame
        
    def __str__(self):
        # By bypassing super().__str__() and using to_string directly
        # we avoid the double-printing artifacts from pandas internal str vs repr logic
        if getattr(self, 'predicted_class', None) is not None:
            return f"Predicted: {self.predicted_class}\n{self.to_string()}"
        return self.to_string()
        
    def __repr__(self):
        # For Jupyter cells and REPL displays
        return self.__str__()

class SequentialSHAP:
    def __init__(self, model, X_train, y_train, class_order=None):
        self.base_model = model
        self.X_train = X_train
        
        # If class_order is not provided, infer it from y_train
        if class_order is None:
            inferred_order = sorted(list(np.unique(y_train)))
            # Check if any inferred class is a string
            if any(isinstance(c, str) for c in inferred_order):
                 print(f"Warning: Categorical classes detected and sorted alphabetically: {inferred_order}. "
                       f"If this is not the semantic order, please provide the 'class_order' parameter manually.")
            class_order = inferred_order
            
        # Internally map class labels to integers to prevent errors with some models
        self.original_class_order = list(class_order)
        # Create mapping from original label to integer index
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.original_class_order)}
        
        # Convert y_train to mapped integer values
        self.y_train = np.array([self.class_to_idx[y] if y in self.class_to_idx else y for y in y_train])
        
        self.class_order = list(range(len(self.original_class_order)))
        
        self.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(X_train.shape[1])]
        self.results_df = None
        
        # Standard SHAP attributes
        self.std_explainer = None
        self.std_observation = None
        self.std_pred_raw = None
        self.std_shap_values = None

    def _get_upper_shap_values(self, explainer, obs_array):
        """Extracts SHAP values only for the group labeled as 1 (Upper)"""
        sv = explainer.shap_values(obs_array)
        if isinstance(sv, list):
            return sv[1][0] if len(sv) > 1 else sv[0][0]
        elif isinstance(sv, np.ndarray):
            if len(sv.shape) == 3:
                return sv[0, :, 1]
            elif len(sv.shape) == 2:
                return sv[0]
        else:
            return sv.values[0]

    def explain_by_index(self, index, dataset=None):
        if dataset is None:
            dataset = self.X_train
            
        observation = dataset.iloc[index] if isinstance(dataset, pd.DataFrame) else dataset[index]
        obs_array = observation.values.reshape(1, -1) if isinstance(observation, pd.Series) else np.array(observation).reshape(1, -1)
        
        # 1- Find the predicted class for the observation at the given index
        pred_class_raw = self.base_model.predict(obs_array)[0]
        
        # If the model returns the original string label, map it to the internal integer index
        if pred_class_raw in self.class_to_idx:
            pred_class_idx = self.class_to_idx[pred_class_raw]
            pred_class_original = pred_class_raw
        else:
            pred_class_idx = self.class_order.index(pred_class_raw)
            pred_class_original = self.original_class_order[pred_class_idx]
            
        self.predicted_class = pred_class_original

        # --- BOUNDARY CHECK (STOP COMPUTATION) ---
        if pred_class_idx == 0 or pred_class_idx == len(self.class_order) - 1:
            print("\n⚠️ WARNING: This sample is in the lowest or highest class. Computation cancelled as there is no natural upper/lower level.")
            print("Please use the standard SHAP method for this sample.")
            self.results_df = None
            return None # Exit function immediately, ML model is not trained!

        # --- STAGE 1 ---
        # Rule: The class the index belongs to and below = 0, above = 1
        y_train_s1 = np.zeros(len(self.y_train), dtype=int)
        for i, actual_class_idx in enumerate(self.y_train):
            y_train_s1[i] = 0 if actual_class_idx <= pred_class_idx else 1
            
        model_s1 = clone(self.base_model).fit(self.X_train, y_train_s1)
        explainer1 = shap.TreeExplainer(model_s1)
        arr1 = self._get_upper_shap_values(explainer1, obs_array)

        # --- STAGE 2 ---
        # Rule: The class the given index belongs to and above = 1, below = 0
        y_train_s2 = np.zeros(len(self.y_train), dtype=int)
        for i, actual_class_idx in enumerate(self.y_train):
            y_train_s2[i] = 1 if actual_class_idx >= pred_class_idx else 0
            
        model_s2 = clone(self.base_model).fit(self.X_train, y_train_s2)
        explainer2 = shap.TreeExplainer(model_s2)
        arr2 = self._get_upper_shap_values(explainer2, obs_array)

        # Sum both results
        total_shap = arr1 + arr2
        
        # Visualization Rule:
        categories = []
        for val1, val2 in zip(arr1, arr2):
            if val1 > 0 and val2 > 0:
                categories.append('UPPER')
            elif val1 < 0 and val2 < 0:
                categories.append('LOWER')
            else:
                categories.append('AMBIGUOUS')
                
        self.results_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Stage1_Val': arr1,
            'Stage2_Val': arr2,
            'Total_SHAP': total_shap,
            'Category': categories
        })
        
        # --- STANDARD SHAP WATERFALL PREPARATION ---
        try:
            self.std_explainer = shap.TreeExplainer(self.base_model)
            self.std_observation = observation
            self.std_pred_raw = pred_class_raw
            self.std_shap_values = self.std_explainer.shap_values(observation)
        except Exception as e:
            self.std_explainer = None
            
        # Wrap the returned output in CustomDataFrame to attach prediction text
        ret_df = CustomDataFrame(self.results_df[['Feature', 'Category']])
        ret_df.predicted_class = self.predicted_class
        return ret_df

    def plot(self, show_classical=False):
        if self.results_df is None:
            print("Warning: No results to plot. The examined sample might be in a boundary class or computation has not been performed yet.")
            return
            
        # Visualization order and colors
        # 0: AMBIGUOUS (bottom), 1: LOWER (middle), 2: UPPER (top)
        order_map = {'AMBIGUOUS': 0, 'LOWER': 1, 'UPPER': 2}
        color_map = {'AMBIGUOUS': 'gray', 'LOWER': 'blue', 'UPPER': 'red'}
        
        plot_df = self.results_df.copy()
        plot_df['Sort_Order'] = plot_df['Category'].map(order_map)
        
        # Inside each category, sort by absolute value so larger bars are at the top
        plot_df['Abs_SHAP'] = plot_df['Total_SHAP'].abs()
        plot_df = plot_df.sort_values(by=['Sort_Order', 'Abs_SHAP']).reset_index(drop=True)
        
        colors = plot_df['Category'].map(color_map).tolist()
        
        # Invert the direction specifically for AMBIGUOUS category
        plot_df['Plot_Val'] = np.where(plot_df['Category'] == 'AMBIGUOUS', 
                                       -plot_df['Total_SHAP'], 
                                       plot_df['Total_SHAP'])
        
        if show_classical:
            fig = plt.figure(figsize=(18, 7))
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(1, 2, width_ratios=[1, 1.5])
            
            # --- CLASSICAL SHAP WATERFALL PLOT ---
            ax1 = plt.subplot(gs[0])
            if getattr(self, 'std_explainer', None) is not None:
                try:
                    # Find the internal index of the predicted class to extract the correct SHAP values
                    if hasattr(self.base_model, 'classes_'):
                        model_classes = list(self.base_model.classes_)
                        pred_idx = model_classes.index(self.std_pred_raw) if self.std_pred_raw in model_classes else self.std_pred_raw
                    else:
                        pred_idx = self.std_pred_raw
                    
                    sv = self.std_shap_values
                    if sv is None:
                        raise ValueError("Standard SHAP values were not computed.")
                    
                    # Extract the 1D values array for the predicted class
                    if isinstance(sv, list):
                        vals = sv[pred_idx]
                    elif isinstance(sv, np.ndarray):
                        if len(sv.shape) == 2:
                            vals = sv[:, pred_idx]
                        elif len(sv.shape) == 3:
                            vals = sv[0, :, pred_idx]
                        else:
                            vals = sv
                    else:
                        vals = sv
                    
                    # Extract the baseline value for the predicted class
                    if self.std_explainer is None:
                        raise ValueError("Standard SHAP explainer is missing.")
                    ev = self.std_explainer.expected_value
                    
                    # Some versions/models return a single expected_value, others return a list
                    if isinstance(ev, (list, np.ndarray)) and isinstance(pred_idx, int) and pred_idx < len(ev):
                        base_val = ev[pred_idx]
                    else:
                        base_val = ev
                        
                    explanation = shap.Explanation(
                        values=vals, 
                        base_values=base_val, 
                        data=self.std_observation, 
                        feature_names=self.feature_names
                    )
                    
                    # shap.plots.waterfall handles plt.show() internally unless show=False is passed
                    shap.plots.waterfall(explanation, max_display=14, show=False)
                    plt.title("Classical SHAP Waterfall Plot", pad=15, fontsize=12)
                    plt.xticks(fontsize=10)
                    plt.yticks(fontsize=10)
                    
                except Exception as e:
                    ax1.text(0.5, 0.5, f"Warning: Could not generate standard SHAP Waterfall plot. Details:\n{e}", ha='center', va='center', wrap=True)
                    ax1.axis('off')

            # --- SEQUENTIAL SHAP PLOT (RIGHT SIDE) ---
            ax2 = plt.subplot(gs[1])
            ax2.barh(plot_df['Feature'], plot_df['Plot_Val'], color=colors)
            ax2.axvline(0, color="black", linewidth=0.8)
            
            ax2.set_xlabel("Magnitude of the Shapley Value", fontsize=10)
            ax2.set_ylabel("Features", fontsize=10)
            ax2.set_title("Sequential SHAP Plot", pad=15, fontsize=12)
            ax2.tick_params(axis='both', which='major', labelsize=10)
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', label='UPPER (Red)'),
                Patch(facecolor='blue', label='LOWER (Blue)'),
                Patch(facecolor='gray', label='AMBIGUOUS (Gray)')
            ]
            ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)
            
            # Adjust spacing so words don't overlap between the two plots
            plt.subplots_adjust(wspace=1.2, bottom=0.15, right=0.95, left=0.08)
            plt.show()
            
        else:
            # --- DEFAULT: ONLY PRINT SEQUENTIAL SHAP PLOT ---
            plt.figure(figsize=(10, 8))
            plt.barh(plot_df['Feature'], plot_df['Plot_Val'], color=colors)
            plt.axvline(0, color="black", linewidth=0.8)
            
            plt.xlabel("Magnitude of the Shapley Value")
            plt.ylabel("Features")
            plt.title("Sequential SHAP Plot")
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', label='UPPER (Red)'),
                Patch(facecolor='blue', label='LOWER (Blue)'),
                Patch(facecolor='gray', label='AMBIGUOUS (Gray)')
            ]
            plt.legend(handles=legend_elements, loc='lower right')
            
            plt.tight_layout()
            plt.show()