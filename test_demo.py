import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Bizim az önce güncellediğimiz dosyadan yeni sınıfımızı çağırıyoruz!
from sequential_shap.core import SequentialSHAP

print("1. Veri yükleniyor ve model eğitiliyor...")
# Load a simple dataset
data = load_iris()
X_train = pd.DataFrame(data.data, columns=data.feature_names)
y_train = data.target

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("2. SequentialSHAP başlatılıyor...")
# Sınıf isimlerini sıralı verelim
class_names = ['Setosa_Level_1', 'Versicolor_Level_2', 'Virginica_Level_3']

# Initialize the Explainer
explainer = SequentialSHAP(
    model=model,
    X_train=X_train,
    y_train=data.target_names[y_train], # Use string names
    class_order=class_names
)

print("3. Index 50 (Versicolor_Level_2) inceleniyor...")
# Explain a specific instance (e.g., index 50 is a Versicolor)
results_df = explainer.explain_by_index(index=50)

print(results_df)

print("\n4. Grafik çizdirme komutu (plot) çalıştırılıyor...")
# Plot the results! (This should show BOTH graphs now)
explainer.plot()

print("\nTest Başarıyla Tamamlandı!")
