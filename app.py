from flask import Flask, render_template, request, url_for
import pandas as pd
import os
from mlxtend.frequent_patterns import apriori, association_rules
from werkzeug.utils import secure_filename
import uuid
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the non-GUI backend
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

app = Flask(__name__)
UPLOAD_FOLDER = 'dataset'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return render_template('index.html', error="No file selected.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="Please select a file.")

    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    try:
        file.save(file_path)
        data = pd.read_csv(file_path)

        # Basic Cleaning
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        
        if 'Member_number' not in data.columns or 'item' not in data.columns:
            return render_template('index.html', error="Dataset must contain 'Member_number' and 'item' columns.")
        
        data.rename(columns={'Member_number': 'TransactionID', 'item': 'Item'}, inplace=True)

        # Preprocess transactions
        transactions = data.groupby(['TransactionID', 'Item'])['Item'].count().unstack().fillna(0)
        transactions = transactions.applymap(lambda x: 1 if x > 0 else 0)

        # Apply Apriori for Association Rules
        frequent_itemsets = apriori(transactions, min_support=0.02, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        top_rules = rules.sort_values(by='lift', ascending=False).head(10)

        # Prepare data for ML models
        le = LabelEncoder()
        data['Item_encoded'] = le.fit_transform(data['Item'])
        transactions = transactions.astype(int)
        
        # Create artificial labels
        data['label'] = (data['Item'].str.len() % 2)
        
        y = data.groupby('TransactionID')['label'].first().reindex(transactions.index)

        # Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(transactions)

        # Handle Class Imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Dimensionality Reduction with PCA
        pca = PCA(n_components=0.95)
        X_reduced = pca.fit_transform(X_resampled)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_resampled, test_size=0.2, random_state=42)

        # Random Forest Classifier
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_predictions)
        rf_precision = precision_score(y_test, rf_predictions)
        rf_recall = recall_score(y_test, rf_predictions)
        rf_f1 = f1_score(y_test, rf_predictions)
        print("Random Forest Model:")
        print(f" - Accuracy: {rf_accuracy}")
        print(f" - Precision: {rf_precision}")
        print(f" - Recall: {rf_recall}")
        print(f" - F1 Score: {rf_f1}")

        # Gaussian Naive Bayes Classifier
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)
        nb_predictions = nb_model.predict(X_test)
        nb_accuracy = accuracy_score(y_test, nb_predictions)
        nb_precision = precision_score(y_test, nb_predictions)
        nb_recall = recall_score(y_test, nb_predictions)
        nb_f1 = f1_score(y_test, nb_predictions)
        print("Gaussian Naive Bayes Model:")
        print(f" - Accuracy: {nb_accuracy}")
        print(f" - Precision: {nb_precision}")
        print(f" - Recall: {nb_recall}")
        print(f" - F1 Score: {nb_f1}")

        # Support Vector Classifier
        svm_model = SVC()
        svm_model.fit(X_train, y_train)
        svm_predictions = svm_model.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_predictions)
        svm_precision = precision_score(y_test, svm_predictions)
        svm_recall = recall_score(y_test, svm_predictions)
        svm_f1 = f1_score(y_test, svm_predictions)
        print("Support Vector Machine Model:")
        print(f" - Accuracy: {svm_accuracy}")
        print(f" - Precision: {svm_precision}")
        print(f" - Recall: {svm_recall}")
        print(f" - F1 Score: {svm_f1}")

        # Visualization and rendering code remains the same
        # ... [Visualization code here, omitted for brevity] ...

        return render_template(
            'index.html',
            rules=top_rules.to_dict(orient='records'),
            all_rules=rules.to_dict(orient='records'),
            model_results={
                'Random Forest Accuracy': rf_accuracy,
                'Gaussian NB Accuracy': nb_accuracy,
                'SVM Accuracy': svm_accuracy
            },
            # Paths to saved images for charts
            top_items_img='top_items.png',
            support_confidence_img='support_confidence.png',
            lift_distribution_img='lift_distribution.png',
            top_lift_img='top_lift.png',
            heatmap_img='heatmap.png',
            support_confidence_dist_img='support_confidence_dist.png'
        )

    except Exception as e:
        return render_template('index.html', error=f"Data processing error: {e}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
