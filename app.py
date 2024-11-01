from flask import Flask, render_template, request, url_for
import pandas as pd
import os
from mlxtend.frequent_patterns import apriori, association_rules
from werkzeug.utils import secure_filename
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

        # Preprocessing
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        
        if 'Member_number' not in data.columns or 'item' not in data.columns:
            return render_template('index.html', error="Dataset must contain 'Member_number' and 'item' columns.")
        
        data.rename(columns={'Member_number': 'TransactionID', 'item': 'Item'}, inplace=True)

        transactions = data.groupby(['TransactionID', 'Item'])['Item'].count().unstack().fillna(0)
        transactions = transactions.applymap(lambda x: 1 if x > 0 else 0)

        # Apply Apriori
        frequent_itemsets = apriori(transactions, min_support=0.02, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        top_rules = rules.sort_values(by='lift', ascending=False).head(10)

        # Machine Learning Model
        data['label'] = (data['Item'].str.len() % 2)
        X = transactions
        y = data['label'].iloc[:len(transactions)]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

        nb_model = MultinomialNB()
        nb_model.fit(X_train, y_train)
        nb_accuracy = accuracy_score(y_test, nb_model.predict(X_test))

        svm_model = SVC()
        svm_model.fit(X_train, y_train)
        svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))

        # Visualization 1: Top 10 Most Frequent Items
        item_counts = data['Item'].value_counts().head(10)
        plt.figure(figsize=(6, 4))
        sns.barplot(x=item_counts.values, y=item_counts.index, palette='viridis')
        plt.title('Top 10 Most Frequent Items')
        top_items_path = os.path.join(app.config['STATIC_FOLDER'], 'top_items.png')
        plt.savefig(top_items_path)
        plt.close()

        # Visualization 2: Support vs Confidence of Top Rules
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=top_rules['support'], y=top_rules['confidence'], size=top_rules['lift'], sizes=(50, 300), hue=top_rules['lift'], palette='coolwarm', legend=False)
        plt.title('Support vs Confidence (Top Association Rules)')
        support_confidence_path = os.path.join(app.config['STATIC_FOLDER'], 'support_confidence.png')
        plt.savefig(support_confidence_path)
        plt.close()

        # Visualization 3: Lift Distribution
        plt.figure(figsize=(6, 4))
        sns.histplot(rules['lift'], bins=20, kde=True, color='skyblue')
        plt.title('Lift Distribution of Association Rules')
        lift_distribution_path = os.path.join(app.config['STATIC_FOLDER'], 'lift_distribution.png')
        plt.savefig(lift_distribution_path)
        plt.close()

        # Visualization 4: Top 10 Item Pairs by Lift
        top_lift = top_rules.head(10).sort_values(by='lift', ascending=False)
        plt.figure(figsize=(6, 4))
        sns.barplot(x=top_lift['lift'], y=top_lift['antecedents'].apply(lambda x: ', '.join(list(x))))
        plt.title('Top 10 Item Pairs by Lift')
        top_lift_path = os.path.join(app.config['STATIC_FOLDER'], 'top_lift.png')
        plt.savefig(top_lift_path)
        plt.close()

        # Visualization 5: Heatmap of Frequent Item Pairs
        plt.figure(figsize=(10, 8))
        sns.heatmap(transactions.corr(), cmap='coolwarm', center=0)
        plt.title('Heatmap of Frequent Item Pairs')
        heatmap_path = os.path.join(app.config['STATIC_FOLDER'], 'heatmap.png')
        plt.savefig(heatmap_path)
        plt.close()

        # Visualization 6: Support and Confidence Distribution
        plt.figure(figsize=(6, 4))
        sns.histplot(rules['support'], color='orange', label='Support', kde=True)
        sns.histplot(rules['confidence'], color='purple', label='Confidence', kde=True)
        plt.title('Support and Confidence Distribution')
        plt.legend()
        support_confidence_dist_path = os.path.join(app.config['STATIC_FOLDER'], 'support_confidence_dist.png')
        plt.savefig(support_confidence_dist_path)
        plt.close()

        return render_template(
            'index.html',
            rules=top_rules.to_dict(orient='records'),
            all_rules=rules.to_dict(orient='records'),
            model_results={
                'Random Forest Accuracy': rf_accuracy,
                'Naive Bayes Accuracy': nb_accuracy,
                'SVM Accuracy': svm_accuracy
            },
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
