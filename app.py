from flask import Flask, render_template, request
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'dataset'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = 'static'

@app.route('/')
def index():
    return render_template('index.html')

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

        frequent_itemsets = apriori(transactions, min_support=0.02, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        top_rules = rules.sort_values(by='lift', ascending=False).head(10)

        item_counts = data['Item'].value_counts()
        
        if item_counts.empty:
            return render_template('index.html', error="No items found in the dataset.")
        
        most_frequent_item = item_counts.idxmax()
        y = transactions[most_frequent_item].values

        if len(set(y)) <= 1:
            return render_template('index.html', error=f"Not enough classes in '{most_frequent_item}' for training.")

        X = transactions.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model_results = {}

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        model_results['Random Forest'] = accuracy_score(y_test, rf.predict(X_test))

        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        model_results['Naive Bayes'] = accuracy_score(y_test, nb.predict(X_test))

        svm = SVC()
        svm.fit(X_train, y_train)
        model_results['SVM'] = accuracy_score(y_test, svm.predict(X_test))

        return render_template(
            'results.html', 
            rules=top_rules.to_dict(orient='records'),  
            model_results=model_results  
        )
    
    except Exception as e:
        return render_template('index.html', error=f"Data processing error: {e}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
