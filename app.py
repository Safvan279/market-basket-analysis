from flask import Flask, render_template, request, url_for
import pandas as pd
import os
from mlxtend.frequent_patterns import apriori, association_rules
from werkzeug.utils import secure_filename
import uuid
import matplotlib.pyplot as plt
import seaborn as sns

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

        # Visualization 1: Top 10 Most Frequent Items
        item_counts = data['Item'].value_counts().head(10)
        plt.figure(figsize=(4, 3))  # Smaller size
        sns.barplot(x=item_counts.values, y=item_counts.index, palette='viridis')
        plt.title('Top 10 Most Frequent Items')
        plt.xlabel('Frequency')
        plt.ylabel('Item')
        top_items_path = os.path.join(app.config['STATIC_FOLDER'], 'top_items.png')
        plt.savefig(top_items_path)
        plt.close()

        # Visualization 2: Support vs Confidence of Top Rules
        plt.figure(figsize=(4, 3))  # Smaller size
        sns.scatterplot(x=top_rules['support'], y=top_rules['confidence'], size=top_rules['lift'], sizes=(50, 300), hue=top_rules['lift'], palette='coolwarm', legend=False)
        plt.title('Support vs Confidence (Top Association Rules)')
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        support_confidence_path = os.path.join(app.config['STATIC_FOLDER'], 'support_confidence.png')
        plt.savefig(support_confidence_path)
        plt.close()

        # Visualization 3: Lift Distribution of Rules
        plt.figure(figsize=(4, 3))  # Smaller size
        sns.histplot(rules['lift'], bins=20, kde=True, color='purple')
        plt.title('Lift Distribution of Association Rules')
        plt.xlabel('Lift')
        plt.ylabel('Frequency')
        lift_distribution_path = os.path.join(app.config['STATIC_FOLDER'], 'lift_distribution.png')
        plt.savefig(lift_distribution_path)
        plt.close()

        # Visualization 4: Top 10 Item Pairs by Lift
        top_lift_rules = top_rules[['antecedents', 'consequents', 'lift']].head(10)
        top_lift_rules['item_pair'] = top_lift_rules['antecedents'].astype(str) + ' -> ' + top_lift_rules['consequents'].astype(str)
        plt.figure(figsize=(4, 3))
        sns.barplot(x=top_lift_rules['lift'], y=top_lift_rules['item_pair'], palette='Blues_d')
        plt.title('Top 10 Item Pairs by Lift')
        plt.xlabel('Lift')
        plt.ylabel('Item Pair')
        top_lift_path = os.path.join(app.config['STATIC_FOLDER'], 'top_lift.png')
        plt.savefig(top_lift_path)
        plt.close()

        # Visualization 5: Heatmap of Frequent Item Pairs
        item_pair_counts = data.groupby(['TransactionID', 'Item']).size().unstack().fillna(0).corr()
        plt.figure(figsize=(4, 3))
        sns.heatmap(item_pair_counts, cmap="YlGnBu", linewidths=0.5)
        plt.title('Heatmap of Frequent Item Pairs')
        heatmap_path = os.path.join(app.config['STATIC_FOLDER'], 'heatmap.png')
        plt.savefig(heatmap_path)
        plt.close()

        # Visualization 6: Support and Confidence Distribution
        plt.figure(figsize=(4, 3))
        sns.kdeplot(rules['support'], shade=True, color="red", label="Support")
        sns.kdeplot(rules['confidence'], shade=True, color="blue", label="Confidence")
        plt.title('Distribution of Support and Confidence')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        support_confidence_dist_path = os.path.join(app.config['STATIC_FOLDER'], 'support_confidence_dist.png')
        plt.savefig(support_confidence_dist_path)
        plt.close()

        return render_template(
            'index.html', 
            rules=top_rules.to_dict(orient='records'),  
            model_results={'Random Forest': 0.85, 'Naive Bayes': 0.80, 'SVM': 0.82},  # Dummy values for example
            top_items_img='top_items.png',
            support_confidence_img='support_confidence.png',
            lift_distribution_img='lift_distribution.png',
            top_lift_img='top_lift.png',
            heatmap_img='heatmap.png',
            support_confidence_dist_img='support_confidence_dist.png',
            all_rules=rules.to_dict(orient='records')  # Pass all rules
        )
    
    except Exception as e:
        return render_template('index.html', error=f"Data processing error: {e}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
