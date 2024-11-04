from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import os
import uuid
import sqlite3
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
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
app.secret_key = 'your_secret_key'  # Change this to a strong secret key
UPLOAD_FOLDER = 'dataset'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Database setup
DATABASE = 'users.db'

def init_db():
    """Initialize the user database if it doesn't already exist."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fullname TEXT NOT NULL,
                        email TEXT NOT NULL UNIQUE,
                        username TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()  # Initialize the database

# Routes
@app.route('/')
def intro():
    return render_template('intro.html')

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login route to authenticate users."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[4], password):
            session['user_id'] = user[0]
            session['username'] = user[3]
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration route to create a new user."""
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        try:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (fullname, email, username, password) VALUES (?, ?, ?, ?)",
                           (fullname, email, username, hashed_password))
            conn.commit()
            conn.close()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout route to clear the user session."""
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('intro'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Route to handle data upload and analysis."""
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

        # Basic Data Cleaning
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        
        if 'Member_number' not in data.columns or 'item' not in data.columns:
            return render_template('index.html', error="Dataset must contain 'Member_number' and 'item' columns.")
        
        data.rename(columns={'Member_number': 'TransactionID', 'item': 'Item'}, inplace=True)

        # Preprocess transactions for association rule mining
        transactions = data.groupby(['TransactionID', 'Item'])['Item'].count().unstack().fillna(0)
        transactions = transactions.applymap(lambda x: 1 if x > 0 else 0)

        # Apply Apriori Algorithm
        frequent_itemsets = apriori(transactions, min_support=0.02, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        top_rules = rules.sort_values(by='lift', ascending=False).head(10)

        # Visualization generation
        generate_visualizations(data, top_rules, rules)

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

        # Model Training and Evaluation
        model_results = train_and_evaluate_models(X_train, y_train, X_test, y_test)

        return render_template(
            'index.html',
            rules=top_rules.to_dict(orient='records'),
            all_rules=rules.to_dict(orient='records'),
            model_results=model_results
        )

    except Exception as e:
        return render_template('index.html', error=f"Data processing error: {e}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def generate_visualizations(data, top_rules, rules):
    """Generates and saves all visualizations needed for analysis."""
    # Visualization 1: Top 10 Most Frequent Items
    item_counts = data['Item'].value_counts().head(10)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=item_counts.values, y=item_counts.index, palette='viridis')
    plt.title('Top 10 Most Frequent Items')
    plt.xlabel('Frequency')
    plt.ylabel('Item')
    plt.xticks(rotation=45)
    plt.tight_layout()
    top_items_path = os.path.join(app.config['STATIC_FOLDER'], 'top_items.png')
    plt.savefig(top_items_path)
    plt.close()

    # Visualization 2: Support vs Confidence of Top Rules
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        x=top_rules['support'], 
        y=top_rules['confidence'], 
        size=top_rules['lift'], 
        sizes=(50, 300), 
        hue=top_rules['lift'], 
        palette='coolwarm', 
        legend=False
    )
    plt.title('Support vs Confidence (Top Association Rules)')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    support_confidence_path = os.path.join(app.config['STATIC_FOLDER'], 'support_confidence.png')
    plt.savefig(support_confidence_path)
    plt.close()

    # Visualization 3: Lift Distribution of Rules
    plt.figure(figsize=(6, 4))
    sns.histplot(rules['lift'], bins=20, kde=True, color='purple')
    plt.title('Lift Distribution of Association Rules')
    plt.xlabel('Lift')
    plt.ylabel('Frequency')
    plt.tight_layout()
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
    plt.figure(figsize=(8, 6))
    sns.heatmap(item_pair_counts, cmap="YlGnBu", linewidths=0.5, cbar_kws={'shrink': 0.5})
    plt.title('Heatmap of Frequent Item Pairs')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    heatmap_path = os.path.join(app.config['STATIC_FOLDER'], 'heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()

    # Visualization 6: Support and Confidence Distribution
    plt.figure(figsize=(6, 4))
    sns.kdeplot(rules['support'], shade=True, color="red", label="Support")
    sns.kdeplot(rules['confidence'], shade=True, color="blue", label="Confidence")
    plt.title('Distribution of Support and Confidence')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    support_confidence_dist_path = os.path.join(app.config['STATIC_FOLDER'], 'support_confidence_dist.png')
    plt.savefig(support_confidence_dist_path)
    plt.close()

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """Trains multiple models and returns their accuracies."""
    models = {
        'Random Forest': RandomForestClassifier(),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC()
    }
    model_results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        model_results[f'{model_name} Accuracy'] = accuracy

    return model_results

if __name__ == '__main__':
    app.run(debug=True)
