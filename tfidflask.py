from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle

app = Flask(__name__)

# Load product data
product_dict = pickle.load(open('products.pkl', 'rb'))
productdata = pd.DataFrame(product_dict)

# Preprocess titles using TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(productdata['title'])
# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommended(select_product_name):
    # Calculate TF-IDF vector for the selected product name
    select_product_tfidf = tfidf_vectorizer.transform([select_product_name])
    # Calculate cosine similarity of the selected product name with all products
    sim_scores = cosine_similarity(select_product_tfidf, tfidf_matrix)
    # Get indices of top similar products
    indices = sim_scores.argsort()[0][-4:-1][::-1]  # Get top 3 similar products (excluding itself)
    recommendations = productdata.iloc[indices]
    return recommendations.to_dict(orient='records')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    titles = data['title']
    all_recommendations = [recommended(title) for title in titles]
    return jsonify(recommendations=all_recommendations)
    
if __name__ == '__main__':
    app.run(debug=True)
