from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


product_dict = pickle.load(open('products.pkl', 'rb'))
productdata = pd.DataFrame(product_dict)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(productdata['title'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommended(select_product_name):
 

    select_product_tfidf = tfidf_vectorizer.transform([select_product_name])
    sim_scores = cosine_similarity(select_product_tfidf, tfidf_matrix)
    # Get indices of top similar products
    indices = sim_scores.argsort()[0][-4:-1][::-1]  # Get top 3 similar products (excluding itself)
    recommendations = productdata.iloc[indices]
    return recommendations.to_dict(orient='records')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json() 
    titles = data['title']
    all_recommendations = []
    #if(len(data['title'])==0):
    #    titles=productdata['title']
    for title in titles:   
        recommendation = recommended(title)
        all_recommendations.append(recommendation)
    return jsonify(recommendation=all_recommendations )
if __name__ == '__main__':
    app.run(debug=True)
