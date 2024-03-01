from flask import Flask, jsonify, request
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle

app = Flask(__name__)


product_dict = pickle.load(open('products.pkl', 'rb'))
productdata = pd.DataFrame(product_dict)
tfid = TfidfVectorizer()
vectors = tfid.fit_transform(productdata['title'])
cosine_sim = cosine_similarity(vectors, vectors)

similarity = pickle.load(open('similarity.pkl', 'rb'))

def recommended(select_product_name):
    best_match = max(productdata['title'], key=lambda x: fuzz.partial_ratio(x.lower(), select_product_name.lower()))
    idx = productdata[productdata['title'] == best_match].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_similar_products = sim_scores[1:4]
    return productdata.iloc[(x[0] for x in top_similar_products)].to_dict(orient='records')


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    titles = data['title']
    all_recommendations = []
    for title in titles:
        print(title)
        recommendation = recommended(title)
        all_recommendations.append(recommendation)
    return jsonify(recommendation=all_recommendations )
    
if __name__ == '__main__':
    app.run(debug=True)
