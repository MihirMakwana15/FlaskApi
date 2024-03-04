from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
from flask_cors import CORS
import json
import numpy as np
from openai import OpenAI

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
    # Get top 3 similar products (excluding itself)
    indices = sim_scores.argsort()[0][-4:-1][::-1]
    recommendations = productdata.iloc[indices]
    recommendations['discount'] = recommendations['discount'].replace(
        np.NaN, None)
    recommendations['original_price'] = recommendations['original_price'].replace(
        np.NaN, None)
    return recommendations.to_dict(orient='records')


client = OpenAI()

instructionMessage = {
    "role": "system",
    "content": "You are working as a recommendation system. I will give you the title of a product, and you have to recommend 10 complimentary products based on that. Do not give brand name along with response, only give general product name or category. No need for any kind of explanation and instructions. Give only a string of products separated by comma. Do not give number or new line character to product.",
}


def handle_post_request(title):
    res = []
    try:
        message = {"role": "system", "content": title}

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[instructionMessage, message]
        )
        res.append(completion.choices[0].message.content)
        return res

    except Exception as e:
        print("Error handling POST request:", e)
        return {"error": "Internal Server Error"}


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    titles = data['title']
    all_recommendations = []
    json_recommendations = []
    for title in titles:
        recommendation = recommended(title)
        all_recommendations.extend(recommendation)
    json_recommendations = []
    for rec in all_recommendations:
        try:
            json_recommendations.append(rec)
        except json.JSONDecodeError:
            json_recommendations.append('rec')

    return jsonify(json_recommendations)


@app.route("/complementary", methods=["POST"])
def recommendations():
    try:
        data = request.get_json()
        title = data["title"]
        result = handle_post_request(title)
        return jsonify(result), 200

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Bad Request"}), 400


if __name__ == '_main_':
    app.run(debug=True)
