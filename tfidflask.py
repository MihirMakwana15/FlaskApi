from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle

app = Flask(__name__)


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
    if(len(data['title'])==0):
        titles=productdata['title']

        #titles=[
        #        "Fire-Boltt Ninja Call Pro Plus 1.83\" Smart Watch with Bluetooth Calling, AI Voice Assistance, 100 Sports Modes IP67 Rating, 240 * 280 Pixel High Resolution",
        #        "Forbuz Monster Truck Toy for Kids, Amazing Toys, 360 De...",
        #        "Casual Shirt for Men|| Shirt for Men|| Men Stylish Shirt || Men Printed Shirt (Patta)",
        #        "Casual Shirt for Men|| Shirt for Men|| Men Stylish Shirt || Men Printed Shirt (Mistry)",
        #        "Maizic Smarthome Studio Classy Dynamic Microphone with ...",
        #        "Sounce Spiral Charger 12 Pcs Cable Protector Data Cable Saver Charging Cord Protective Cable Cover Headphone MacBook Laptop Earphone Cell Phone Set of 3",
        #        "Aarna Monster truck 360 Degree Stunt car with Rubber ty...",
        #        "TrueBucks Cactus Talking Toy Dancing Cactus Repeats Wha...",
        #        "Ipad air m1",
        #        "Apple iPad Air (5th gen) 64 GB ROM 10.9 Inch with Wi-Fi+5G (space Grey)",
        #        "Apple iPad Air (5th Generation): with M1 chip, 27.69 cm (10.9″) Liquid Retina Display, 64GB, Wi-Fi 6, 12MP front/12MP Back Camera, Touch ID, All-Day Battery Life – Space Gray",
        #        "Apple iPad Air (5th Generation): with M1 chip, 27.69 cm (10.9″) Liquid Retina Display, 256GB, Wi-Fi 6, 12MP front/12MP Back Camera, Touch ID, All-Day Battery Life – Space Gray",
        #        "Women Polyester Blend Solid Trousers",
        #        "wonderchef grinder"
        #    ]
    all_recommendations = [recommended(title) for title in titles]
    return jsonify(recommendations=all_recommendations)
    
if __name__ == '__main__':
    app.run(debug=True)
