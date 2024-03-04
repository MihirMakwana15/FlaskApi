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
    if(len(data['title'])==0):
        titles=[
                "Fire-Boltt Ninja Call Pro Plus 1.83\" Smart Watch with Bluetooth Calling, AI Voice Assistance, 100 Sports Modes IP67 Rating, 240 * 280 Pixel High Resolution",
                "Forbuz Monster Truck Toy for Kids, Amazing Toys, 360 De...",
                "Casual Shirt for Men|| Shirt for Men|| Men Stylish Shirt || Men Printed Shirt (Patta)",
                "Casual Shirt for Men|| Shirt for Men|| Men Stylish Shirt || Men Printed Shirt (Mistry)",
                "Maizic Smarthome Studio Classy Dynamic Microphone with ...",
                "Sounce Spiral Charger 12 Pcs Cable Protector Data Cable Saver Charging Cord Protective Cable Cover Headphone MacBook Laptop Earphone Cell Phone Set of 3",
                "Aarna Monster truck 360 Degree Stunt car with Rubber ty...",
                "TrueBucks Cactus Talking Toy Dancing Cactus Repeats Wha...",
                "Ipad air m1",
                "Apple iPad Air (5th gen) 64 GB ROM 10.9 Inch with Wi-Fi+5G (space Grey)",
                "Apple iPad Air (5th Generation): with M1 chip, 27.69 cm (10.9″) Liquid Retina Display, 64GB, Wi-Fi 6, 12MP front/12MP Back Camera, Touch ID, All-Day Battery Life – Space Gray",
                "Apple iPad Air (5th Generation): with M1 chip, 27.69 cm (10.9″) Liquid Retina Display, 256GB, Wi-Fi 6, 12MP front/12MP Back Camera, Touch ID, All-Day Battery Life – Space Gray",
                "Women Polyester Blend Solid Trousers",
                "wonderchef grinder"
            ]
    all_recommendations = []
    for title in titles:
        print(title)
        recommendation = recommended(title)
        all_recommendations.append(recommendation)
    return jsonify(recommendation=all_recommendations )
    
if __name__ == '__main__':
    app.run(debug=True)
