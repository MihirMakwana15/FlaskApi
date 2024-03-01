from flask import Flask, jsonify, request
from bs4 import BeautifulSoup
import requests
from findProduct import findMatch
import streamlit as st
import pandas as pd
import pickle 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from PIL import Image

product_dict=pickle.load(open('movies.pkl','rb'))
productdata=pd.DataFrame(product_dict)
similarity=pickle.load(open('similarity.pkl','rb'))



cv=CountVectorizer()
vectors = cv.fit_transform(productdata['title'])
cosine_sim = cosine_similarity(vectors,vectors)
def recommended(select_prodcut_name,selected_product_discounted_price,selected_product_image,selected_product_link,selected_original,selected_product_discount):
    best_match = max(productdata['title'], key=lambda x: fuzz.partial_ratio(x.lower(), select_product_name.lower()))    
    idx = productdata[productdata['title'] == best_match].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_similar_products = sim_scores[1:4]
    return productdata['title'].iloc[[x[0] for x in top_similar_products]],productdata['discount_price'].iloc[[x[0] for x in top_similar_products]],productdata['image'].iloc[[x[0] for x in top_similar_products]],productdata['link'].iloc[[x[0] for x in top_similar_products]],productdata['original_price'].iloc[[x[0] for x in top_similar_products]],productdata['discount'].iloc[[x[0] for x in top_similar_products]]

st.title("Recommendation")

product_list=productdata['title'].values
product_discounted_price=productdata['discount_price'].values
product_image=productdata['image'].values
product_original =productdata['original_price'].values
product_link=productdata['link'].values
product_discount=productdata['discount'].values

select_product_name= st.selectbox(
    "Type or select the product",
    product_list
)

if st.button('Recommend'):
    recommendation=recommended(select_product_name,product_discounted_price,product_image,product_link,product_original,product_discount)
    for i in recommendation:
       st.write(i)
       