from api2 import Flask, jsonify, request
from bs4 import BeautifulSoup
import requests
from findProduct import findMatch

app = Flask(__name__)


@app.route('/amazon/price-scrape', methods=['GET'])
def scrape_amazon():
    search_query = request.args.get('search')
    if not search_query:
        return jsonify({'error': 'No search query provided'}), 400

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US, en;q=0.5'
    }
    URL = f"https://www.amazon.in   /s?k={search_query}"
    webpage = requests.get(URL, headers=HEADERS)
    soup = BeautifulSoup(webpage.content, "html.parser")
    mainProduct = soup.find(
        "span", class_="rush-component s-latency-cf-section")
    products = mainProduct.find_all(
        "div", class_="puis-card-container s-card-container s-overflow-hidden aok-relative puis-include-content-margin puis puis-v2ef34xpz4g7e42bsw0vrg0v0gx s-latency-cf-section puis-card-border")

    data = []
    titles = []

    for product in products:
        title = product.find("span", attrs={
                             'class': 'a-size-medium a-color-base a-text-normal'}).text.strip()
        price = product.find(
            "span", attrs={'class': 'a-price-whole'}).text.strip()
        link = "https://www.amazon.in" + \
            product.find(
                "a", attrs={'class': 'a-link-normal s-no-outline'})['href']
        data.append({
            'title': title,
            'price': price,
            'link': link
        })
        titles.append(title)

    print(search_query)
    product_index = findMatch(search_query, titles)
    return jsonify(data[product_index[0]])

if __name__ == '__main__':
    app.run(debug=True)