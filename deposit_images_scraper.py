import os
import requests
from bs4 import BeautifulSoup

def scrape_deposit_images(query, folder_path, num_images):
    url = f"https://www.pexels.com/search/yellow/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    count = 0
    for i, img in enumerate(img_tags):
        img_url = img.get('src')
        if img_url and 'http' in img_url:
            img_data = requests.get(img_url).content
            with open(os.path.join(folder_path, f'{query}_bing_{i}.jpg'), 'wb') as handler:
                handler.write(img_data)
                count += 1
                if count >= num_images:
                    break

color_types = ["red", "orange", "yellow", "green", "blue", "purple", "black", "white"]

for color in color_types:
    scrape_deposit_images(color, f'./color_images/{color}', 100)
