import time
import random
import requests
import streamlit as st
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import re

API_KEY = "93eeba7d0efc2277582854a6897619b3" 

def scrape_amazon_products(search_query, max_products=100):
    print(f"Searching Amazon for: '{search_query}'")
    
    search_url = f"https://www.amazon.com/s?k={search_query.replace(' ', '+')}"
    api_url = f"http://api.scraperapi.com?api_key={API_KEY}&url={search_url}"
    
    response = requests.get(api_url)
    
    if response.status_code != 200:
        st.error(f"Failed to retrieve Amazon search results. Status code: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    products = []
    
    for idx, item in enumerate(soup.select("[data-component-type='s-search-result']")):
        try:
            asin = item.get('data-asin')
            if not asin:
                continue
                
            product_link = f"https://www.amazon.com/dp/{asin}"
            
            img_elements = item.select(".s-image") or item.select("img.s-image")
            image_urls = [img.get("src") for img in img_elements if img.get("src")]
            
            title_element = item.select_one("h2 a span") or item.select_one("h2 span") or item.select_one(".a-size-base-plus")
            title = title_element.text.strip() if title_element else f"Product {asin}"
            
            price_spans = [
                item.select_one(".a-price .a-offscreen"),
                item.select_one(".a-price span.a-offscreen"),
                item.select_one("span.a-price span.a-offscreen"),
                item.select_one("span[data-a-color='price'] span.a-offscreen")
            ]
            
            price = "Price not available"
            for price_span in price_spans:
                if price_span:
                    price = price_span.text.strip()
                    break
            
            product = {
                "image_urls": image_urls,
                "price": price,
                "link": product_link,
                "title": title,
                "source": "Amazon"
            }
            
            products.append(product)
            
            if len(products) >= max_products:
                break
            
        except Exception as e:
            print(f"Error extracting product {idx+1}: {str(e)}")
            continue
        
        time.sleep(random.uniform(0.5, 1.5))
       
    return products



def scrape_ebay_products(search_query, max_products=100):
    print(f"Searching eBay for: '{search_query}'")
    
    search_url = f"https://www.ebay.com/sch/i.html?_nkw={search_query.replace(' ', '+')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    response = requests.get(search_url, headers=headers)
    
    print(f"eBay Response Status Code: {response.status_code}")
    
    if response.status_code != 200:
        st.error(f"Failed to retrieve eBay search results. Status code: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Debug: Save the HTML to a file for inspection
    with open("ebay.html", "w", encoding="utf-8") as f:
        f.write(soup.prettify())
    
    print("eBay HTML saved to 'ebay.html' for inspection.")
    
    products = []
    
    # Debug: Print the number of product containers found
    product_containers = soup.select(".s-item")
    print(f"Found {len(product_containers)} product containers on eBay.")
    
    for idx, item in enumerate(product_containers):
        try:
            title_element = item.select_one(".s-item__title span")
            title = title_element.text.strip() if title_element else f"Product {idx+1}"
            print(f"eBay Product {idx+1} Title: {title}")
            
            price_element = item.select_one(".s-item__price")
            price = price_element.text.strip() if price_element else "Price not available"
            print(f"eBay Product {idx+1} Price: {price}")
            
            link_element = item.select_one(".s-item__link")
            product_link = link_element.get("href") if link_element else "#"
            print(f"eBay Product {idx+1} Link: {product_link}")
            
            img_elements = item.select(".s-item__image-wrapper img")
            image_urls = [img.get("src") for img in img_elements if img.get("src")]
            print(f"eBay Product {idx+1} Image URLs: {image_urls}")
            
            product = {
                "image_urls": image_urls,
                "price": price,
                "link": product_link,
                "title": title,
                "source": "eBay"
            }
            
            products.append(product)
            
            if len(products) >= max_products:
                break
            
        except Exception as e:
            print(f"Error extracting eBay product {idx+1}: {str(e)}")
            continue
        
        time.sleep(random.uniform(0.5, 1.5))
       
    print(f"Found {len(products)} eBay products.")
    return products