# %%
import json
link_to_entity = json.load(open("/home/suyash/Research/ADA_BACKUP/WTQ_MM/outputs/link_to_entity.json", "r"))

# %%
entity_to_link = {}
for link in link_to_entity:
    entity = link_to_entity[link]
    if entity in link_to_entity:
        print("Duplicate entity")
    entity_to_link[entity] = link

# %%
links = set(list(link_to_entity.keys()))
already_processed_links = set(list(json.load(open("/home/suyash/Research/ADA_BACKUP/WTQ_MM/outputs/pageview_stats.json", "r")).keys()))
links = list(links - already_processed_links)

# %%
import pickle

import requests
from bs4 import BeautifulSoup
import json
import os
from collections import defaultdict
import json
import requests
from PIL import Image
from io import BytesIO
import io
from tqdm import tqdm
import urllib.request
import wget
import re 
import aiohttp
import asyncio
import argparse
from urllib.parse import unquote
from time import sleep


def get_wikipedia_pageviews_request(url):
    """
    Get request for fetching monthly pageviews for a given Wikipedia page URL.

    Args:
    url (str): The URL of the Wikipedia page.

    Returns:
    str: A string for the API request URL.
    """
    # Extract the article title from the URL
    parts = url.split('/')
    if "wiki" in parts:
        article_title = parts[parts.index("wiki") + 1]
    else:
        raise ValueError("Invalid Wikipedia URL")

    # URL decode the article title
    article_title = unquote(article_title)

    # Construct the API request URL
    api_url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{article_title}/monthly/20181223/20231223"

    return api_url

#################### Code for asyncio page requests ####################//
# Getting all pages in parallelized fashion

async def get_page(session, url, headers, max_retries=10, retry_delay=60):
    asli_url = get_wikipedia_pageviews_request(url)
    for _ in range(max_retries):
        try:
            async with session.get(asli_url, headers=headers) as r:
                a = await r.json()
                return [url, a]
        except Exception as e:
            print(f"An error occurred: {e}")
            await asyncio.sleep(retry_delay)
    print(f"Max retries reached for URL: {url}")
    return [url, None]

async def get_all(session, urls, semaphore, headers, delay=0, max_retries=10, retry_delay=600):
    tasks = []
    for i, url in tqdm(enumerate(urls)):
        async with semaphore:
            task = asyncio.create_task(get_page(session, url, headers, max_retries, retry_delay))
            tasks.append(task)
            await asyncio.sleep(delay)  # Introduce delay between requests
    results = await asyncio.gather(*tasks)
    return results

async def main(urls, headers, delay=0, max_retries=3, retry_delay=2):
    semaphore = asyncio.Semaphore(10)  # Set the maximum number of concurrent requests
    async with aiohttp.ClientSession() as session:
        data = await get_all(session, urls, semaphore, headers, delay, max_retries, retry_delay)
        return data

########################################################################//                                                          


if __name__ == "__main__":
    # now we need to scrape the wikipedia pages for the links with no entity
    print()
    print("#############################################################")
    print("Scraping Wikipedia Pages")
    all_results = json.load(open("/home/suyash/Research/ADA_BACKUP/WTQ_MM/outputs/pageview_stats.json", "r"))
    for i in tqdm(range(0, len(links), 100)):
        headers = {f'User-Agent': 'walter.white/1.0 (walter.white@research{i}.standford.edu) research purpose'}
        # Fetching the page HTML into results
        results = asyncio.run(main(links[i:i+100], headers))
        
        # Getting the corresponding QID from the HTML
        all_results.update(results)
    json.dump(all_results, open("/home/suyash/Research/ADA_BACKUP/WTQ_MM/outputs/pageview_stats.json", "w"), indent=4)
        
