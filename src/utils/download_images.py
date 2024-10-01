import pickle
import requests
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import aiohttp
import asyncio
import time  # Import the time module for the blocking sleep
import json
import random
import warnings
import os 
from PIL import Image
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933130000 
import io
import re
from urllib.parse import unquote

image_id_to_paths = {}
# with open('./results_image_id_url.json','r') as p:
#     all_links= json.load(p)

# links_all = {v:k for k,v in all_links.items()}
DATASET_PATH = "/scratch/jainit_ftq/ftq_images/"

async def get_page(session, url, headers, max_retries=10, retry_delay=60):
    for _ in range(max_retries):
        try:
            async with session.get(url, headers=headers) as r:
                a = await r.read()
                img = Image.open(io.BytesIO(a))
                img = img.resize((512, 512), Image.BICUBIC)
                new_url = re.sub(r'\?[^?]*$', '', url)
                file_extension = new_url.split(".")[-1].lower()
                image_name = new_url.split("/")[-1]
                last_occurrence_index = image_name.rfind('.')
                before_last_occurrence = unquote(image_name[:last_occurrence_index]).replace(" ", "_")
                before_last_occurrence= before_last_occurrence[:200]

                # Save the compressed image
                path = os.path.join(DATASET_PATH, f"{before_last_occurrence}.{file_extension}")
                img.save(path, optimize=True, quality=90)
                image_id_to_paths[url]= path 
                print(path)
                return [url, a]
        except Exception as e:
            print(f"An error occurred: {e}",url)
            await asyncio.sleep(retry_delay)
    print(f"Max retries reached for URL: {url}")
    return [url, None]

async def get_all(session, urls, semaphore, headers, delay=0, max_retries=10, retry_delay=60):
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


if __name__ == '__main__':
    links_with_no_entity = []
    # SET THIS TO THE PATH OF THE JSON FILE CONTAINING THE LINKS NAMED 'link_to_single_image.json'
    with open('/home2/jainit/FeTAQA_MM/new_outputs/link_to_single_image.json', 'r') as pkl:
        links_with_no_entity = list(set(list(json.load(pkl).values())))
    new_links_with_no_entity = []
    print(len(links_with_no_entity))

    for i in links_with_no_entity:
        if i.startswith('https://maps'):
            continue
        if i.split('.')[-1].lower() != 'mid' :
            new_links_with_no_entity.append(i)
    print(len(new_links_with_no_entity))
    print("I am here ")
    # headers = {'User-Agent': 'jainitjainit/1.0 (jainit.bafna@research.iiit.ac.in) research purpose'}
    all_results = {}
    for i in tqdm(range(0, len(new_links_with_no_entity), 200)):
        headers = {f'User-Agent': 'walter{i}lewin{i}/1.0 (walter.lewin@standford.edu) research purpose'}
        results = asyncio.run(main(new_links_with_no_entity[i:i+200], headers))
        # pp = get_image_infobox(results)
        # # print(results)
        # if pp is None:
        #     # print(results)
        #     warnings.warn("No results found")
        #     continue
        # all_results.update(pp)
        if i%200 ==0:
            # with open("results_link_to_reference.json", "w") as pkl:
            #     json.dump(all_results, pkl)
            time.sleep(20)  # Add a delay of 1 second after updating all_results
    with open("/home2/jainit/FeTAQA_MM/new_outputs/download_paths.json", "w") as pkl:
                json.dump(image_id_to_paths, pkl)
