import pickle
import requests
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import aiohttp
import asyncio
import time  # Import the time module for the blocking sleep
import json
import os 
from datetime import datetime
list_no_tables =[]
url_to_table_id = {}
table_id_global =[]

BASE_DIR = "/home/suyash/temp_testing_mmtabqa_upload" # IMPORTANT: Update this path.
Tables_path = os.path.join(BASE_DIR, "tables")

os.makedirs(Tables_path, exist_ok=True)

t1 = json.load(open(os.path.join(BASE_DIR, "asyncio_outputs", "revision_id_to_url.json"), "r"))  # revision_id_query_link -> reivision_id_wikipedia_page_link

async def get_page(session, url, headers):
    
    for _ in range(5):
        try :
            async with session.get(url, headers=headers) as r:
                a = await r.text()
                
                return [url, a]
        except Exception as e:
                print(f"An error occurred: {e}",url)
                await asyncio.sleep(20)
    print(f"Max retries 5 reached for URL: {url}")

async def get_all(session, urls, semaphore, headers, delay=0):
    tasks = []
    for i, url in tqdm(enumerate(urls)):
        async with semaphore:
            task = asyncio.create_task(get_page(session, url, headers))
            tasks.append(task)
    results = await asyncio.gather(*tasks)
    return results

async def main(urls, headers, delay=0):
    semaphore = asyncio.Semaphore(10)  # Set the maximum number of concurrent requests
    async with aiohttp.ClientSession() as session:
        data = await get_all(session, urls, semaphore, headers, delay)
        return data

def parse(results):
    linkstables = {}
    urls_to_tables = {} 
    for result in tqdm(results):
        if result is None or len(result) == 0:
            print("NOOOO")
            continue
        url_i = result[0]
        html = result[1]
        soup = BeautifulSoup(html, 'lxml')
        tables = soup.find_all('table')
        linkstables[url_i] = tables
        if len(tables) == 0:
            list_no_tables.append(url_i)
            print( "No tables found for ", url_i)
            # print(html)
            continue
    
    for i in t1:
        if t1[i] in linkstables:
            urls_to_tables[i] = linkstables[t1[i]]
        # else:
        #     print(i)
        #     print("Links not found")
        #find all nested tables 
    url_to_new_tables = {}
    for url, tables in urls_to_tables.items():
        new_tables = []
        for table in tables:
            n_tables = table.find_all('table')
            if len(n_tables)>0:
                new_tables.extend(n_tables)
            else:
                new_tables.append(table)
        url_to_new_tables[url] = new_tables
    for url , tables in tqdm(url_to_new_tables.items()):
        for table in tables: 
            table_id = len(table_id_global) + 200000
            url_to_table_id.setdefault(url, [])
            url_to_table_id[url].append(table_id)
            table_id_global.append(table_id)
            table_html = str(table)
            table_path = os.path.join(Tables_path, f"{table_id}.html")
            
            with open(table_path, 'w') as f:
                print(table_path)            
                print("YAY")
                f.write(table_html)
    print(len(url_to_table_id))
    


with open(os.path.join(BASE_DIR, "asyncio_outputs", "revision_id_to_revision_url.json"), 'r') as pkl:
    input_asyncio = list(set(json.load(pkl).values()))

# headers = {'User-Agent': 'walter.white/1.0 (walter.white@standford.edu) research purpose'}

print(len(input_asyncio))
for i in tqdm(range(0, len(input_asyncio), 100)):
    headers = {f'User-Agent': 'walter.white{i}/1.0 (walter.white@standford.edu) research purpose'}
    results = asyncio.run(main(input_asyncio[i:i+100], headers))
    print(results)
    if results is None or len(results) == 0:
        continue 
    parse(results)

table_id_path = os.path.join(BASE_DIR, "asyncio_outputs", "table_id.json")
print("Saving table_id to ", table_id_path)
with open(table_id_path, "w") as f:
    # Save the url_to_table_id dictionary to a JSON file
    print("Saving url_to_table_id to ", table_id_path)
    json.dump(url_to_table_id, f, indent=4)

