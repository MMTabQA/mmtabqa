import requests
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import aiohttp
import asyncio
import time  # Import the time module for the blocking sleep
import json
from datetime import datetime

async def get_page(session, url, headers):
    for _ in range(5):
        try : 
            async with session.get(url, headers=headers) as r:
                a = await r.text()
                a = json.loads(a)
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
    links_to_new_links = {}

    target_date = datetime(year=2020, month=5, day=5)
    for result in tqdm(results):
        url = result[0]
        DATA = result[1]
        PAGES = DATA["query"]["pages"]
        closest_revision = None
        if len(PAGES) == 0:
            DATA = requests.get(url).json()
        for page_key, page_dict in PAGES.items():
            # print(page_dict)
            if 'revisions' not in page_dict:
                print("No revision found for ", url)
                links_to_new_links[url] =None
                continue
            revisions = (page_dict['revisions'])
            # pprint(revisions)
            closest_revision = None
            min_time_difference = float("inf")

            for revision in revisions:
                revision_date = revision["timestamp"]
                timestamp_obj = datetime.strptime(revision_date, '%Y-%m-%dT%H:%M:%SZ')
                time_difference = abs((target_date - timestamp_obj).days)

                if time_difference < min_time_difference:
                    min_time_difference = time_difference
                    closest_revision = revision

        if closest_revision is None:
            print( "No revision found for ", url)
            links_to_new_links[url] =None 
        else : 
            # print(closest_revision["revid"])
            links_to_new_links[url] = closest_revision["revid"]

            
    return links_to_new_links

if __name__ == '__main__':
    input_asyncio = []
    with open('/home2/jainit/FeTAQA_MM/new_outputs/asyncio_inputs/revision_ids_queries.json','r') as pkl:
        input_asyncio = list(set(json.load(pkl).values()))
    try : 
        with open("/home2/jainit/FeTAQA_MM/new_outputs/asyncio_outputs/revision_ids.json",'r') as pkl:
            all_results =json.load(pkl)
        
    except :
        all_results = {}
    # all_results = {}
    all_results = {k: v for k, v in all_results.items() if v is not None}
    input_asyncio = list(set(input_asyncio) - set(all_results.keys()))

    print("Length of input_asyncio: ", len(input_asyncio))
    # headers = {'User-Agent': 'jimmy.detor/1.0 (jimmy.detor@stanford.edu) nlp lab'}
    
    for i in tqdm(range(0, len(input_asyncio), 100)):
        headers = {f'User-Agent': 'jimmy.detor{i}/1.0 (jimmy.detor@stanford.edu) nlp lab'}
        results = asyncio.run(main(input_asyncio[i:i+100], headers))
        if results is None:
            continue
        results = parse(results)
        all_results.update(results)
        if i%300 ==0:
            with open("/home2/jainit/FeTAQA_MM/new_outputs/asyncio_outputs/revision_ids.json", "w") as pkl:
                json.dump(all_results, pkl)
            time.sleep(10)  # Add a delay of 1 second after updating all_results

    with open("/home2/jainit/FeTAQA_MM/new_outputs/asyncio_outputs/revision_ids.json", "w") as pkl:
        json.dump(all_results, pkl)

