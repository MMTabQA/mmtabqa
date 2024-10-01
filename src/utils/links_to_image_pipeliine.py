# Instructions to execute:
# python3 links_to_image_pipeline --input <file_containing_list_of_links.json> --output <output_dir>
# Output files created:
    # INTERMEDIATE OUTPUTS:
        # link_with_no_entity.json : contains the Wikipedia page with no wikidata entity ID
        # link_to_entity.json: contains the mapping from Wikipedia page to wikidata entity ID
        # link_to_reference_images.json: contains the mapping from Wikipedia page to reference image URL

    # link_to_single_image.json: contains the mapping from Wikipedia page to image URL -> MAIN OUTPUT
import pickle

# first we need to get all links except the one with no page
import requests
from bs4 import BeautifulSoup
import json
import os
from collections import defaultdict
import json
from wiki import *
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

#################### Code for asyncio page requests ####################//
# Getting all pages in parallelized fashion

async def get_page(session, url, headers, max_retries=10, retry_delay=60):
    for _ in range(max_retries):
        try:
            async with session.get(url, headers=headers) as r:
                a = await r.text()
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

def get_qid(results):
    '''
    Code for getting qid from the html page
    Parameters: results from the asyncio page requests
    '''

    links_to_qid = {}
    for i in results:
        html = i[1]
        url = i[0]
        soup = BeautifulSoup(html, 'html.parser')
        target_a_tag = soup.find('li', attrs={'id': 't-wikibase'})
        if target_a_tag is None:
            html = requests.get(url).text
            soup = BeautifulSoup(html, 'html.parser')
            target_a_tag = soup.find('li', attrs={'id': 't-wikibase'})
            if target_a_tag is None:
                print(url)
                continue
                
        qid = target_a_tag.find('a').get('href').split('/')[-1]

        links_to_qid.update({url: qid})
    return links_to_qid



def get_image_infobox(results):
    '''
    Code for getting image from the html page
    Parameters: results from the asyncio page requests
    '''

    link_to_image ={}
    for i in results:
        html = i[1]
        url = i[0]
        soup = BeautifulSoup(html, 'lxml')
        target = soup.find('td' , class_ = 'infobox-image')
        if target is None:
                # if 'File:' in url:
                #     html = requests.get(url).text
                #     soup = BeautifulSoup(html, 'lxml')
                #     target = soup.find('td' , class_ = 'infobox-image')
                # if target is None:
                print("infobox not found")

                print(url)
                continue
        target = target.find('img')
        if target is None:
            print(url)
            continue
        target = target['src']
        link_to_image.update({url: target})
    return link_to_image


#################### Code for getting image from the html page ####################//
def parse(results):
    
    link_to_data = {}
    for _ in results: 
        url = _[0]
        html = _[1]
        req_div = get_images(html, url)
        link_to_data[url] = req_div
        
    return link_to_data


def get_images(html, url):
    """
    Handles SVGs, TIF and other kind of images link fetching
    """
    soup = BeautifulSoup(html, 'lxml')
    req_div = soup.find('div', class_='fullImageLink')
    try :
        if req_div is None:
           
            html = requests.get(url).text
            soup = BeautifulSoup(html, 'lxml')
            req_div = soup.find('div', class_='fullImageLink')
            if req_div is None:
                print(url)
        req_div = req_div.find('img')
        if req_div is None:
            print("img not dounf")
            print(url)
            return  "https://commons.wikimedia.org/wiki/Special:FilePath/"+url.split('/')[-1]
        req_div = req_div.get('src')
        if req_div is None:
            print("src not found")
            print(url)
            return  "https://commons.wikimedia.org/wiki/Special:FilePath/"+url.split('/')[-1]
    except Exception as e:
        print(e)
        return  "https://commons.wikimedia.org/wiki/Special:FilePath/"+url.split('/')[-1]
    # req_div = re.sub(r'\?[^?]*$', '', req_div)

    return req_div
####################################################################################//




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script's description")
    parser.add_argument("--input", help="Input file path", required=True)
    parser.add_argument("--output", help="Output directory", required=True)
    args = parser.parse_args()
    input_file = args.input
    output_dir = args.output
    # making dir if not exists
    if not os.path.exists(output_dir):

        os.makedirs(output_dir)
    with open(input_file, 'r') as pkl:
        all_urls = list(set(json.load(pkl)))
    print("input file " , input_file)

    urls_with_file= []
    urls_without_file = []
    for i in all_urls:
        if 'File:' in i :
            urls_with_file.append(i)
        else:
            urls_without_file.append(i)
    links2entities = {}
    print("#Links with File", len(urls_with_file))
    print("#Links without File", len(urls_without_file))
    print()
    print("############################################################")
    print("Querying Sparql Endpoint")

    # Querying the QID from Wikidata for the correspionding wikipedia page(workds for most of the pages)
    results = query_sparql_entities(WIKIDATA_URL_TO_ENTITY,WIKIDATA_ENDPOINT,list(urls_without_file),prefix="",is_links=True, n = 500 )
    for result in results:
            links2entities[result["file"]["value"]] = result["item"]["value"].split("/")[-1]
    
    
    print("#############################################################")
    # Some entities which had a proper page(i.e. were in urls_without_file list) but were not found in the sparql endpoint and so would need scraping separately
    links_with_no_entity = []
    for i in urls_without_file:
        if i not in links2entities:
            links_with_no_entity.append(i)
    print("Links with No Entity", len(links_with_no_entity))
    with open(os.path.join(output_dir, "link_with_no_entity.json"), "w") as pkl:
        json.dump(links_with_no_entity, pkl)
    
    # now we need to scrape the wikipedia pages for the links with no entity
    print()
    print("#############################################################")
    print("Scraping Wikipedia Pages")
    all_results = {} # Contains all the QIDs for the entities which weren;t found in the sparql endpoint
    for i in tqdm(range(0, len(links_with_no_entity), 100)):
        headers = {f'User-Agent': 'walter.white/1.0 (walter.white@research{i}.standford.edu) research purpose'}
        # Fetching the page HTML into results
        results = asyncio.run(main(links_with_no_entity[i:i+100], headers))
        
        # Getting the corresponding QID from the HTML
        results= get_qid(results)
        all_results.update(results)
        if i%300 ==0:
            time.sleep(20)
    
    print("#############################################################")
    for pair in all_results:
        
        links2entities[pair] = all_results[pair]
    
    with open(os.path.join(output_dir, "link_to_entity.json"), "w") as pkl:
        json.dump(links2entities, pkl)
    
    # now we need query the sparql endpoint for the links with file
    entities = {}
    for v in links2entities.values():
        entities.update({v:{}})

    print("#Entities", len(entities))

    print()
    print("#############################################################")
    print("Querying Sparql Endpoint for reference images")
    temp_entities = {} # Would store the reference image URL(and other SPARQL data) for the entity IDs
    for entity in entities.keys():
        temp_entities.update({entity:{}})
    
    # Querying the reference image URL from Wikidata for the correspionding entity ID and setting the reference image URL based on preference
    temp_entities = update_from_data(temp_entities)
    temp_entities = set_reference_images(temp_entities)
    entities2links={} # Contains mapping from entity ID to the corresponding wikipedia page(s)

    for link in links2entities: 
        entities2links.setdefault(links2entities[link],[])
        entities2links[links2entities[link]].append(link)
    link_to_reference_images= {}
    for entity in temp_entities: 
        ref = temp_entities[entity].get('reference_image',{})
        for i in entities2links[entity]:
            link_to_reference_images[i] = ref
    print("#############################################################")
    with open(os.path.join(output_dir, "link_to_reference_images.json"), "w") as pkl:
        json.dump(link_to_reference_images, pkl)
    
    # now we need to find the images with no reference image 
    entities_with_no_reference_images=[]
    for link , imgs  in link_to_reference_images.items():
        if imgs=={}:
            entities_with_no_reference_images.append(link)
    
    for link in urls_without_file: 
        if link not in link_to_reference_images:
            entities_with_no_reference_images.append(link)
    print("Entities with no reference images", len(entities_with_no_reference_images))


    # we need to scrape the wikipedia pages for the links with no reference images or entity. 
    all_results = {}
    for i in tqdm(range(0, len(entities_with_no_reference_images), 100)):
        headers = {f'User-Agent': 'walter.white/1.0 (walter.white@research{i}.standford.edu) research purpose'}
        results = asyncio.run(main(entities_with_no_reference_images[i:i+100], headers))
        results= get_image_infobox(results)
        all_results.update(results)
        if i%300 ==0:
            time.sleep(20)
    
    RESERVED_IMAGES = ['image', 'logo', 'flag', 'coat_of_arms', 'service_ribbon',  'seal','locator_map_image']
    print(RESERVED_IMAGES , "Preferences")
    print("Entities with no reference images", len(entities_with_no_reference_images))
    link_to_single_image = {}
    for entity_link, images in link_to_reference_images.items():
        selected_image = None
        for pref in RESERVED_IMAGES:
            for url, image_info in images.items():
                if image_info['source'] == pref:
                    selected_image = url
                    break
            if selected_image:
                break
        if selected_image:
            link_to_single_image[entity_link] = selected_image
    link_to_single_image = {k: "https://commons.wikimedia.org/wiki/Special:FilePath/"+v.split('/')[-1] for k,v in link_to_single_image.items()} # Converting URLs to downloadable links
    for link in all_results:
        if all_results[link].startswith('//'):
            link_to_single_image[link] = "https:" + all_results[link]
        else : 
            link_to_single_image[link] = all_results[link]
    
    for link in urls_with_file: 
        if link.split('.')[-1]!= 'mid': 
            link_to_single_image[link] ="https://commons.wikimedia.org/wiki/Special:FilePath/"+link.split('File:')[-1]
    
    print("#############################################################SVG")
    svg_links_tif = []
    for link in link_to_single_image:
        if link_to_single_image[link].split('.')[-1].lower()  in ['tiff','svg', 'tif','djvu']:
            print(link_to_single_image[link])
            svg_links_tif.append("https://commons.wikimedia.org/wiki/File:"+link_to_single_image[link].split('/')[-1])
    
    print("SVG links", len(svg_links_tif))
    # Now we need to get the Full_imagelink for svg images 
    print("Scraping SVG links")
    all_results = {}
    for i in tqdm(range(0, len(svg_links_tif), 100)):
        headers = {f'User-Agent': 'walter.white/1.0 (walter.white@research{i}.standford.edu) research purpose'}
        results = asyncio.run(main(svg_links_tif[i:i+100], headers))
        results= parse(results)
        all_results.update(results)
        if i%300 ==0:
            time.sleep(20)
    
    for link in link_to_single_image: 
        temp_url = "https://commons.wikimedia.org/wiki/File:"+link_to_single_image[link].split('/')[-1]
        if temp_url in all_results:
            print(link_to_single_image[link])
            link_to_single_image[link] = all_results[temp_url]
    with open(os.path.join(output_dir, "link_to_single_image.json"), "w") as pkl:
        json.dump(link_to_single_image, pkl)
    

    
    
            
    
    

    


    

    







    


    
            

    
    