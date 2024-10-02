import pickle
import requests
from bs4 import BeautifulSoup
import json
import os
from collections import defaultdict
import json
import hashlib
# from wiki import *
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
import time
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933130000 
import io
import re
from urllib.parse import unquote
image_link_to_path = {}
import argparse
DATASET_PATH = "/home/suyash/final_repo/WikiTableQuestions/temp_collage_images"

############# Asyncio code to download images in a parallelized fashion################################################
async def get_page_d(session, url, headers, max_retries=5, retry_delay=60):                                                         
    """
    Async function to download the images from a particular URL.
    """
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
                image_link_to_path[url]= path 
                print(path)
                return [url, a]
        except Exception as e:
            print(f"An error occurred: {e}",url)
            await asyncio.sleep(retry_delay)
    print(f"Max retries reached for URL: {url}")
    return [url, None]

async def get_all_d(session, urls, semaphore, headers, delay=0, max_retries=10, retry_delay=60):
    tasks = []
    for i, url in tqdm(enumerate(urls)):
        async with semaphore:
            task = asyncio.create_task(get_page_d(session, url, headers, max_retries, retry_delay))
            tasks.append(task)
            await asyncio.sleep(delay)  # Introduce delay between requests
    results = await asyncio.gather(*tasks)
    return results

async def main_d(urls, headers, delay=0, max_retries=3, retry_delay=2):
    semaphore = asyncio.Semaphore(10)  # Set the maximum number of concurrent requests
    async with aiohttp.ClientSession() as session:
        data = await get_all_d(session, urls, semaphore, headers, delay, max_retries, retry_delay)
        return data
#######################################################################################################################



def generate_unique_path(link, location):
    # Function to generate a unique path for each collage/seal link's image file. Treat as a blackbox
    hash_object = hashlib.sha256()

    if location :
        link = link + str(location)
    hash_object.update(link.encode('utf-8'))

    hash_hex = hash_object.hexdigest()
    unique_path = hash_hex[:8]

    return unique_path



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

########################################################################

# TODO: Ask jainit what is the parse and get_images function doing?
def parse(results):
    
    link_to_data = {}
    for _ in results: 
        url = _[0] # Image page URL
        html = _[1] # Image page raw HTML
        req_div = get_images(html, url) # Handles SVG, TIF
        link_to_data[url] = req_div
        
    return link_to_data

def get_images(html, url):
    """
    Handles SVGs, TIF and other kind of images link fetching. Returns the link to the image.
    """
    soup = BeautifulSoup(html, 'lxml')
    req_div = soup.find('div', class_='fullImageLink')
    try :
        if req_div is None:
           
            html = requests.get(url).text
            soup = BeautifulSoup(html, 'lxml')
            req_div = soup.find('div', class_='fullImageLink')
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

    return "https:"+req_div
####################################################################################//

################################################################################
######code for extraction of image links from wikipedia page ###################
################################################################################
def create_collage(images, output_path):
    # Calculate the number of rows and columns based on the number of images
    num_images = len(images)
    if num_images == 0:
        print("No images for collage")
        return
    rows = int(num_images ** 0.5)
    columns = (num_images // rows) + (1 if num_images % rows != 0 else 0)

    # Set the collage size (you can adjust this based on your preference)
    collage_width = columns * 300
    collage_height = rows * 300

    # Create a new image for the collage
    collage = Image.new('RGB', (collage_width, collage_height))

    # Paste each image into the collage
    for i, image_path in enumerate(images):
        img = Image.open(image_path)
        img = img.resize((300, 300))  # Resize images to fit in the collage
        row = i // columns
        col = i % columns
        collage.paste(img, (col * 300, row * 300))

    # Save the collage
    collage.save(output_path)
    print(f"Collage saved at {output_path}")
###############################################################33
def extract_landscapes(url, html):
    soup = BeautifulSoup(html, 'html.parser')

    # Find all <td> elements with class 'infobox-full-data'
    all_table_cells =soup.find_all('td', {'class': 'infobox-full-data'}) + soup.find_all('td', {'class': 'infobox-image'})
    
    # Iterate through each <td> element
    for table_cell in all_table_cells:
        if table_cell.find('div', {'class': 'locmap'}):
            print("Skipping <td> element with class 'locmap'.")
            continue
        # Find span elements with type='mw:File' or typeof='mw:File' within the table cell
        span_elements = table_cell.find_all('span', {'type': 'mw:File'}) + table_cell.find_all('span', {'typeof': 'mw:File'}) + table_cell.find_all('span', {'typeof': 'mw:File/Frameless'}) 

        # Check if any non-empty span elements are found
        if span_elements:
            # Initialize the return list
            return_list = []

            # Iterate through each span element
            for span_element in span_elements:
                # Navigate to the parent <a> element
                a_element = span_element.find('a')

                # Check if the <a> element is found
                if a_element:
                    # Get the value of the 'href' attribute
                    href_value = a_element.get('href')

                    # Print or use the 'href' value as needed
                    return_list.append("https://en.wikipedia.org" + href_value)
                else:
                    print(url)
                    print("No parent <a> element found.")

            # Check if the return list is not empty
            if return_list:
                return return_list

    # If no suitable <td> element is found, print a message and return an empty list
    print(url)
    print("No <td> element with non-empty span_elements found.")
    return []


def extract_all_landscapes(results):
    pagelink_to_landscapes={}
    try :
        for i in tqdm(results): 
            url = i[0]
            html = i[1]
            links = extract_landscapes(url , html)
            pagelink_to_landscapes[url] = links; 
            
        
        return pagelink_to_landscapes
    except Exception as e:
        print(e)
        return {}
#####################################################################################3
######################################
# def extract_seals(url, html):
#     try : 
#         soup = BeautifulSoup( html , 'lxml')
#         table_class_a = soup.find('table', {'class': 'infobox ib-settlement vcard'})
#         table_class_a = table_class_a.find('td' , {'class':'infobox-full-data maptable'})
#         links  = table_class_a.findAll("a")
            
#         image_links = [link for link in links if link.find('img') is not None]

#         # Print the filtered image links
#         return_links = []
#         for image_link in image_links:
#             return_links.append("https://en.wikipedia.org"+image_link['href'])
#         return return_links

#     except Exception as e:
#         print(e)
#         print(url)
#         return []
def extract_seals(url, html):
    soup = BeautifulSoup(html, 'html.parser')

    # Find all <td> elements with class 'infobox-full-data' or 'infobox-image'
    all_table_cells = soup.find_all('td', {'class': 'infobox-full-data'}) + soup.find_all('td', {'class': 'infobox-image'}) + soup.find_all('td' , {'class':'infobox-full-data maptable'})

    # Iterate through each <td> element
    for table_cell in all_table_cells:
        if table_cell.find('div', {'class': 'locmap'}):
            print("Skipping <td> element with class 'locmap'.")
            continue

        # Find span elements with type='mw:File' or typeof='mw:File' within the table cell and add them to the list
        span_elements = table_cell.find_all('span', {'type': 'mw:File'}) + table_cell.find_all('span', {'typeof': 'mw:File'}) + table_cell.find_all('span', {'typeof': 'mw:File/Frameless'}) 
        # print(span_elements)
        # Check if any non-empty span elements are found
        if span_elements:
            # Initialize the return list
            return_list = []

            # Iterate through each span element
            for span_element in span_elements:
                # Navigate to the parent <a> element
                a_element = span_element.find('a')
                break_this = False

                # Check if the <a> element is found
                if a_element:
                    # Get the value of the 'href' attribute
                    href_value = a_element.get('href')
                    title = a_element.get('title')
                    # print(title)
                    if title is not None  and  (( not ('coa' in title.lower() or 'flag' in title.lower() or 'logo' in title.lower() or 'seal' in title.lower() or 'emblem' in title.lower())) or (not ('coa' in href_value.lower() or 'flag' in href_value.lower() or 'logo' in href_value.lower() or 'seal' in href_value.lower() or 'emb')))  :
                        # print(title)
                        break_this = True
                        break
                    elif (title is None) and  (not ('coa' in href_value.lower() or 'flag' in href_value.lower() or 'logo' in href_value.lower() or 'seal' in href_value.lower() or 'emblem' in href_value.lower())):
                        # print(href_value , 'here' , return_list)
                        break_this = True
                        break
                    
                    # Print or use the 'href' value as needed
                    return_list.append("https://en.wikipedia.org" + href_value)
                    
                else:
                    print(url)
                    print("No parent <a> element found.")
                
            if break_this:
                # return_list = []
                continue
            elif return_list: 
                return return_list
    print("returning empty list")
    print(url)
    return []

def extract_all_seals( results ):
    pagelinks_to_seals={}
    for i in results: 
        url = i[0]
        html = i[1]
        links = extract_seals(url, html )
        pagelinks_to_seals[url] = links 
    
    return pagelinks_to_seals
        







if __name__=='__main__':
    
    argp = argparse.ArgumentParser()
    argp.add_argument("--location", type=int, default=0, help="0 for landscapes and 1 for seals")
    argp.add_argument("--path", type=str, default="/home/suyash/final_repo/WikiTableQuestions/category_filtered_outputs/seal_links.json", help="path to the json file containing the Wikipedia page links in list for which landscape images are to be scraped")
    args = argp.parse_args()
    
    download_images = True
    with open(args.path, 'r') as f:
        links = json.load(f)
    
    # links = ["https://en.wikipedia.org/wiki/Italy"]
    
    if args.location == 0:    
        links_to_landscapes={} # Wikipedia link -> List of image URLs for the landscape
        # get landscapes link from wikipedia
        for i in tqdm(range(0 , len(links) , 50 )) : 
            headers = {f'User-Agent': 'walter.white/1.0 (walter.white@research{i}.standford.edu) research purpose'}
            results= asyncio.run(main(links[i:i+50], headers))
            links_to_landscapes.update(extract_all_landscapes(results))
        
        print(links_to_landscapes)
        print()

        
        combined_list =[] # This is a combined list of unique images from all the Wikipedia pages
        for i in links_to_landscapes.values():
            combined_list.extend(list(i))

        combined_list = list(set(combined_list))
        print(combined_list)
        file_to_image_links={}
        
        # This function gets the link to each image which can be directly downloaded
        for i in tqdm(range(0 , len(combined_list), 50 )): 
            headers = {f'User-Agent': 'walter.white/1.0 (walter.white@research{i}.standford.edu) research purpose'}
            results= asyncio.run(main(combined_list[i:i+50], headers))
            rr = parse(results)
            file_to_image_links.update(rr)
            
        
        ### now we download the images and store the paths 
        if download_images:
            print(list(file_to_image_links.values()))
            for i in tqdm(range(0 , len(file_to_image_links), 50 )):
                headers = {f'User-Agent': 'walter{i}lewin{i}/1.0 (walter.lewin@standford.edu) research purpose'}
                results = asyncio.run( main_d(list(file_to_image_links.values())[i: i+50], headers))
        else:
            for i in file_to_image_links:
                new_url = re.sub(r'\?[^?]*$', '', url)
                file_extension = new_url.split(".")[-1].lower()
                image_name = new_url.split("/")[-1]
                last_occurrence_index = image_name.rfind('.')
                before_last_occurrence = unquote(image_name[:last_occurrence_index]).replace(" ", "_")
                before_last_occurrence= before_last_occurrence[:200]

            # Save the compressed image
            path = os.path.join(DATASET_PATH, f"{before_last_occurrence}.{file_extension}")
            
        
        
        #####################################
        # image_link_to_path has paths of downloaded paths now we need to make collage 
        #####################################
        
        # create a dict of url to images 
        page_link_to_paths = {} # Wikipedia page link to image path
        for k , v in links_to_landscapes.items(): 
            new_list = [] 
            for i in v:
                if file_to_image_links[i] in image_link_to_path:
                    new_list.append(image_link_to_path[file_to_image_links[i]])
            
            page_link_to_paths[k] = new_list
        
        
        # create collage for images 
        links_to_landscape_collage_path = {}
        for i in page_link_to_paths: 
            if len(page_link_to_paths[i]) == 0:
                continue
            collage_path = "../../collage_outputs/landscape_collages/"+generate_unique_path(i , args.location)+".jpg"
            create_collage( page_link_to_paths[i], collage_path)
            links_to_landscape_collage_path[i] = collage_path
            
        
        print( links_to_landscape_collage_path)
        json.dump(links_to_landscape_collage_path , open("links_to_landscape_collage_path.json" , 'w'))
        json.dump( links_to_landscapes, open("links_to_landscapes.json" , 'w'))

    
    else: 
        ## now we need make collages for seals and flag 
        links_to_seals={}
        ## get landscapes link from wikipedia
        for i in tqdm(range(0 , len(links) , 50 )) : 
            headers = {f'User-Agent': 'walter.white/1.0 (walter.white@research{i}.standford.edu) research purpose'}
            results= asyncio.run(main(links[i:i+50], headers))
            links_to_seals.update(extract_all_seals(results))
            
        print(links_to_seals)
        print()
        
        combined_list_seals =[]
        for i in links_to_seals.values():
            combined_list_seals.extend(list(i)) 
        
        combined_list_seals = list(set(combined_list_seals))
        print(combined_list_seals)
        file_to_image_links_seals={}
        
        for i in tqdm(range(0 , len(combined_list_seals), 50 )):
            headers = {f'User-Agent': 'walter.white/1.0 (walter.white@research{i}.standford.edu) research purpose'}
            results= asyncio.run(main(combined_list_seals[i:i+50], headers))
            rr = parse(results)
            file_to_image_links_seals.update(rr)
            
        print(list(file_to_image_links_seals.values()))
        
        for i in tqdm(range(0 , len(file_to_image_links_seals), 50 )):
            
            headers = {f'User-Agent': 'walter.white/1.0 (walter.white@research{i}.standford.edu) research purpose'}
            results = asyncio.run( main_d(list(file_to_image_links_seals.values())[i: i+50], headers))
            
        # image_link_to_path has paths of downloaded paths now we need to make collage
        # # json.dump( image_link_to_path , open("image_link_to_path.json" , 'w'))
        # create a dict of url to images
        page_link_to_paths_seals = {}
        for k , v in links_to_seals.items():
            new_list = []
            for i in v:
                if file_to_image_links_seals[i] in image_link_to_path:
                    new_list.append(image_link_to_path[file_to_image_links_seals[i]])
        
            page_link_to_paths_seals[k] = new_list
            
        # create collage for images
        links_to_seals_collage_path = {}
        for i in page_link_to_paths_seals:
            if len(page_link_to_paths_seals[i]) == 0:
                continue
            collage_path = "/home/suyash/final_repo/WikiTableQuestions/seals_collages/"+generate_unique_path(i, location=args.location)+".jpg"
            create_collage( page_link_to_paths_seals[i], collage_path)
            links_to_seals_collage_path[i] = collage_path
        
        print( links_to_seals_collage_path)
        
        json.dump(links_to_seals_collage_path , open("links_to_seals_collage_path.json" , 'w'))
        
        
        json.dump( links_to_seals, open("links_to_seals.json" , 'w'))
