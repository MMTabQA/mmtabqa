#!/usr/bin/env python
# coding: utf-8







#




from pprint import pprint
import requests
import concurrent.futures
import warnings
from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.error import URLError, HTTPError
import requests
from http.cookiejar import Cookie
import os
import requests
from urllib.parse import urlparse
from tqdm import tqdm
from bs4 import BeautifulSoup
# One client (user agent + IP) is allowed 60 seconds of processing time each 60 seconds
# https://www.mediawiki.org/wiki/Wikidata_Query_Service/User_Manual
WIKIDATA_COMPUTE_LIMIT = 60

QID_URI_PREFIX = "http://www.wikidata.org/entity/"
HUMAN = QID_URI_PREFIX + 'Q5'

SPECIAL_PATH_URI_PREFIX = "http://commons.wikimedia.org/wiki/Special:FilePath/"
SPECIAL_FILE_PATH_URI_PREFIX = SPECIAL_PATH_URI_PREFIX + "File:"
UPLOAD_URI_PREFIX = "http://upload.wikimedia.org/wikipedia/commons/"
VALID_DATE_TYPE = 'http://www.w3.org/2001/XMLSchema#dateTime'

# restrict media to be images handleable by PIL.Image (or convertible with Wikimedia thumbnails)
VALID_ENCODING = {"png", "jpg", "jpeg", "tiff", "svg", "tif",  "djvu"}

# used to make thumbnails in file_name_to_thumbnail
EXTENSIONS_PRE_AND_SUFFIXES = {
    "svg": ("", ".png"),
    "tif": ("lossy-page1-", ".jpg"),
    "tiff": ("lossy-page1-", ".jpg"),
    "pdf": ("page1-", ".jpg"),
    "djvu": ("page1-", ".jpg")
}

# rules of preferences over licenses, the higher the better (0 is reserved for missing values or other licenses)
LICENSES = {
    "CC0": 8,
    "PUBLIC DOMAIN MARK": 7,
    "PUBLIC DOMAIN": 7,
    "PDM": 7,
    "BY": 6,
    "BY-SA": 5,
    "BY-NC": 4,
    "BY-ND": 3,
    "BY-NC-SA": 2,
    "BY-NC-ND": 1
}

# fileTitle shouild be of form "File:filename.ext"
WIKIDATA_GET_COMMONS_IMAGES_CAT = """
SELECT * WHERE {
  VALUES ?fileTitle { %s }
  # BIND(CONCAT("File:", STRAFTER(wikibase:decodeUri(STR(?image)), "http://commons.wikimedia.org/wiki/Special:FilePath/")) AS ?fileTitle)
  # # Query the MediaWiki API Query Service for
  SERVICE wikibase:mwapi {
      # Categories that contain these pages
      bd:serviceParam wikibase:api "Categories";
                      wikibase:endpoint "commons.wikimedia.org";
                      wikibase:limit 1000;
                      mwapi:titles  ?fileTitle.
      # Output the page title and category
      ?title wikibase:apiOutput mwapi:title.
      ?category wikibase:apiOutput mwapi:category .  
  }
"""
# Template for wikidata to query many different attributes of a list of entities
# should be used like
# >>> WIKIDATA_QUERY % "wd:Q76 wd:Q78579194 wd:Q42 wd:Q243"
# i.e. entity ids are space-separated and prefixed by 'wd:'
WIKIDATA_QUERY = """
SELECT ?entity ?entityLabel ?instanceof ?instanceofLabel ?commons ?image ?flag ?coat_of_arms ?seal  ?logo ?locator_map_image ?service_ribbon ?occupation ?occupationLabel ?gender ?genderLabel ?freebase ?date_of_birth ?date_of_death ?taxon_rank ?taxon_rankLabel ?country ?countryLabel
{
  VALUES ?entity { %s }
  OPTIONAL { ?entity wdt:P373 ?commons . }
  ?entity wdt:P31 ?instanceof .
  OPTIONAL { 
    ?entity wdt:P18 ?image . 
  }
  OPTIONAL { 
    ?entity wdt:P41 ?flag . 
  }
  OPTIONAL { 
  ?entity wdt:P237 ?coat_of_arms .
  }
  OPTIONAL { 
    ?entity wdt:P154 ?logo . 
  }
  OPTIONAL { 
    ?entity wdt:P242 ?locator_map_image . 
  }
  OPTIONAL { 
    ?entity wdt:P158 ?seal . 
  }
  OPTIONAL { 
    ?entity wdt:P2425 ?service_ribbon . 
  }
  OPTIONAL { ?entity wdt:P21 ?gender . }
  OPTIONAL { ?entity wdt:P106 ?occupation . }
  OPTIONAL { ?entity wdt:P646 ?freebase . }
  OPTIONAL { ?entity wdt:P569 ?date_of_birth . }
  OPTIONAL { ?entity wdt:P570 ?date_of_death . }
  OPTIONAL { ?entity wdt:P105 ?taxon_rank . }
  OPTIONAL { ?entity wdt:P17 ?country . }
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""
# get all feminine labels
WIKIDATA_FEMININE_QUERY = """
SELECT ?entity ?entity_female_label
{
  VALUES ?entity { %s }
  ?entity wdt:P2521 ?entity_female_label .
  FILTER(LANG(?entity_female_label) = "en").
}
"""

# query super classes of a given class list
# use
# >>> WIKIDATA_SUPERCLASSES_QUERY % (qids, "wdt:P279+")
# to query all superclasses
WIKIDATA_SUPERCLASSES_QUERY = """
SELECT ?class ?classLabel ?subclassof ?subclassofLabel
WHERE
{
  VALUES ?class { %s }.
  ?class %s ?subclassof.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""


WIKIDATA_ENDPOINT = "	https://query.wikidata.org/sparql"

# see update_from_data
RESERVED_IMAGES = ['image', 'logo', 'flag', 'coat_of_arms', 'service_ribbon', 'seal' , 'locator_map_image']
MULTIPLE_KEYS = {'instanceof', 'occupation'}.union(RESERVED_IMAGES)
UNIQUE_KEYS = {'entityLabel', 'gender', 'genderLabel', 'commons', 'freebase', 'date_of_birth', 'date_of_death', 'taxon_rank', 'taxon_rankLabel','country', 'countryLabel'}

# template for beta-commons SPARQL API to query images that depict (P180) entities
# same usage as WIKIDATA_QUERY
COMMONS_SPARQL_QUERY = """
SELECT ?depicted_entity ?commons_entity ?special_path ?url ?encoding ?all_depicted_entities  (YEAR(?date) AS ?year) WHERE {
  VALUES ?depicted_entity {%s }
  ?commons_entity wdt:P180 ?depicted_entity .
  ?commons_entity schema:contentUrl ?url .
  ?commons_entity schema:encodingFormat ?encoding .
  ?commons_entity wdt:P180 ?all_depicted_entities .
OPTIONAL{  ?commons_entity wdt:P571 ?date .}

  # Restrict media to be images handleable by PIL.Image
  VALUES ?encoding { "image/png" "image/jpg" "image/jpeg" "image/tiff" "image/gif" }

  BIND(iri(concat("http://commons.wikimedia.org/wiki/Special:FilePath/", wikibase:decodeUri(substr(str(?url), 53)))) AS ?special_path)
}

"""
# query entities depicted in images given image identifier (see above for more details)
COMMONS_DEPICTED_ENTITIES_QUERY = """
SELECT ?file ?entity ?url WHERE {
  VALUES ?file { %s }  # Replace with your desired entity IDs
  ?file wdt:P180 ?entity .
}
"""

COMMONS_REF_IMG_DEPICTIONS_QUERY = """
SELECT ?entity ?urls  ?depicted_entities (YEAR(?date) AS ?year)
WHERE
{
  VALUES ?urls {%s}
  ?entity schema:url ?urls .
  ?entity wdt:P180 ?depicted_entities .
  OPTIONAL{?entity wdt:P571 ?date .}
}
"""
COMMONS_IMG_DATE_QUERY = """
SELECT *
{
  VALUES ?urls { %s }
  ?entity schema:url ?urls .
  ?entity wdt:P571 ?date .
}
"""

COMMONS_SPARQL_ENDPOINT = "https://commons-query.wikimedia.org/sparql"

# get all files or sub-categories in a Commons category
# use like
# >>> COMMONS_REST_LIST.format(cmtitle=<str including "Category:" prefix>, cmtype="subcat"|"file")
# e.g.
# >>> COMMONS_REST_LIST.format(cmtitle="Category:Barack Obama in 2004", cmtype="subcat")
COMMONS_REST_LIST = "https://commons.wikimedia.org/w/api.php?action=query&list=categorymembers&cmtitle={cmtitle}&cmprop=title|type&format=json&cmcontinue&cmlimit=max&cmtype={cmtype}"

# query images URL, categories and description
# use like
# >>> COMMONS_REST_TITLE.format(titles=<title1>|<title2>) including the "File:" prefix
# e.g.
# >>> COMMONS_REST_TITLE.format(titles="File:Barack Obama foreign trips.png|File:Belgique%20-%20Bruxelles%20-%20Grand-Place%20-%20C%C3%B4t%C3%A9%20nord-est.jpg")
COMMONS_REST_TITLE = "https://commons.wikimedia.org/w/api.php?action=query&titles={titles}&prop=categories|description|imageinfo&format=json&iiprop=url|extmetadata&clshow=!hidden"
COMMONS_REST_TITLE_CAT = "https://commons.wikimedia.org/w/api.php?action=query&titles={titles}&prop=categories&format=json&iiprop=url|extmetadata&clshow=!hidden"

WIKIDATA_URL_TO_ENTITY = """
prefix schema: <http://schema.org/>
SELECT  ?file ?item  ?itemLabel WHERE {
  VALUES ?file { %s } 
  ?file schema:about ?item .
    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } 
}
"""

VALID_IMAGE_HEURISTICS = {"categories", "description", "depictions", "title"}









def init_session(endpoint, token):
    domain = urlparse(endpoint).netloc
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'jainitjainit/1.0 jainit.bafna@research.iiit.ac.in',
    })
    session.cookies.set_cookie(Cookie(0, 'wcqsOauth', token, None, False, domain, False, False, '/', True,
        False, None, True, None, None, {}))
    return session


def query_sparql_entities(query, ENDPOINT, wikidata_ids, prefix='wd:',
                          n=500, return_format=JSON, is_links=False, description=None):
    """
    Queries query%entities by batch of n (defaults 100)
    where entities is n QIDs in wikidata_ids space-separated and prefixed by prefix
    (should be 'wd:' for Wikidata entities and 'sdc:' for Commons entities)

    Returns query results
    """


    # ENDPOINT = 'https://commons-query.wikimedia.org/sparql'
    session = init_session(ENDPOINT, "dee5291f472b9404aea451030e137a97.8ed3225303f910c817f9dafb1b0c67b22c53fb3a")

    results, qids = [], []
    skip_total = 0
    # query only n qid at a time
    for i, qid in enumerate(tqdm(wikidata_ids, desc=description)):
        if is_links: qid=f"<{qid}>"
        qids.append(prefix+qid)
        if (i + 1) % n == 0 or i == (len(wikidata_ids) - 1):
            q_string = query % " ".join(qids)

            #print(q_string)
            # sparql.setQuery(q_string)
            try:
                # #print(q_string)
                response = session.post(
                    url=ENDPOINT,
                    data={'query': q_string},
                    headers={'Accept': 'application/json'})
                # #print(response)
            except HTTPError as e1:
                if str(e1.code).strip() == '429':
                    # HACK: sleep WIKIDATA_COMPUTE_LIMIT seconds to avoid 'HTTP Error 429: Too Many Requests'
                    time.sleep(WIKIDATA_COMPUTE_LIMIT)
                    # try one more time
                    try:
                        response = session.post(
                          url=ENDPOINT,
                          data={'query': q_string},
                          headers={'Accept': 'application/json'})
                    except HTTPError as e2:
                        warnings.warn(f"HTTPError: {e2}\n"
                                      f"Query failed twice after waiting {WIKIDATA_COMPUTE_LIMIT}s in-between, "
                                      f"skipping the following QIDs:\n{qids}")
                        skip_total += len(qids)
                        qids = []
                        continue
                else:
                    warnings.warn(f"HTTPError: {e1}\nskipping the following QIDs:\n{qids}")
                    skip_total += len(qids)
                    qids = []
                    continue
            # ##print("AAAAAAAAAAAAAAAAA")
            # #print(response.json())
            except Exception as e:
                warnings.warn(f"Exception: {e}\nskipping the following QIDs:\n{qids}")
                skip_total += len(qids)
                qids = []
                continue
            try:
                results += response.json()['results']['bindings']
            except: 
                print(q_string)
                # print(response.json())
            qids = []

    ##print(f"Query succeeded! Got {len(results)} results, skipped {skip_total} QIDs")
    # ##print(results)
    return results






def get_commons_qid(url):
    title = url.split("/")[-1]
    response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&titles={title}&format=json")
    try:
        keyz = (list(response.json()["query"]["pages"].keys())[0])
        return url, response.json()["query"]["pages"][keyz]["pageprops"]["wikibase_item"]
    except:
        print(response.json() )
        print(url)
        return url, None

def parallel_get_commons_qid(urls):
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks to the executor
        futures = [executor.submit(get_commons_qid, url) for url in urls]
        # Get the results as they become available
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
    return results









def update_from_data(entities, skip=None):
    """Updates entities with info queried in from Wikidata"""
    ##print('jainit')
    # query Wikidata
    if skip is None:
        wikidata_ids = list(entities.keys())
    # skip all entities that already have the `skip` attribute
    else:
        wikidata_ids = list([entity for entity in entities.keys()])
    results = query_sparql_entities(WIKIDATA_QUERY, WIKIDATA_ENDPOINT, wikidata_ids,
                                    description="Querying Wikidata")
    ##print('jainit')
    # ##print(results)
    ##print(entities)
    # update entities with results
    for result in tqdm(results, desc="Updating entities"):
        qid = result['entity']['value'].split('/')[-1]
        # handle keys/attributes that are unique
        for unique_key in (UNIQUE_KEYS & result.keys()):
            # simply add or update the key/attribute
            entities[qid][unique_key] = result[unique_key]
        # handle keys/attributes that may be multiple
        for multiple_key in (MULTIPLE_KEYS & result.keys()):
            # create a new dict for this key/attribute so we don't duplicate data
            entities[qid].setdefault(multiple_key, {})
            # store corresponding label in the 'label' field
            result[multiple_key]['label'] = result.get(multiple_key + 'Label')
            # value (e.g. QID) of the attribute serves as key
            multiple_value = result[multiple_key]['value']
            entities[qid][multiple_key][multiple_value] = result[multiple_key]

    return entities


def set_reference_images(entities):
    """Set a reference image using RESERVED_IMAGES as order of preference if the entity has any available"""

    for qid,entity in entities.items():
        # try to get illustrative image, fallback on other images if available
        # "image" is expected to be the first element of RESERVED_IMAGES
        entity['reference_image']={}
        for entity_image_key in RESERVED_IMAGES:
           
            entity_image = entity.get(entity_image_key)
            if entity_image is not None:
                encoding = None
                # HACK: pop 'type' and 'value' that might have been gathered
                # when we considered only a single illustrative image per entity
                # entity_image.pop('type', None)
                # entity_image.pop('value', None)

                # filter encodings
                for v in entity_image.values():
                    url = v.get('value')
                    if v is None:
                        continue
                    encoding = url.split('.')[-1].lower()
                    if encoding in VALID_ENCODING:
                        entity['reference_image'][url]={}
                        entity['reference_image'][url]['source'] = entity_image_key
                        break
                
    # remove the entity if it has no reference image we can try to scrap from wikipedia
    # entities = {qid: entity for qid, entity in entities.items() if 'reference_image' in entity}
    

    return entities


#

def get_categories_for_all_images(entities):
    
    all_files=set()
    for entity in entities:
        if 'reference_image' in entities[entity]:
            all_files.add(entities[entity]['reference_image']['url'])
        for depictions in entities[entity]['depictions']:
            all_files.add(entities[entity]['depictions'][depictions]['url']['value'])

    file2strings = {}
    strings2categories = {}
    for file in all_files:
        
        file2strings[file] = "File:" + file.split('/')[-1]
    
    # print(len(file2strings))
    results = query_sparql_entities(WIKIDATA_GET_COMMONS_IMAGES_CAT,COMMONS_SPARQL_ENDPOINT, list(file2strings.values()), is_links=True , prefix="", n= 50 )
    print(results)
    for result in results:
        
        file = result['fileTitle']['value'].split(':')[-1]
        # print(file)
        strings2categories.setdefault(file, [])
        strings2categories[file].append(result['category']['value'].split(':')[-1])

    for entity in entities:
        if 'reference_image' in entities[entity]:
            # print(entities[entity]['reference_image']['url'].split('/')[-1])
            if entities[entity]['reference_image']['url'].split('/')[-1] in strings2categories:
                # print(entities[entity]['reference_image']['url'].split('/')[-/1])
                # print("dafsdf")
                entities[entity]['reference_image']['categories'] = strings2categories[entities[entity]['reference_image']['url'].split('/')[-1]]
            
        for depictions in entities[entity]['depictions']:
            if entities[entity]['depictions'][depictions]['url']['value'].split('/')[-1] in strings2categories.keys():
                # print(entities[entity]['depictions'][depictions]['url']['value'].split('/')[-1])
                # print("dafsdf")
                entities[entity]['depictions'][depictions]['categories'] = strings2categories[entities[entity]['depictions'][depictions]['url']['value'].split('/')[-1]]

    return entities


def get_depictions_in_reference_image(entities):
  references = {}
  references_to_entities = {}
  for entity in entities:
    #print(entity)
    linkz = entities[entity]['reference_image']['url']
    entities[entity]['reference_image']['depictions'] = set([entity])
    references[entities[entity]['reference_image']['url']] = {}
    if linkz not in references_to_entities:
      references_to_entities[linkz] = []
    references_to_entities[linkz].append(entity)
  results = query_sparql_entities(COMMONS_REF_IMG_DEPICTIONS_QUERY, COMMONS_SPARQL_ENDPOINT, references.keys(), is_links=True)
  # ##print(references.keys())
  for result in results:
    url = result['urls']['value']
    corresp_og_entities = references_to_entities[url]
    for ent in corresp_og_entities:
      dic = entities[ent]['reference_image']
      entities[ent]['reference_image']['depictions'].add(result['depicted_entities']['value'].split("/")[-1])
      if 'year' in result:
        year = result['year']['value']
      else :
        year = None
      entities[ent]['reference_image']['year'] = year

  return entities





# entities = get_depictions_in_reference_image(entities)


#


# entities


#


from collections import defaultdict
def update_from_commons_sparql(entities):
    # query Wikimedia Commons
    results = query_sparql_entities(COMMONS_SPARQL_QUERY, COMMONS_SPARQL_ENDPOINT,
                                    entities.keys(),
                                    description="Querying Wikimedia Commons")
    # ##print(results)
    # set one of the depictionas as the reference image for all entities
    for entity in entities.keys():
        entities[entity]['depictions'] = {}

    # update entities with results

    for result in tqdm(results, desc="Updating entities"):
        qid = result['depicted_entity']['value'].split('/')[-1]
        commons_qid = result['commons_entity']['value'].split("/")[-1]
        # create a new key 'depictions' to store depictions in a dict
        entities[qid].setdefault("depictions", {})
        
        # use commons_qid (e.g. https://commons.wikimedia.org/entity/M88412327) as key in this dict
        entities[qid]["depictions"].setdefault(commons_qid, {})
        entities[qid]["depictions"][commons_qid]['url'] = result['url']
        entities[qid]["depictions"][commons_qid]['special_path'] = result['special_path']
        year= None
        if "year" not in result:
            year = None
        else :
            year = result["year"]['value']

        entities[qid]["depictions"][commons_qid]["year"]= year
        entities[qid]["depictions"][commons_qid].setdefault("all_depictions", [])
        entities[qid]["depictions"][commons_qid]['all_depictions'].append(result['all_depicted_entities'])

    return entities





# update_from_commons_sparql(entities)


#


# entities


#


def get_top_images(entities, top_n=5):

    for entity_id in entities:
        count=0
        total_count=0
        gold_depictions = entities[entity_id]['reference_image']["depictions"]
        # gold_categories = entities[entity_id]['reference_image']["categories"]
        if "year" not in entities[entity_id]['reference_image'] or entities[entity_id]['reference_image']["year"] is None:
            gold_year = 100000 
        else:
            gold_year = entities[entity_id]['reference_image']["year"]
        score_array = []
        
        for depiction in entities[entity_id]["depictions"]:
            curr_score = 0
            depiction_dict = entities[entity_id]["depictions"][depiction]
            commons_entity_depictions = depiction_dict["all_depictions"]
            for comm_entity in commons_entity_depictions:
                ent = comm_entity["value"].split("/")[-1]
                if ent in gold_depictions:
                    curr_score+=10
                else:
                    # TODO: Change this to if the category is same
                    curr_score-=5
            if "year" in depiction_dict:
                curr_year = depiction_dict["year"]
                # #print(curr_year, gold_year)
                if curr_year is not None:
                    curr_score += 20 - (abs(int(curr_year) - int(gold_year)))**2
                    count+=1
                else: 
                    curr_score -= 20
            
            total_count+=1
            # for category in depiction_dict["categories"]:
            #     if category in entities[entity_id]["categories"]:
            #         curr_score+=5
            #     else:
            #         curr_score-=5
            score_array.append((curr_score, depiction))
        
        #print(f"{entity_id}: {count} :: {total_count} ")
        
        score_array = sorted(score_array, key=lambda x: x[0], reverse=True)
        top_depictions = [(entities[entity_id]["depictions"][depiction[1]]["url"]["value"], depiction[0]) for depiction in score_array]
        if entities[entity_id]["reference_image"]["url"] not in top_depictions:
            top_depictions.insert(0, (entities[entity_id]["reference_image"]["url"],1000000))
        entities[entity_id]["top_images"] = top_depictions
        # entities[entity_id]['score_array'] = score_array
    return entities



import json
def bytes2dict(b):
    return json.loads(b.decode("utf-8"))

import time
from urllib3.exceptions import MaxRetryError

def request(query, session, tries=0, max_tries=2):
    """GET query via requests, handles exceptions and returns None if something went wrong"""
    response = None
    base_msg = f"Something went wrong when requesting for '{query}':\n"
    if tries >= max_tries:
        warnings.warn(f"{base_msg}Maximum number of tries ({max_tries}) exceeded: {tries}")
        return response
    try:
        response = session.get(query, headers={'User-Agent':'meerqat bot 0.1'})
    except requests.exceptions.ConnectionError as e:
        warnings.warn(f"{base_msg}requests.exceptions.ConnectionError: {e}")
    except MaxRetryError as e:
        warnings.warn(f"{base_msg}MaxRetryError: {e}")
    except OSError as e:
        warnings.warn(f"{base_msg}OSError: {e}")
    except Exception as e:
        warnings.warn(f"{base_msg}Exception: {e}")

    if response is not None and response.status_code != requests.codes.ok:
        if response.status_code == 429:
            time.sleep(int(response.headers.get("Retry-After", 1)))
            return request(query, session, tries+1, max_tries=max_tries)
        warnings.warn(f"{base_msg}status code: {response.status_code}")
        response = None

    return response


def query_image(title, session):
    # query images URL, categories and description
    # note: it might be better to batch the query but when experimenting with
    # batch size as low as 25 I had to deal with 'continue' responses...
    query = COMMONS_REST_TITLE.format(titles=title)
    ##print(query)
    response = request(query, session)
    if not response:
        return None
    result = bytes2dict(response.content)['query']['pages']
    # get first (only) value
    result = next(iter(result.values()))
    ##print(result)
    imageinfo = result.get('imageinfo', [{}])[0]
    image_categories = [c.get('title') for c in result['categories']] if 'categories' in result else None
    # filter metadata
    extmetadata = imageinfo.get('extmetadata', {})
    extmetadata.pop('Categories', None)
    # TODO add some preference rules according to extmetadata["LicenseShortName"]
    # not sure how the description of an image is metadata but anyway, I fount it there...
    imageDescription = extmetadata.pop('ImageDescription', {})
    image = {
        "categories": image_categories,
        "url": imageinfo.get("url"),
        "description": imageDescription,
        "extmetadata": extmetadata,
        "title": result.get("title")
    }
    return image
# session= requests.Session()
# title = "File:Barack Obama foreign trips.png|File:Women for Obama luncheon September 23, 2004.png"
# images = query_image(title, session)


#


# images



def get_all_images(entitiess):
    print(entitiess)
    temp_entities = {}
    for entity in entitiess.keys():
        temp_entities.update({entity:{}})
    temp_entities = update_from_data(temp_entities)
    temp_entities = set_reference_images(temp_entities)
    temp_entities = get_depictions_in_reference_image(temp_entities)
    temp_entities = update_from_commons_sparql(temp_entities)
    temp_entities = get_top_images(temp_entities , 5)
    for entity in entitiess.keys():
        # during iteration there were some entities with no images or reference images so we are removing them
        try:
            # print(temp_entities[entity]['top_images'])
            entitiess[entity] = temp_entities[entity]['top_images']
        except:
            # print(entity)
            entitiess[entity] = []
    return temp_entities

def get_top5_images(entitiess):
    print(entitiess)
    temp_entities = {}
    for entity in entitiess.keys():
        temp_entities.update({entity:{}})
    temp_entities = update_from_data(temp_entities)
    temp_entities = set_reference_images(temp_entities)
    temp_entities = get_depictions_in_reference_image(temp_entities)
    temp_entities = update_from_commons_sparql(temp_entities)
    temp_entities = get_top_images(temp_entities , 5)
    for entity in entitiess.keys():
        # during iteration there were some entities with no images or reference images so we are removing them
        try:
            # print(temp_entities[entity]['top_images'])
            entitiess[entity] = temp_entities[entity]['top_images']
        except:
            # print(entity)
            entitiess[entity] = []
    return entitiess



def get_canonical_link( url):
    # Create a session object for making requests

    if "/wiki/" not in url :
        print("No Canonical Link Found.")
        print(url)
        return tuple((url , None))
    session = requests.Session()

    # Make a GET request to the URL
    
    response = session.get(url)

    # Create a BeautifulSoup object from the response HTML using the 'lxml' parser
    soup = BeautifulSoup(response.text, "lxml")

    # Find the <link rel="canonical"> tag within the <head> section
    canonical_link = soup.head.find("link", rel="canonical")

    # Extract the value of the "href" attribute
    if canonical_link:
        href = canonical_link.get("href")
        # print("Canonical Link:", href)
        session.close()
        return tuple((url, href))
    else:
        print(url)
        print("No Canonical Link Found.")
        session.close()
        return tuple((url , None))

def parallel_process(urls , function):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(function, url) for url in urls]
        # Get the results as they become available
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
    return results
    # Close the session after making the request

# from bs4 import BeautifulSoup
# import requests
# def get_date(url):
#     ''''''
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'lxml')
#     date_td = soup.find("td", id="fileinfotpl_date")
#     if date_td is None:
#         return None
#     next_td = date_td.find_next_sibling("td")

#     # Extract the text of the next <td> element
#     if next_td:
#         text = next_td.get_text(strip=True)
#         return text
#     else:
#         return None



# def get_date_for_all_images(entities):
#     for entity_id in entities:
#         for type in RESERVED_IMAGES.append('reference_image'):
#             # just to check because reference image has different construction
#             if type == 'reference_image':
#                 if 'reference_image' not in entities[entity_id]:
#                     continue
#                 image_url = entities[entity_id]['reference_image']['url']
#                 pre='https://commons.wikimedia.org/wiki/File:'
#                 file_name = image_url.split('/')[-1]
#                 url = pre+file_name
#                 date = get_date(url)
#                 if date:
#                     entities[entity_id]['reference_image']["date"] = date
#                     break
#             else:
#                 if type not in entities[entity_id]:
#                     continue
#                 for image_url in entities[entity_id][type]:
#                     pre='https://commons.wikimedia.org/wiki/File:'
#                     file_name = image_url.split('/')[-1]
#                     url = pre+file_name
#                     date = get_date(url)
#                     ##print(url)
#                     if date:
#                         entities[entity_id][type][image_url]["date"] = date
#                         break





from datetime import datetime

def get_date_for_all_images(entities):
    # getting all the images link in the dict
    image_url = set()
    images_to_date = {}
    RESERVED_IMAGES_TEMP = RESERVED_IMAGES.copy()
    RESERVED_IMAGES_TEMP.append('reference_image')
    for entity_id in entities:

        for type in RESERVED_IMAGES_TEMP:
            # just to check because reference image has different construction
            if type == 'reference_image':
                if 'reference_image' not in entities[entity_id]:
                    continue
                image_url.add(entities[entity_id]['reference_image']['url'])
            else:
                if type not in entities[entity_id]:
                    continue
                for url in entities[entity_id][type]:
                    image_url.add(url)
    results = query_sparql_entities(COMMONS_IMG_DATE_QUERY, COMMONS_SPARQL_ENDPOINT, image_url, is_links=True)
    for result in results :
        images_to_date[result['urls']['value']] = result['date']['value']
    for key in images_to_date:
        date = images_to_date[key]
        date = datetime.fromisoformat(date).date().year
        images_to_date[key] = date
    ##print(images_to_date)
    for entity_id in entities:
        for type in RESERVED_IMAGES_TEMP:
            # just to check because reference image has different construction
            if type == 'reference_image':
                if 'reference_image' not in entities[entity_id]:
                    continue
                image_url = entities[entity_id]['reference_image']['url']
                if image_url in images_to_date:
                    entities[entity_id]['reference_image']["date"] = images_to_date[image_url]
            else:
                if type not in entities[entity_id]:
                    continue
                for image_url in entities[entity_id][type]:
                    if image_url in images_to_date:
                        entities[entity_id][type][image_url]["date"] = images_to_date[image_url]



def get_categories_for_all_images(entities):

    all_files=set()
    for entity in entities:
        if 'reference_image' in entities[entity]:
            all_files.add(entities[entity]['reference_image']['url'])
        for depictions in entities[entity]['depictions']:
            all_files.add(depictions['url']['value'])

    file2strings = {}
    for file in all_files:
        file2strings[file] = file.split('/')[-1]
    query_sting = "|File: ".join(file2strings.values())
    response = requests.get(COMMONS_REST_TITLE_CAT.format(query_sting))
    data = response.json()
    file_name2categories = {}
    for page in data['query']['pages']:
        file_name = page['title'].split(':')[-1]
        categories  =[]
        for category in page['categories']:
            categories.append(category['title'].split(':')[-1])
        file_name2categories[file_name] = categories
    for entity in entities:
        if 'reference_image' in entities[entity]:
            file_name = entities[entity]['reference_image']['url'].split('/')[-1]
            if file_name in file_name2categories:
                entities[entity]['reference_image']['categories'] = file_name2categories[file_name]
        for depictions in entities[entity]['depictions']:
            file_name = depictions['url']['value'].split('/')[-1]
            if file_name in file_name2categories:
                depictions['categories'] = file_name2categories[file_name]
                
    return entities





# import requests
# from PIL import Image
# from io import BytesIO

# for entity_id in entities:
#     images = []
#     for image_url in entities[entity_id]['top_10']:
#         ##print(image_url)
#         headers = {
#     'User-Agent': 'Wget/1.21.1 (linux-gnu)'
# }
#         response = requests.get(image_url, headers=headers)
#         img = Image.open(BytesIO(response.content))
#         images.append(img)

#      # Calculate the total width and height of the collage
#     collage_width = max(img.width for img in images)
#     collage_height = sum(img.height for img in images)

#     # Create a blank canvas for the collage
#     collage = Image.new("RGB", (collage_width, collage_height))

#     # Paste the images onto the canvas
#     y_offset = 0
#     for img in images:
#         collage.paste(img, (0, y_offset))
#         y_offset += img.height

#     collage.save(f"./images/collage_{entities[entity_id]['entityLabel']['value']}.jpg")