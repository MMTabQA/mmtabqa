# get url to table id mapping
import json# now we need to get the table ids for each url and the table array from 
import pandas as pd
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from itertools import product
from tqdm import tqdm
from bs4 import BeautifulSoup
url_to_table_id = json.load(open("/home2/jainit/Hybrid_QA_MM/outputs_new_date/url_to_table_id.json", "r"))
URL_template = 'https://en.wikipedia.org/w/api.php?action=query&titles={title}&prop=revisions&rvlimit=500&rvstart=2020-09-09T07%3A59%3A00Z&rvdir=older&format=json&redirects=1'


file_path = "/home2/jainit/Hybrid_QA_MM/outputs_new_date/all_tables.jsonl"
rev_cnt=0
most_similar_tables = {}
iii=0
# Open the JSONL file and read line by line
redirect_titles = []
with open(file_path, "r") as jsonl_file:
    for line  in tqdm(jsonl_file) :
       
        json_data = json.loads(line)
        
        # Now you can work with the JSON data as a Python dictionary
        # print(json_data)  # You can replace this with your processing logic
        page_url = json_data['url']
        # print(page_url)
        
        page_url = URL_template.format(title = page_url.split("en.wikipedia.org/wiki/")[1])
        id = json_data['table_id']
        # raw_title = page_url.split("en.wikipedia.org/wiki/")[1]
        table_1_df  = pd.DataFrame(json_data['table'])
        words = set()
        for i in range(table_1_df.shape[0]):
            for j in range(table_1_df.shape[1]):
                cell_value = str(table_1_df.iloc[i,j]).lower()
                if len(cell_value.split())>1:
                    n=2 
                    ngramss = set(ngrams(word_tokenize(cell_value), n))
                    words.update(set(ngramss))
                else :
                    words.add(cell_value)
                
        
        table_id_with_max_similarity = None
        max_similarity = 0
        
        if page_url in url_to_table_id:
            for table_ids in url_to_table_id[page_url]:
                try : 
                    # print(f"/home2/jainit/Hybrid_QA_MM/outputs_new_date/tables/{table_ids}.html")
                    table_2_df = pd.read_html(f"/home2/jainit/Hybrid_QA_MM/outputs_new_date/tables/{table_ids}.html")[0]
                    # print("hi")
                    words2 = set()
                    for i in range(table_2_df.shape[0]):
                        for j in range(table_2_df.shape[1]):
                            cell_value = str(table_2_df.iloc[i,j]).lower()
                            if len(cell_value.split())>1:
                                n=2
                                ngras = set(ngrams(word_tokenize(cell_value), n))
                                words2.update(set(ngras))
                            else :
                                words2.add(cell_value)
                    similarity = len(words.intersection(words2))/len(words.union(words2)) #jaccard
                    if similarity>max_similarity:
                        max_similarity = similarity
                        table_id_with_max_similarity = table_ids
                except Exception as e:
                    print(e)
                    print( "Path : ", f"/home2/jainit/Hybrid_QA_MM/outputs_new_date/tables/{table_ids}.html")
                    iii+=1 
                    continue

        else : 
            print(page_url)
        most_similar_tables[id] =(page_url,table_id_with_max_similarity, max_similarity) 


json.dump(most_similar_tables,open("/home2/jainit/Hybrid_QA_MM/outputs_new_date/most_similar_tables.json","w"))

# open the  html of the table and get table_d:link_list for each cell 
table_links = {}
for feta_id, table_info in tqdm(most_similar_tables.items()):
    page_url = table_info[0]
    table_id = table_info[1]
    if table_id is None:
           
        continue
    with open(f"/home2/jainit/Hybrid_QA_MM/outputs_new_date/tables/{table_id}.html", "r") as f:
        html = f.read()
        soup = BeautifulSoup(html, 'lxml')
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                for cell in cells:
                    links = cell.find_all('a')
                    for link in links:
                        table_links.setdefault(table_id, {})
                        if link.get('href') is not None and  "/wiki/" in link.get('href') and "redlink=1" not in link.get('href') and "Special:Upload?wpDestFile" not in link.get('href'):
                            table_links[table_id][ link.get('href')] = link.text

# clean the table links
for table_id in table_links:
    for link in table_links[table_id]:
        table_links[table_id][link]= table_links[table_id][link].lower().strip().replace("\n", " ").replace("\t", " ").replace("\r", " ").replace("  ", " ")
json.dump(table_links,open("/home2/jainit/Hybrid_QA_MM/outputs_new_date/table_links.json","w"))

file_path = "/home2/jainit/Hybrid_QA_MM/outputs_new_date/all_tables.jsonl"
f  = open("/home2/jainit/Hybrid_QA_MM/outputs_new_date/all_tables_new.jsonl", "w")
i=0
j=0 
links_not_in_html = []
strings_to_links = {}
all_links = set()

with open(file_path, "r") as jsonl_file:
    for line  in tqdm(jsonl_file) :
        json_data = json.loads(line)
        page_url = json_data['url']
        table = json_data['table']
        cell_links = json_data['cells_to_link']
        ttable_id  = json_data['table_id']
        # print(cell_links)
        # now we get the table_id for this fetad_id adn get teh table and then get the links for each cell
        if str(json_data['table_id']) not in most_similar_tables:
            # print("WTF!", json_data['table_id'])
            continue
        table_id = str(most_similar_tables[str(json_data['table_id'])][1])
        new_tables = table.copy()
        if table_id is None:
            continue
        strings_to_links.setdefault(ttable_id, {})
        if table_id not in table_links:
            # print("WTF!", table_id)
            # print(table , cell_links)
            for row_id, row in enumerate(table):
                for cell_id , cell in enumerate(row):
                    cell = cell.lower().strip().replace("\n", " ").replace("\t", " ").replace("\r", " ").replace("  ", " ")
                    if cell is not None and cell != "":

                        j+=1
                        # print( ttable_id)
                        links = cell_links[row_id][cell_id]
                        if len(links)>0:
                            
                            for link in links:
                                link_text = link.split('/')[-1].replace("_", " ").lower().strip().replace("\n", " ").replace("\t", " ").replace("\r", " ").replace("  ", " ")
                                if link_text != "" and link_text in cell:
                                    new_tables[row_id][cell_id] = cell.replace(link_text, f"{{LINK{{{link_text}}}{{{link}}}}}")
                                    strings_to_links[ttable_id][link_text] = link
                                    all_links.add(link)
                                    i+=1 
                                    # print("WTF!")
                                    link_found = True
                                    # break
                                    links_not_in_html.append((cell, link_text, link))
            
            json_data['table_new'] = new_tables
            f.write(json.dumps(json_data)+"\n")
            

            continue
        # else :
        #     continue
        for row_id , row in enumerate(table):
            for cell_id , cell in enumerate(row):
                j+=1
                
                link_found = False
                old = cell
                cell = cell.lower().strip().replace("\n", " ").replace("\t", " ").replace("\r", " ").replace("  ", " ")
                if cell!= "":
                    for link, text in table_links[table_id].items():
                        if text != "" and text in cell: # and link in hybridQA [row_id][cell_id]
                            # print(row_id, cell_id, cell, text, link)
                            new_tables[row_id][cell_id] = cell.replace(text, f"{{LINK{{{text}}}{{{link}}}}}")
                            strings_to_links[ttable_id][text] = link
                            all_links.add(link)
                            i+=1 
                            link_found = True
                            # break
                    try: 
                        if not link_found and cell_links[row_id][cell_id] is not None:
                            # print("sfgd")
                            links = cell_links[row_id][cell_id]
                            if len(links)>0:
                                for link in links:
                                    link_text = link.split('/')[-1].replace("_", " ").lower().strip().replace("\n", " ").replace("\t", " ").replace("\r", " ").replace("  ", " ")
                                    if link_text != "" and link_text in cell:
                                        new_tables[row_id][cell_id] = cell.replace(link_text, f"{{LINK{{{link_text}}}{{{link}}}}}")
                                        strings_to_links[ttable_id][link_text] = link
                                        all_links.add(link)
                                        i+=1 
                                        link_found = True
                                        # break
                                        links_not_in_html.append((cell, link_text, link))
                    except Exception as e:
                        # print(e)
                        # print("row_id, cell_id", row_id, cell_id)
                        # # print(cell_links[row_id])
                        # # print(cell_links[row_id][cell_id])
                        # print(cell)
                        # print("WTF")
                        continue
                                
                        
        json_data['table_new'] = new_tables
        f.write(json.dumps(json_data)+"\n")
        # print("wriiten")
f.close()

all_links_new = {}
all_links_set = set()
for link in all_links :
    if "/wiki/" in link and "https:" not in link  : 
        all_links_set.add("https://en.wikipedia.org"+link)
        all_links_new[link] = "https://en.wikipedia.org"+link
    elif "https:" in link and "/wiki/" in link: 
        all_links_new[link] = link
        all_links_set.add(link)
    
json.dump(all_links_new,open("/home2/jainit/Hybrid_QA_MM/outputs_new_date/all_links_dict.json","w"))
json.dump(strings_to_links,open("/home2/jainit/Hybrid_QA_MM/outputs_new_date/strings_to_links.json","w"))
# print(i, j)
json.dump(list(all_links_set),open("/home2/jainit/Hybrid_QA_MM/outputs_new_date/all_links_set.json","w"))