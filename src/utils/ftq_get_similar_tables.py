# # get url to table id mapping
# import json# now we need to get the table ids for each url and the table array from 
# import pandas as pd
# import nltk
# from nltk.util import ngrams
# from nltk.tokenize import word_tokenize
# from itertools import product
# from tqdm import tqdm
# from bs4 import BeautifulSoup
# import os

# BASE_DIR = "/home/suyash/temp_testing_mmtabqa_upload" # IMPORTANT: Update this path.
# DATASET_TYPE = "fetaqa" # IMPORTANT: Update this




# with open(os.path.join(BASE_DIR, "asyncio_outputs", "url_to_table_id.json"), "r") as f:
#     url_to_table_id = json.load(f)
# URL_template = 'https://en.wikipedia.org/w/api.php?action=query&titles={title}&prop=revisions&rvlimit=500&rvstart=2020-09-09T07%3A59%3A00Z&rvdir=older&format=json&redirects=1'


# file_path = os.path.join(BASE_DIR, "all_tables.jsonl") # Update this path to your JSONL file
# rev_cnt=0
# most_similar_tables = {}
# iii=0
# # Open the JSONL file and read line by line
# redirect_titles = []
# URL = "https://en.wikipedia.org/w/api.php?action=query&titles={title}&prop=revisions&rvlimit=500&rvstart=2020-05-05T07%3A59%3A00Z&rvdir=older&format=json&redirects=1"

# # now we need to get the table ids for each url and the table array from Wikipedia
# from collections import Counter

# def compute_similarity(n_grams_1, n_grams_2):
#     counter1 = Counter(n_grams_1)
#     counter2 = Counter(n_grams_2)
    
#     # Compute intersection and union for multisets
#     intersection = sum((counter1 & counter2).values())
#     union = sum((counter1 | counter2).values())
    
#     # Compute similarity
#     similarity = intersection / union if union != 0 else 0
#     return similarity

# import nltk
# import json
# import pandas as pd
# from bs4 import BeautifulSoup
# from nltk.util import ngrams

# from nltk import word_tokenize
# from tqdm import tqdm
# file_path = os.path.join(BASE_DIR, "all_tables.jsonl")
# rev_cnt=0
# different_count = 0

# with open(file_path, "r") as jsonl_file:
#     for line  in tqdm(jsonl_file) :
       
#         json_data = json.loads(line)
        
#         page_url = json_data['page_wikipedia_url']
#         feta_id = json_data['feta_id']
#         table_section_title = json_data['table_section_title']
#         page_title = json_data['table_page_title']
#         raw_title = page_url.split("en.wikipedia.org/wiki/")[1]
#         page_url = URL.format(title=raw_title)
   
#         table_1_df  = pd.DataFrame(json_data['table_array'])
#         words = set()
#         for i in range(table_1_df.shape[0]):
#             for j in range(table_1_df.shape[1]):
#                 cell_value = str(table_1_df.iloc[i,j]).lower()
#                 if len(cell_value.split())>1:
#                     n=2 
#                     ngramss = set(ngrams(word_tokenize(cell_value), n))
#                     words.update(set(ngramss))
#                 else:
#                     words.add(cell_value)
        
#         table_id_with_max_similarity = None
#         max_similarity = 0
        
#         table_id_with_max_similarity_new_strat = None
#         max_similarity_new_strat = 0 # Trying out if different results with a newer stratagey
        
#         if page_url in url_to_table_id:
#             for table_ids in url_to_table_id[page_url]:
#                 try: 
#                     table_2_df = pd.read_html(os.path.join(BASE_DIR, "tables", f"{table_ids}.html"))[0]

#                     words2 = set()

#                     for i in range(table_2_df.shape[0]):
#                         for j in range(table_2_df.shape[1]):
#                             cell_value = str(table_2_df.iloc[i,j]).lower()

#                             if len(cell_value.split())>1:
#                                 n=2
#                                 ngras = set(ngrams(word_tokenize(cell_value), n))
#                                 words2.update(set(ngras))

#                             else :
#                                 words2.add(cell_value)

#                     similarity = len(words.intersection(words2))/len(words.union(words2))
#                     if similarity>max_similarity:
#                         max_similarity = similarity
#                         table_id_with_max_similarity = table_ids
                    
#                     different_strat_similarity = compute_similarity(words, words2)
#                     if different_strat_similarity>max_similarity_new_strat:
#                         max_similarity_new_strat = different_strat_similarity
#                         table_id_with_max_similarity_new_strat = table_ids
                    
#                     if table_id_with_max_similarity != table_id_with_max_similarity_new_strat:
#                         different_count += 1
#                         print("AAAAAAAAAAAAAAAAAAAAAAAA different")
                
#                 except Exception as e:
#                     print(e)
#                     print( "Path : ", f"/home/suyash/final_repo/fetaqa_MM_cleaned/output/tables/{table_ids}.html")
#                     iii+=1 
#                     continue
#         else : 
#             print("not found")
#             print(page_url)
#         if table_id_with_max_similarity is None:
#             print("not found")
#             print(page_url)
#         most_similar_tables[feta_id] =(page_url,table_id_with_max_similarity, max_similarity) 

# print(different_count)



# # json.dump(most_similar_tables,open("/home2/jainit/Hybrid_QA_MM/outputs_new_date/most_similar_tables.json","w"))
# with open(os.path.join(BASE_DIR, "asyncio_outputs", "most_similar_tables.json"), "w") as f:
#     json.dump(most_similar_tables, f, indent=4)

# # open the  html of the table and get table_d:link_list for each cell 
# table_links = {}
# for feta_id, table_info in tqdm(most_similar_tables.items()):
#     page_url = table_info[0]
#     table_id = table_info[1]
#     if table_id is None:
           
#         continue
#     with open(os.path.join(BASE_DIR, "tables", f"{table_ids}.html"), "r") as f:
#         html = f.read()
#         soup = BeautifulSoup(html, 'lxml')
#         tables = soup.find_all('table')
#         for table in tables:
#             rows = table.find_all('tr')
#             for row in rows:
#                 cells = row.find_all('td')
#                 for cell in cells:
#                     links = cell.find_all('a')
#                     for link in links:
#                         table_links.setdefault(table_id, {})
#                         if link.get('href') is not None and  "/wiki/" in link.get('href') and "redlink=1" not in link.get('href') and "Special:Upload?wpDestFile" not in link.get('href'):
#                             table_links[table_id][ link.get('href')] = link.text

# # clean the table links
# for table_id in table_links:
#     for link in table_links[table_id]:
#         table_links[table_id][link]= table_links[table_id][link].lower().strip().replace("\n", " ").replace("\t", " ").replace("\r", " ").replace("  ", " ")
# with open(os.path.join(BASE_DIR, "asyncio_outputs", "table_links.json"), "w") as f:
#     # json.dump(table_links,open("/home2/jainit/Hybrid_QA_MM/outputs_new_date/table_links.json","w"))
#     # print("Wrote table links to file")
#     f.write(json.dumps(table_links, indent=4))

# # file_path = "/home2/jainit/Hybrid_QA_MM/outputs_new_date/all_tables.jsonl"
# file_path = os.path.join(BASE_DIR, "all_tables.jsonl") # Update this path to your JSONL file
# new_file_path = os.path.join(BASE_DIR, "all_tables_new.jsonl") # Update this path to your new JSONL file
# f  = open(new_file_path, "w")
# i=0
# j=0 
# links_not_in_html = []
# strings_to_links = {}
# all_links = set()

# with open(file_path, "r") as jsonl_file:
#     for line  in tqdm(jsonl_file) :
        
#         json_data = json.loads(line)
        
#         if DATASET_TYPE == "fetaqa":
#             page_url = json_data['page_wikipedia_url']
#         else:
#             page_url = json_data['url']
#         if DATASET_TYPE == "fetaqa":
#             table = json_data['table_array']
#         else:
#             table = json_data['table']
        
        
#         cell_links = json_data['cells_to_link']
#         ttable_id  = json_data['table_id']
#         # print(cell_links)
#         # now we get the table_id for this fetad_id adn get teh table and then get the links for each cell
#         if str(json_data['table_id']) not in most_similar_tables:
#             # print("WTF!", json_data['table_id'])
#             continue
#         table_id = str(most_similar_tables[str(json_data['table_id'])][1])
#         new_tables = table.copy()
#         if table_id is None:
#             continue
#         strings_to_links.setdefault(ttable_id, {})
#         if table_id not in table_links:
#             # print("WTF!", table_id)
#             # print(table , cell_links)
#             for row_id, row in enumerate(table):
#                 for cell_id , cell in enumerate(row):
#                     cell = cell.lower().strip().replace("\n", " ").replace("\t", " ").replace("\r", " ").replace("  ", " ")
#                     if cell is not None and cell != "":

#                         j+=1
#                         # print( ttable_id)
#                         links = cell_links[row_id][cell_id]
#                         if len(links)>0:
                            
#                             for link in links:
#                                 link_text = link.split('/')[-1].replace("_", " ").lower().strip().replace("\n", " ").replace("\t", " ").replace("\r", " ").replace("  ", " ")
#                                 if link_text != "" and link_text in cell:
#                                     new_tables[row_id][cell_id] = cell.replace(link_text, f"{{LINK{{{link_text}}}{{{link}}}}}")
#                                     strings_to_links[ttable_id][link_text] = link
#                                     all_links.add(link)
#                                     i+=1 
#                                     # print("WTF!")
#                                     link_found = True
#                                     # break
#                                     links_not_in_html.append((cell, link_text, link))
            
#             json_data['table_new'] = new_tables
#             f.write(json.dumps(json_data)+"\n")
            

#             continue
#         # else :
#         #     continue
#         for row_id , row in enumerate(table):
#             for cell_id , cell in enumerate(row):
#                 j+=1
                
#                 link_found = False
#                 old = cell
#                 cell = cell.lower().strip().replace("\n", " ").replace("\t", " ").replace("\r", " ").replace("  ", " ")
#                 if cell!= "":
#                     for link, text in table_links[table_id].items():
#                         if text != "" and text in cell: # and link in hybridQA [row_id][cell_id]
#                             # print(row_id, cell_id, cell, text, link)
#                             new_tables[row_id][cell_id] = cell.replace(text, f"{{LINK{{{text}}}{{{link}}}}}")
#                             strings_to_links[ttable_id][text] = link
#                             all_links.add(link)
#                             i+=1 
#                             link_found = True
#                             # break
#                     try: 
#                         if not link_found and cell_links[row_id][cell_id] is not None:
#                             # print("sfgd")
#                             links = cell_links[row_id][cell_id]
#                             if len(links)>0:
#                                 for link in links:
#                                     link_text = link.split('/')[-1].replace("_", " ").lower().strip().replace("\n", " ").replace("\t", " ").replace("\r", " ").replace("  ", " ")
#                                     if link_text != "" and link_text in cell:
#                                         new_tables[row_id][cell_id] = cell.replace(link_text, f"{{LINK{{{link_text}}}{{{link}}}}}")
#                                         strings_to_links[ttable_id][link_text] = link
#                                         all_links.add(link)
#                                         i+=1 
#                                         link_found = True
#                                         # break
#                                         links_not_in_html.append((cell, link_text, link))
#                     except Exception as e:
#                         # print(e)
#                         # print("row_id, cell_id", row_id, cell_id)
#                         # # print(cell_links[row_id])
#                         # # print(cell_links[row_id][cell_id])
#                         # print(cell)
#                         # print("WTF")
#                         continue
                                
                        
#         json_data['table_new'] = new_tables
#         f.write(json.dumps(json_data)+"\n")
#         # print("wriiten")
# f.close()

# all_links_new = {}
# all_links_set = set()
# for link in all_links :
#     if "/wiki/" in link and "https:" not in link  : 
#         all_links_set.add("https://en.wikipedia.org"+link)
#         all_links_new[link] = "https://en.wikipedia.org"+link
#     elif "https:" in link and "/wiki/" in link: 
#         all_links_new[link] = link
#         all_links_set.add(link)
    
# with open(os.path.join(BASE_DIR, "asyncio_outputs", "all_links_dict.json"), "w") as f:
#     json.dump(all_links_new, f, indent=4)
# with open(os.path.join(BASE_DIR, "asyncio_outputs", "strings_to_links.json"), "w") as f:
#     json.dump(strings_to_links, f, indent=4)
# with open(os.path.join(BASE_DIR, "asyncio_outputs", "all_links_set.json"), "w") as f:
#     json.dump(list(all_links_set), f, indent=4)
# # json.dump(all_links_new,open("/home2/jainit/Hybrid_QA_MM/outputs_new_date/all_links_dict.json","w"))
# # json.dump(strings_to_links,open("/home2/jainit/Hybrid_QA_MM/outputs_new_date/strings_to_links.json","w"))
# # # print(i, j)
# # json.dump(list(all_links_set),open("/home2/jainit/Hybrid_QA_MM/outputs_new_date/all_links_set.json","w"))