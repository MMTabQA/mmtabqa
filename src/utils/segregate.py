import json
from tqdm import tqdm

with open('/home/suyash/final_repo/fetaqa_MM_cleaned/category_filtered_outputs/links_to_landscapes.json','r') as f:
    landscape_json = json.load(f)

with open('/home/suyash/final_repo/fetaqa_MM_cleaned/category_filtered_outputs/links_to_seals.json','r') as f:
    seal_json = json.load(f)
with open('/home/suyash/final_repo/fetaqa_MM_cleaned/outputs/link_to_single_image.json','r',errors='ignore') as f:
    single_img = json.load(f)
with open('/home/suyash/final_repo/fetaqa_MM_cleaned/experiment_ready_dataset/image_id_to_qid.json','r') as f:
    feta = json.load(f)
with open('/home/suyash/final_repo/fetaqa_MM_cleaned/outputs/categories_seal.json','r') as f:
    categories_seal = json.load(f)
with open('/home/suyash/final_repo/fetaqa_MM_cleaned/old_outputs/filtering_outputs/entity_category.json','r') as f:
    entity_category = json.load(f)
with open('/home/suyash/final_repo/fetaqa_MM_cleaned/old_outputs/link_to_entity.json','r',errors='ignore') as f:
    link_to_entity = json.load(f)
with open('/home/suyash/final_repo/fetaqa_MM_cleaned/experiment_ready_dataset/image_id_to_wikipedia_link.json','r') as f:
    wiki_link = json.load(f)

with open('/home/suyash/final_repo/fetaqa_MM_cleaned/old_outputs/link_to_full_links.json','r',errors='ignore') as f:
    full_link = json.load(f)

with open('/home/suyash/final_repo/fetaqa_MM_cleaned/outputs/categories_single_img.json','r') as f:
    categories_single_img = json.load(f)
dict = {"human":[],
    "landscape":[],
        "seal": [],
    "flag": [],
    "logo": [],
    "coat of arms": [],
    "poster": []}
for img_id in tqdm(feta.keys()):
  try:
    hf_link = wiki_link[img_id]
    link = full_link[hf_link]
    
    entity = link_to_entity[link]
    entity_cat = entity_category[entity]
    f = 0
    for arr in entity_cat:
        if f == 1:
            break
        for w in arr:
            if w == "human":
                dict["human"].append(link)
                f = 1
                break
    if f==1:
        continue
    if link in landscape_json.keys():
        for w in landscape_json[link]:
            dict["landscape"].append(w)
    elif link in seal_json.keys():
        for url in seal_json[link]:
         

          if url in categories_seal.keys():  
            
            for imgs in categories_seal[url]:

                k = url
                parts = imgs.split("/")
                last_part = parts[-1]

                category = last_part.split(":")[-1]
                
                
                if ("seal" in category.lower()) or ("emblem" in category.lower()) or ("seal" in k.lower()) or ("emblem" in k.lower())  :
                    dict["seal"].append(k)
                    break
                elif ("flag" in category.lower()) or ("flag" in k.lower()):
                  
                    dict["flag"].append(k)
                   
                    break
                elif ("logo" in category.lower()) or ("logo" in k.lower()):
                    dict["logo"].append(k)
                    break
                elif (("coats_of_arm") in category.lower()) or (("coat_of_arm") in category.lower()) or (("coats_of_arm") in k.lower()) or (("coat_of_arm") in k.lower()):
                    # if url == "https://en.wikipedia.org/wiki/File:Escudo_de_la_Provincia_de_Santiago_del_Estero.svg":
                    #     print("mkc")
                    dict["coat of arms"].append(k)
                    break
                #    elif "association" in (category.lower() or k.lower()):
                #        dict["logo"].append(k)
                #        break
                elif "poster" in category.lower() or "film" in category.lower() or "movie" in category.lower() or "book" in category.lower() or "novel" in category.lower() or "advertisement" in category.lower() or "music" in category.lower() or "poster" in k.lower() or "film" in k.lower() or "movie" in k.lower() or "book" in k.lower() or "novel" in k.lower() or "advertisement" in k.lower() or "music" in k.lower():
                    dict["poster"].append(k)    
                    break
    elif single_img[link] in categories_single_img.keys(): #It'll be in single_img_json
        img_url = single_img[link]
        categories = categories_single_img[img_url]
        # print(img_url)
        # print(categories)
        for img in categories:
            # for img in url:
                k = img_url
                parts = img.split("/")
                last_part = parts[-1]

                category = last_part.split(":")[-1]
                #    if(k=="https://en.wikipedia.org/wiki/File:1934_fifa_worldcup_poster.jpg"):
                #        if "logo" in (k.lower()):
                #         #    print(k.lower())
                #            print("mkc")
                
                if ("seal" in category.lower()) or ("emblem" in category.lower()) or ("seal" in k.lower()) or ("emblem" in k.lower())  :
                    dict["seal"].append(k)
                    break
                elif ("flag" in category.lower()) or ("flag" in k.lower()):
                  
                    dict["flag"].append(k)
                   
                    break
                elif ("logo" in category.lower()) or ("logo" in k.lower()):
                    dict["logo"].append(k)
                    break
                elif (("coats_of_arm") in category.lower()) or (("coat_of_arm") in category.lower()) or (("coats_of_arm") in k.lower()) or (("coat_of_arm") in k.lower()):
                    # if url == "https://en.wikipedia.org/wiki/File:Escudo_de_la_Provincia_de_Santiago_del_Estero.svg":
                    #     print("mkc")
                    dict["coat of arms"].append(k)
                    break
                #    elif "association" in (category.lower() or k.lower()):
                #        dict["logo"].append(k)
                #        break
                elif "poster" in category.lower() or "film" in category.lower() or "movie" in category.lower() or "book" in category.lower() or "novel" in category.lower() or "advertisement" in category.lower() or "music" in category.lower() or "series" in category.lower() or "poster" in k.lower() or "film" in k.lower() or "movie" in k.lower() or "book" in k.lower() or "novel" in k.lower() or "advertisement" in k.lower() or "music" in k.lower() or "series" in k.lower():
                    dict["poster"].append(k)    
                    break
    else:
        img_url = single_img[link]
        # categories = categories_single_img[img_url]
        # print(img_url)
        # print(categories)
        if True:
            # for img in url:
                k = img_url
                img="kk"
                parts = img.split("/")
                last_part = parts[-1]

                category = last_part.split(":")[-1]
                #    if(k=="https://en.wikipedia.org/wiki/File:1934_fifa_worldcup_poster.jpg"):
                #        if "logo" in (k.lower()):
                #         #    print(k.lower())
                #            print("mkc")
                
                if ("seal" in category.lower()) or ("emblem" in category.lower()) or ("seal" in k.lower()) or ("emblem" in k.lower())  :
                    dict["seal"].append(k)
                    
                elif ("flag" in category.lower()) or ("flag" in k.lower()):
                  
                    dict["flag"].append(k)
                   
                    
                elif ("logo" in category.lower()) or ("logo" in k.lower()):
                    dict["logo"].append(k)
                    
                elif (("coats_of_arm") in category.lower()) or (("coat_of_arm") in category.lower()) or (("coats_of_arm") in k.lower()) or (("coat_of_arm") in k.lower()):
                    if url == "https://en.wikipedia.org/wiki/File:Escudo_de_la_Provincia_de_Santiago_del_Estero.svg":
                        print("mkc")
                    dict["coat of arms"].append(k)
                    
                #    elif "association" in (category.lower() or k.lower()):
                #        dict["logo"].append(k)
                #        break
                elif "poster" in category.lower() or "film" in category.lower() or "movie" in category.lower() or "book" in category.lower() or "novel" in category.lower() or "advertisement" in category.lower() or "music" in category.lower() or "series" in category.lower() or "poster" in k.lower() or "film" in k.lower() or "movie" in k.lower() or "book" in k.lower() or "novel" in k.lower() or "advertisement" in k.lower() or "music" in k.lower() or "series" in k.lower():
                    dict["poster"].append(k)   
  except Exception as e:
      print("Error in link -", e) 
for key,val in dict.items():
    dict[key] = list(set(val))
    print(f'Count of {key} = {len(dict[key])}')

with open("/home/suyash/final_repo/fetaqa_MM_cleaned/outputs/dump.json", 'w') as file:
        json.dump(dict, file, indent=4)
