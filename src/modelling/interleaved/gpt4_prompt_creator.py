#import openai
import os
import json
import base64
import requests
import re
#from openai import OpenAI

# openai.api_key = 

# model: str = "gpt-4o"
# temperature: float = 0.0


def get_all_image_tags(text):
    pattern = r"\{IMG-\{.*?\}\}"
    matches = re.findall(pattern, text)
    return matches

def get_base_64_img(image):
    """
    Converts an image from either a local file or a URL to base64 encoding.

    Parameters:
    - image (str): The image source, which can be a local file path or a URL.

    Returns:
    str: Base64-encoded representation of the image.
    """

    # Check if the image is a local file or a URL
    if "http" not in image:
        # Local File: Read the binary content of the file, encode it in base64, and decode as UTF-8
        base64_image = base64.b64encode(open(image, "rb").read()).decode('utf-8')
    else:
        # File on the Web: Fetch the image content from the URL, encode it in base64, and decode as UTF-8
        response = requests.get(image)
        base64_image = base64.b64encode(response.content).decode('utf-8')

    # Return the base64-encoded image
    return base64_image

#dataset
base_url_image= "/media/vivek/c33fd89b-a307-4208-a045-64d021572535/New_suyash.mthur/refactored_images/"
dataset = "WikiSQL"#"WikiTableQuestions", WikiSQL
question_type="test_implicit_questions.jsonl"

image_id_to_image_path=[]
with open(os.path.join("/home/vivek/harshita/modelling", "image_id_to_refined_path.json")) as f:
    image_id_to_image_path = json.load(f)
    
question_tables={}
with open(os.path.join(f"/home/vivek/harshita/{dataset}/exp_ready/experiment_ready_dataset", "tables.jsonl")) as f:
        lines = f.readlines()
        for line in lines:
            line_json = json.loads(line)
            question_tables[line_json["table_id"]] = line_json
#        print(line)


################WTQ_specific####################
import pandas as pd
with open("/home/vivek/harshita/WikiTableQuestions/exp_ready/experiment_ready_dataset/table-metadata.tsv") as f:
    WTQ_METADATA = pd.read_csv(f, sep="\t")

def get_wtq_metadata(table_id):
    table_metadata = WTQ_METADATA[WTQ_METADATA["contextId"] == table_id.replace("WTQ/", "")]

    page_title = table_metadata["title"].values[0]
    headers = table_metadata['headers'].values[0]
    return page_title, headers
################################################

def get_metadata(table_id):
    if dataset == "WikiTableQuestions":
        page_title, headers = get_wtq_metadata(table_id)
    elif dataset == "WikiSQL":
        page_title = question_tables[table_id]["page_title"]
        headers = question_tables[table_id]["section_title"]
    elif dataset == "FetaQA":
        page_title = question_tables[table_id]["table_page_title"]
        headers = question_tables[table_id]["table_section_title"]
    if pd.isna(headers):
        prompt = f"Table related to the {page_title}."
    else:
        prompt = f"Table related to {headers} in context of {page_title}."
    return prompt

test_questions={}
with open(os.path.join(f"/home/vivek/gpt_remaining/{dataset}/", question_type)) as f: #/home/vivek/harshita/train_test_split/ /home/vivek/gpt_test_samples/WikiTableQuestions/
    lines = f.readlines()
    for line in lines:
        line_json = json.loads(line)
        test_questions[line_json["question_id"]] = line_json
# with open(os.path.join("/home/vivek/harshita/WikiSQL/final_dataset_filtered", "image_id_to_image_path.json")) as f:
#     image_id_to_image_path = json.load(f)



# response = openai.ChatCompletion.create(**args)
# #client.chat.completion.create( model="gpt-4", messages=[{"role": "user", "content": "Generate a 3 sentence story about friendship"}] ) 

# print(response.choices[0].message.content)
    TEXT_PROMPT_FETA = """Answer in a sentence, using the table data given. The table consists of data in the form of text and images. Each row of the table has been represented using [] with data for each column in the row separated by a semi-colon.
 In the table, some entities (mentioned in text form originally) have been replaced by images that represent them. Based upon the context of the table while using real-world knowledge, your task is to identify the entities corresponding to the images in the table and answer the question. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question by identifying the relevant entities represented by images using the context of the table and the question. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for disambiguating the entities and answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

You are also provided with some question-answer examples for better understanding the format of providing the answer:

Example 1:
Table context: Table related to NHL awards in context of 2013–14 NHL season.\n\nQuestion: Which teams were competing for the Stanley Cup in the 2013-14 NHL season?\nStep 1 : The Stanley Cup is the silver-coloured cup represented in the first row, Award column. In the same row under the reciepient's column, we can see a Black-coloured logo with LA written on it, which is the logo for <>.Also in the runners-up column, we can see a Blue-coloured logo with "New-York Rangers" written on it. Thus, we can conclude that The Los Angeles Kings won the Stanley Cup, defeating the New York Rangers.\n\nStep 2:\nThe Los Angeles Kings won the Stanley Cup, defeating the New York Rangers.

Example 2:
Table context: Table related to International competitions in context of Debbie Marti.\n\nQuestion: In which city was the 1991 World Championships held and what distance did Debbie Marti achieve to qualify?\nStep 1: As we can see in the 5th row, the Competiton represented in the Competitions column by a Blue-coloured logo is the World Championships. In the same row, under the venue column, we can see a collage of pictures of the prominent buildings from Tokyo. Thus We can conclude that the venue of the competiton was Tokyo. Also, in the column "Notes", we can see that Debbie Marti qualified with 1.86m.\n\nStep 2: At the 1991 World Championships in Tokyo, Debbie Marti qualified with 1.86 m.  

Example 3:
Table context: Table related to Awards and nominations in context of Project Gutenberg (film).\n\nQuestion: What awards did Project Gutenberg win at the 38th Hong Kong Film Awards?\nStep 1 : The answer to the question can be found by looking at the column titled "Award" in the table. We can infer that there are seven rows in the table, each corresponding to an award won by Project Gutenberg. The categories listed are (Best Film, Best Director, Best Screenplay, Best Cinematography, Best Film Editing, Best Art Direction, and Best Costume Make Up Design) exactly match up to the categories listed in the "Award" column. So, to find the answer, you would need to look for each of these categories in the "Award" column and see which movie title is listed next to it.\n\nStep 2:Project Gutenberg won seven awards at the 38th Hong Kong Film Awards, in the categories Best Film, Best Director, Best Screenplay, Best Cinematography, Best Film Editing, Best Art Direction, and Best Costume Make Up Design.

Example 4:
Table context: Table related to Awards and nominations in context of Mike Cahill (director).\n\nQuestion: What film won the Alfred P. Sloan Prize at the Sundance Film Festival in 2014?\nStep 1 : Look at the "Year" column and find the year 2014. Then, look at the "Award" column for that row. If it says "Alfred P. Sloan Prize", then the movie title in the "Film" column for that row is the answer. In the table you described, on the row where "Year" is 2014, "Award" is "Alfred P. Sloan Prize", and "Film" is "I Origins".\n\nStep 2:\nCahill's film I Origins again won the Alfred P. Sloan Prize at the 2014 Sundance Film Festival, his second time receiving the award.

Example 5:
Table context: Table related to Home attendances in context of 2012–13 Everton F.C. season.\n\nQuestion: How did Everton F.C. do against Manchester United and Tottenham Hotspur during their 2012-13 season?\nStep 1 : Look for Manchester United and Tottenham Hotspur on the “Opponent” column.  Look at the corresponding “Score” for each team. For Manchester United, the score is  1-0 in favor of Everton. For Tottenham Hotspur, the score is 2-1 in favor of Everton. Therefore, Everton won against both Manchester United and Tottenham Hotspur.\n\nStep 2:\nEverton F.C. won over Manchester United in the first game of the season with 1–0, defeated Tottenham Hotspur 2–1, and defeated Manchester City 2–0 in the Premier League.

Example 6:
Table context: Table related to International competitions in context of Süreyya Ayhan.\n\nQuestion: How did Sureyya Ayhan fare at the 2003 World Championships?\nStep 1 : Look for "2003" in the "Year" column. Look across that row to the "Competition" column. It should say "World Championships". In the "Event" column, it shows "1500 m". Finally, under the "Position" column, it shows "2nd", indicating that Süreyya Ayhan won a silver medal.\n\nStep 2:\nSüreyya Ayhan won a silver medal in the 1500 m of the 2003 World Championships.

Example 7:
Table context: Table related to Grammy Awards in context of Roberta Flack.\n\nQuestion: When and for which songs did the singer Roberta Flack win Grammy Awards for Record of they Year?\nStep 1 : Looking at the table under the "Year" column, you can see 1973 listed twice.  In the corresponding rows under "Award" it says "Record of the Year" each time.  Looking at the "Nominee / work" column for those two rows, it shows "The First Time Ever I Saw Your Face" in 1973 and "Killing Me Softly With His Song" in 1974.  This confirms that Flack won the award for these two songs in consecutive years.\n\nStep 2:\nFlack won the Grammy Award for Record of the Year on two consecutive years: "The First Time Ever I Saw Your Face" won at the 1973 Grammys as did "Killing Me Softly with His Song" at the 1974 Grammys.

Example 8:
Table context: Table related to Television series in context of Kim Jung-hyun (actor, born 1990).\n\nQuestion: What did Kim Jung-hyun do in KBS2 in 2017?\nStep 1 : Look for the year "2017" in the "Year" column. Look across that row to the "Network" column. It should say "KBS2". In the "Title" column, it shows "School 2017". This indicates that Kim Jung-hyun played in that drama in 2017 on KBS2.\n\nStep 2:\In 2017, Kim Jung-hyun played in KBS2's School 2017.

Now, based upon the examples given above, you must understand the text and images given in the table and follow the steps 1-2 to answer the question corresponding to the table represented bt the data. Try to keep the answer in active voice. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.

"""
    TEXT_PROMPT_WTQ = """Answer in a sentence, using the table data given. The table consists of data in thr form of text and images. Each row of the table has been represented using [] with data for each column in the row separated by a semi-colon.
 In the table, some entities (mentioned in text form originally) have been replaced by images that represent them. Based upon the context of the table while using real-world knowledge, your task is to identify the entities corresponding to the images in the table and answer the question. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question by identifying the relevant entities represented by images using the context of the table and the question. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for disambiguating the entities and answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

You are also provided with some question-answer examples for better understanding the format of providing the answer:

Example 1:
Table context: Table related to Fifth round proper in context of 1975–76 FA Cup.\n\nQuestion: how many games played by sunderland are listed here?\nStep 1 : We can conclude Sunderland played in 2 games. The table shows teams listed under "Home team" and "Away team" columns [column headers provide this information].  Looking across the rows, Sunderland's logo, which comprises of 2 horses to the side and a Black&White sheild in between, is listed under one of these columns twice [in the 2nd and 3rd row]. Therefore, Sunderland participated in two games.\n\nStep 2:\n2

Example 2:
Table context: Table related to Complete Formula One World Championship results in context of Playlife.\n\nQuestion: when was the benetton b198 chassis used?\nStep 1: The table shows Formula One results with a context of Playlife, possibly a constructor. As we can see, the Benetton b198 Chassis is the blue coloured supporting structure, as seen in the chassis column of the table. In the same row, there is the column year, which gives us the answer as 1998.\n\nStep 2: 1998.  

Example 3:
Table context: Table related to Defunct railroads in context of List of Washington, D.C., railroads.\n\nQuestion: was the pennsylvania railroad under the prr or the rf&p?\nStep 1 : The table shows defunct railroads in Washington D.C. The "Pennsylvania Railroad" is the golden background picture represented in the 11th row with trains visible in it. In the same row, another column named "Mark" has the abbreviation as "PRR". Thus, the Pennsylvania Railroad operated under PRR since "PRR" is its short name.\n\nStep 2:PRR

Example 4:
Table context: Table related to Schedule and results in context of 2013–14 Chicago State Cougars women's basketball team.\n\nQuestion: how many games were played against grand canyon?\nStep 1 : We can see that there are 2 instances of the grand canyon in the opponent column. One in the 20th row, where there is a purple coloured logo which says GCC, which refers to the Grand Canyon College. Another is in the 26th row. Thus, we can conclude that 2 matches were played against the grand canyon.\n\nStep 2:\n2

Example 5:
Table context: Table related to Roster|Letter winners in context of 1915 Michigan Wolverines football team.\n\nQuestion: how many players were taller and weighed more than frank millard?\nStep 1 : Frank Millard is the clean-shaved, short haired guy visible in the 5th row. His height is 5'7 and weight is 212. Thus clearly, there are only 2 players whose height and weight is more than his, one in the 2nd row and other in the 8th row.\n\nStep 2:\n2

Example 6:
Table context: Table related to Racing record|Career summary in context of Conor Daly.\n\nQuestion: the two teams who raced in 2011 are carlin motorsport and what other team?\nStep 1 : In the year column there are 2 rows which have a mention of 2011. Apart from Carlin motorsport, the other one has a green car with the logo Schmidt Motorsports on it.\n\nStep 2:\nSchmidt Motorsports

Example 7:
Table context: Table related to Regular season|Schedule in context of 1995 New York Jets season.\n\nQuestion: team that scored more than 40 points against the jets that is not the miami dolphins\nStep 1 : As clearly visible, the opponent mentioned in the 4th row, which has a Black-coloured logo written as "RAIDERS" on it scored 47 goals against the jets. The logo is of the team Oakland Raiders. Thus, We can conclude that Oakland Raiders is the other team that scored 47 goals against the jets.\n\nStep 2:\nOakland Raiders

Example 8:
Table context: Table related to Winners|By Country in context of EHF Cup Winners' Cup.\n\nQuestion: did france or croatia have a larger finals total?\nStep 1 : Under the country column, in the 5th row we can see a Blue-coloured chicken logo, with FFHANDBALL written under it. That is the logo for France's handball federation. In the 8th row, we can see and Red-Blue coloured handall logo, with the Croatia Handball federation written underneath it, which represents Croatia. Thus, we can conclude that France had more Finals Total, 4, than Croatia, 1.\n\nStep 2:\nFrance

Now, based upon the examples given above, you must understand the text and images given in the table and follow the steps 1-2 to answer the question corresponding to the table represented bt the data. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.


# """
    TEXT_PROMPT_WIKISQL = """Answer in a sentence, using the table data given. The table consists of data in thr form of text and images. Each row of the table has been represented using [] with data for each column in the row separated by a semi-colon.
 In the table, some entities (mentioned in text form originally) have been replaced by images that represent them. Based upon the context of the table while using real-world knowledge, your task is to identify the entities corresponding to the images in the table and answer the question. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question by identifying the relevant entities represented by images using the context of the table and the question. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for disambiguating the entities and answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

You are also provided with some question-answer examples for better understanding the format of providing the answer:

Example 1:
Table related to Teams and venues in context of 2004 Belarusian Premier League.\n\nQuestion:  What is the Location for Belshina?\nStep 1 : The table shows the teams and their corresponding locations. To find the location for Belshina, we need to locate Belshina in the 'Team' column. Belshina is located in row 9, and its location is given in the 'Location' column as 'Bobruisk'. Therefore, the location for Belshina is Bobruisk.\n\nStep 2 : bobruisk

Example 2:
Table related to Made the cut in context of 2009 U.S. Open (golf).\n\nQuestion:  what is the to par for retief goosen?\nStep 1 : The question asks for the "to par" for Retief Goosen. Looking at the table, we can see that Retief Goosen is listed on row 1, and the "To par" column for row 1 is +3. Therefore, the answer is +3.\n\nStep 2 : +3

Example 3:
Table related to Round 16 in context of 1969 VFL season.\n\nQuestion:  If the Venue was kardinia park what was the highest Crowd attended?\nStep 1 : The question asks for the highest crowd that attended a game at Kardinia Park. Looking at the table, we see that Kardinia Park is listed as the venue for round 3.  The crowd for that game was 16,211. The question asked for the highest crowd.  Looking at the rest of the table, the only higher crowd is 21,025.  Therefore, 21,025 is the answer.\n\nStep 2 : 21,025

Example 4:
Table related to Prime ministers in context of Interwar unemployment and poverty in the United Kingdom.\n\nQuestion:  What is the birth place of the prime minister who served George V and entered office on 23 October 1922?\nStep 1 : The answer can be arrived by locating the row where the prime minister entered office on 23 October 1922. This is the row corresponding to Andrew Bonar Law. Looking at the 'Birth Place' column for this row, the answer is given as "Rexton, Kent County, New Brunswick, Canada".\n\nStep 2 : rexton, kent county, new brunswick, canada

Example 5:
Table related to Round 12 in context of 1976 VFL season.\n\nQuestion:  What date did North Melbourne score 22.14 (146) as the home team?\nStep 1 : The question asks what date North Melbourne scored 22.14 (146) as the home team. We can find the row that corresponds to North Melbourne by finding the row with their logo. Row 5 shows their logo, which shows a Blue-coloured kangaroo with "North Melbourne" written on it, and their home score, which is 22.14 (146). In the last column, the date is 19 June 1976. Therefore, the answer to the question is 19 June 1976.\n\nStep 2 : 19 june 1976

Example 6:
Table related to Game Log in context of 1982 Atlanta Braves season.\n\nQuestion:  Where was the game on Thursday, April 29, and they played the Chicago Cubs?\nStep 1 : The table shows the 1982 Atlanta Braves game log. We need to find the game on Thursday, April 29. We see this date on row 19. Looking at the "Opponent" column, we see the Chicago Cubs. It is represented by a Purple background logo with a Cwritten in Red colour. Therefore, the game was played at the location listed in the "Site" column for row 19: Atlanta-Fulton County Stadium.\n\nStep 2 : atlanta-fulton county stadium

Example 7:
Table related to Medal table in context of 2007 Military World Games.\n\nQuestion:  Which nation has a Silver of 1, a Gold of 0, and a Total of 1?\nStep 1 : The question asks for the nation with 1 silver medal, 0 gold medals, and a total of 1 medal.Looking at the table, we can see that Bulgaria, Cameroon and Hungary, which are represented by a White-Green_Red striped flag, Green-Red-Tellow flag with a star, and a Red-White Green striped flag respectively, have 1 silver medal, 0 gold medals and a total of 1 medal.\n\nStep 2 : bulgaria, cameroon, hungary

Example 8:
Table related to Round 4 in context of 1956 VFL season.\n\nQuestion:  Which home team scored 12.15 (87)?\nStep 1 : The table shows the scores of different teams in round 4 of the 1956 VFL season. The row with the home team score of 12.15 (87) is the second row which corresponds to Collingwood. Therefore, the home team that scored 12.15 (87) is Collingwood.\n\nStep 2 : collingwood

Now, based upon the examples given above, you must understand the text and images given in the table and follow the steps 1-2 to answer the question corresponding to the table represented bt the data. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.

"""
    TEXT_PROMPT_HYBRID=""""""
    TEXT_PROMPT_0SHOT = """You are given a table in the form of an image. In the table, some entities (mentioned in text form originally) have been replaced by images that represent them. Based upon the context of the table while using real-world knowledge, your task is to identify the entities corresponding to the images in the table and answer the question. You must perform this task in the following steps:

You are also provided with some question-answer examples for better understanding the format of providing the answer:

Step 1: Reason about what should be the answer to the question by identifying the relevant entities represented by images using the context of the table and the question. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for disambiguating the entities and answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

Example 1:
Table context: Table related to Fifth round proper in context of 1975–76 FA Cup.\n\nQuestion: how many games played by sunderland are listed here?\n\nStep 2:\n2

Example 2:
Table context: Table related to Complete Formula One World Championship results in context of Playlife.\n\nQuestion: when was the benetton b198 chassis used?\n\nStep 2: 1998.  

Example 3:
Table context: Table related to Defunct railroads in context of List of Washington, D.C., railroads.\n\nQuestion: was the pennsylvania railroad under the prr or the rf&p?\n\nStep 2:PRR

Example 4:
Table context: Table related to Schedule and results in context of 2013–14 Chicago State Cougars women's basketball team.\n\nQuestion: how many games were played against grand canyon?\n\nStep 2:\n2

Example 5:
Table context: Table related to Roster|Letter winners in context of 1915 Michigan Wolverines football team.\n\nQuestion: how many players were taller and weighed more than frank millard?\nin the 2nd row and other in the 8th row.\n\nStep 2:\n2

Example 6:
Table context: Table related to Racing record|Career summary in context of Conor Daly.\n\nQuestion: the two teams who raced in 2011 are carlin motorsport and what other team?\n\nStep 2:\nSchmidt Motorsports

Example 7:
Table context: Table related to Regular season|Schedule in context of 1995 New York Jets season.\n\nQuestion: team that scored more than 40 points against the jets that is not the miami dolphins.\n\nStep 2:\nOakland Raiders

Example 8:
Table context: Table related to Winners|By Country in context of EHF Cup Winners' Cup.\n\nQuestion: did france or croatia have a larger finals total?\n\nStep 2:\nFrance

Now, based upon the examples given above, you must understand the text and images given in the table and follow the steps 1-2 to answer the question corresponding to the table represented bt the data. Try to keep the answer in active voice. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.

You must understand the image given and follow the steps 1-2 to answer the question corresponding to the table represented in the image. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.
"""

def get_prompt_alt(question_id, table, meta, query, image_id_to_image_path, base_prompt, answer):  #question,base_url_image,image_id_to_image_path
    
    content=[]
    tmp={}
    entity_to_image_map={}
    images = []
    img_cnt=0
    tmp={"type":"text", "text":"["}
    
    for i in table:

        for j in range(len(i)):
            #print(i[j])
            if("IMG" in i[j]):#i[j] in image_id_to_image_path
                img_tags = get_all_image_tags(i[j])
                
                for k in img_tags:
                    image_name = image_id_to_image_path[k]
                    if(image_name in entity_to_image_map):
                        # print("hii")
                        tag = entity_to_image_map[image_name]
                    else:
                        tag = f"{{ENTITY-{img_cnt}}}"
                        entity_to_image_map[image_name] = tag
                        img_cnt+=1
                        
                        images.append({"type":"text", "text": tag + " : " })
                        images.append({ "type": "image_url", "image_url": {"url": str(base_url_image+image_name), "detail": "low"}}) #data:image/jpeg;base64,{img}

                        # img = get_base_64_img(base_url_image+image_id_to_image_path[k])
                        # images.append({ "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "low"}})
                        
                    # print(img_cnt,image_id_to_image_path[k])
                    tmp['text']+=" "+ tag
                    #print(base_url_image+image_id_to_image_path[k])

                tmp['text']+=";"
                    
            else:
                tmp['text']+=" " + i[j] +";"

        tmp['text']+="] \n["
    tmp['text']+="] \n The mappings of entity to image are as follows:\n"
    content.append(tmp)    
    content.extend(images)
    # print(entity_to_image_map, len(entity_to_image_map))
    # print("-----------------------------------------------------")
    # print(set(entity_to_image_map.keys()), len(set(entity_to_image_map.keys())))
    print(len(entity_to_image_map))

    
    #base_prompt = "Answer the following question briefly using the table knowledge given below. The table has text and entity tags as data, The entity tags are place holders for the real images given at the end of this text. \n"
    content[0]["text"]= base_prompt+" \n"+meta+"\nQuestion: \n"+query+"\nTable: \n"+content[0]["text"]
#    print(content[0])
    return {"question_id":question_id, "prompt":content, "gold_truth":answer}
    

###loop###

base_prompt = TEXT_PROMPT_WIKISQL
keys = list(test_questions.keys())
for i in range(len(keys)):#len(keys)
    print(i, keys[i])
    x = keys[i]
    table = question_tables[test_questions[keys[i]]["table_context"]]
    answer = test_questions[keys[i]]["answer"]
    #print(table)
    table_str = str(table['table_array'])
    meta = get_metadata(table["table_id"])
    query = test_questions[keys[i]]["question"]
    # print(query)
    # print(x)
    # print(table["table_id"])
    # print(table)
    # final_prompt = get_prompt(question_tables[test_questions[x]['table_context']]['table_array'],test_questions[x]["question"],image_id_to_image_path,test_questions[x]['table_context'])

    prompts = get_prompt_alt(keys[i],table['table_array'], meta ,query, image_id_to_image_path, base_prompt, answer)
    # print(prompts)

    with open(os.path.join(f"/home/vivek/harshita/modelling/prompts_gpt4_remaining/{dataset}",question_type), "a") as f: #open(os.path.join(f"/home/vivek/harshita/train_test_split/{dataset}/", question_type))
        f.write(json.dumps(prompts) + "\n")



# final_requests = []
# qid_to_gold = {}

# for qid, prompt in prompts.items():
#     # qid_to_gold[qid] = prompt['gold_ans']
#     # image_encoding = encode_image(prompt['image'])
#     messages = [
#         {
#             "role": "user",
#             "content": prompt
#         }
#     ]
#     body = {
#         "model": "gpt-4o",
#         "messages": messages,
#         "max_tokens": 4096,
#         "temperature": 0.0,
#     }
    
#     final_request = {
#         "custom_id": qid,
#         "method": "POST",
#         "url": "/v1/chat/completions",
#         "body": body
#     }
#     final_requests.append(final_request)

# # %%
# with open("test_file.jsonl", "w") as f:
#     for request in final_requests:
#         f.write(json.dumps(request) + "\n")

# # %%
# len(final_requests)

# # %%
# from openai import OpenAI
# client = OpenAI(api_key="")

# batch_input_file = client.files.create(
#   file=open("test_file.jsonl", "rb"),
#   purpose="batch"
# )

# # %%
# batch_input_file_id = batch_input_file.id

# response = client.batches.create(
#     input_file_id=batch_input_file_id,
#     endpoint="/v1/chat/completions",
#     completion_window="24h",
#     metadata={
#       "description": "nightly eval job"
#     }
# )

# #print(response)

# batch_id = "batch_6KTeuSIC12IhvSmgYKiEgL4k"

# from IPython.display import clear_output
# from time import sleep
# while True:
#     status_response = client.batches.retrieve(batch_id=batch_id)
#     print(status_response.status)
#     content = client.files.content(file_id=status_response.output_file_id)
#     print(content.content)
#     #cost = content.content['response']['body']['usage']['prompt_tokens'] + 3*content.content['response']['body']['usage']['completion_tokens']
#     sleep(2)
#     clear_output(wait=True)
#     break
    
    
    

























# args = {
#     "messages": [{"role": "user", "content": prompt}],#"Generate a 3 sentence story about friendship" messages[0]
#     "temperature": temperature,
#     "model": model,

# }


# response = openai.ChatCompletion.create(**args)
# #client.chat.completion.create( model="gpt-4", messages=[{"role": "user", "content": "Generate a 3 sentence story about friendship"}] ) 
# cost = response['usage']['prompt_tokens'] + 3*response['usage']['completion_tokens']
# print(response)
# print("Cost:", cost)
# print(response.choices[0].message.content)




"""
Result
{
  "id": "chatcmpl-9VjpyyNXFaMo6KrTswsUgbyP1UTMa",
  "object": "chat.completion",
  "created": 1717351394,
  "model": "gpt-4o-2024-05-13",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Step 1: The question asks for the team with a 36-29 record. Looking at the table, we see that the record 36-29 is listed in row 9. The team in this row is represented by {ENTITY-21}. The image for {ENTITY-21} is the logo of the Dallas Mavericks.\n\nStep 2: dallas mavericks"
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 17137,
    "completion_tokens": 77,
    "total_tokens": 17214
  },
  "system_fingerprint": "fp_aa87380ac5"
}
"""