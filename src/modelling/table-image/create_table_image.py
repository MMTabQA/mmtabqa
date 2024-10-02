import pandas as pd
import json
import re
from IPython.core.display import HTML
import os
import imgkit
from tqdm import tqdm
from multiprocessing import Pool
import argparse


parser = argparse.ArgumentParser(description="Run your Python script with arguments")

parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (e.g., WikiTableQuestions)")

args = parser.parse_args()

path_arr = ['/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-1635.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-20961.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-9841.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-18269.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-13896.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-21287.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-17634.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-1330.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-14689.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-11231.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-2028.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-8095.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-18624.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-13164.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-7719.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-18095.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-12792.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-18063.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-9894.jpg', '/home/suyash/final_repo/temp_test_table_images_new/fetaqa_MM_cleaned/FTQ-18118.jpg']
DATASET = args.dataset
IMAGE_FILES_PATH = "/home/suyash/final_repo/temp_train_table_images/"
jsonl_file_path = f'/home/suyash/final_repo/HybridQA_Tab_MM/{DATASET}/experiment_ready_dataset/tables.jsonl'
image_id_to_path = json.load(open(f"/home/suyash/final_repo/Refactor/outputs/image_id_to_refined_path.json"))

import pandas as pd

import re
import os

def path_to_image_html(cell_value):
    if isinstance(cell_value, str):
        image_tags = get_all_image_tags(cell_value)
        for image_tag in image_tags:
            image_path = image_id_to_path.get(image_tag, "")
            image_path = "/home/suyash/final_repo/refactored_images/" + image_path
            if image_path:
                replaced_str = f'<img src="{image_path}" width="90" >'
                cell_value = cell_value.replace(image_tag, replaced_str)
    return cell_value

# Function to get all image tags
def get_all_image_tags(text):
    pattern = r"\{IMG-\{.*?\}\}"
    matches = re.findall(pattern, text)
    return matches

# Function to modify table style
def modify_table_style(html_str, border_width=10, border_color="black"):
    style = f'border-collapse: collapse; border: {border_width}px solid {border_color};'
    return html_str.replace("<table>", f'<table style="{style}">')

# Function to process each line in JSONL file
def process_line(line):
    json_obj = json.loads(line)
    table_id = json_obj.get('table_id', None)
    if table_id and table_id in relevant_table_ids:
        table_data = json_obj.get('table_array', [])
        table = pd.DataFrame(table_data[1:], columns=table_data[0])
        table = table.fillna("")
        table_df_html = table.applymap(path_to_image_html)
        html_df = HTML((table_df_html.to_html(escape=False)))
        table_id = table_id.replace("/", "_")
        path = os.path.join(BASE_DIR, DATASET, f"{table_id}.jpg")
        try:
            modified_html = modify_table_style(html_df.data)  # Increase border width
            imgkit.from_string(modified_html, path, options=options)
        except:
            pass

# Main function
if __name__ == "__main__":
    relevant_table_ids = set()
    for QUEST_TYPE in ["explicit", "answer", "implicit","visual"]:
        test_file_path = f"/home/suyash/final_repo/[temp]train_test_questions/{DATASET}/test_{QUEST_TYPE}_questions.jsonl"
        train_file_path = f"/home/suyash/final_repo/[temp]train_test_questions/{DATASET}/train_{QUEST_TYPE}_questions.jsonl"

        with open(test_file_path, 'r') as f:
            for line in f:
                json_obj = json.loads(line)
                relevant_table_ids.add(json_obj['table_context'])

    # print(len(relevant_table_ids))
    # relevant_table_ids=['Amanda/Bynes/1', 'SC/Toronto/0', 'Pittsburgh/Penguins/Radio/Network/0', 'List/of/television/stations/in/Canada/0', 'List/of/current/and/former/Super/League/venues/1', '1995/Skate/Canada/International/1', 'Triple/J/Hottest/100,/2000/0', '2010/Crystal/Skate/of/Romania/0', 'Enschede/Marathon/0', 'Air/Austral/0', 'Indonesia/at/the/Asian/Games/40', 'UEFA/Cup/and/Europa/League/records/and/statistics/0', 'Sports/in/Maryland/9', 'Star/(football/badge)/7', 'Edge/(magazine)/0', 'Triple/J/Hottest/100,/2010/1', 'FITA/Archery/World/Cup/3', '1995/Skate/Canada/International/1', 'Commuter/rail/in/North/America/0', 'List/of/football/clubs/in/Italy/3', '2012/Summer/Olympics/and/Paralympics/gold/post/boxes/1', 'List/of/improvisational/theatre/companies/3', 'List/of/Medal/of/Honor/recipients/for/World/War/II/3', 'WLIR/0', 'Azerbaijan/Premier/League/1', 'ADX/Florence/3', 'Greatest/Britons/spin-offs/0', '2013/Charlotte/Eagles/season/0', 'List/of/largest/buildings/in/the/world/1', 'List/of/former/national/capitals/13', 'List/of/districts/of/Seoul/1', 'Statler/Hotels/0', 'List/of/participants/of/the/Gaza/flotilla/0', 'List/of/European/Athletics/Championships/records/2', 'Maryland/Public/Service/Commission/0', 'List/of/the/oldest/buildings/in/the/world/4', 'International/Democrat/Union/0', 'List/of/airports/in/New/South/Wales/1', "2013/in/men's/road/cycling/1", 'List/of/NCAA/football/programs/at/Catholic/colleges/3', 'List/of/football/clubs/in/Italy/7', 'List/of/doping/cases/in/athletics/5', 'List/of/largest/monoliths/in/the/world/1', 'Azerbaijan/Premier/League/1', 'List/of/cricket/grounds/in/India/32', 'List/of/Canadian/submissions/for/the/Academy/Award/for/Best/Foreign/Language/Film/0', '2012/Thai/Division/2/League/Bangkok/&/field/Region/0', 'Administrative/divisions/of/Uttar/Pradesh/0', 'Schenker/League/0', 'List/of/ancient/watermills/4', 'List/of/winners/of/the/London/Marathon/2', 'Minnesota/United/FC/0', 'List/of/Ultras/of/the/Eastern/Himalayas/1', 'Liberal/Party/of/Canada/candidates,/2011/Canadian/federal/election/8', 'List/of/Malaysian/football/transfers/2014/20', 'List/of/maritime/boundary/treaties/4', "List/of/Australian/Open/women's/singles/champions/2", 'Canal/Hotel/bombing/0', 'World/record/progression/100/metres/backstroke/0', 'Soccer-specific/stadium/6', '3000/metres/steeplechase/3', '2012/IAAF/Diamond/League/0', 'List/of/Phi/Kappa/Psi/brothers/20', '2012/Veikkausliiga/0', 'Indraneil/Sengupta/0', 'List/of/diplomatic/missions/in/Vietnam/0', 'List/of/museums/in/Metro/Manila/0', 'List/of/museums/in/the/Grisons/0', 'List/of/National/Treasures/of/Japan/(writings:/Japanese/books)/0', 'List/of/ancient/watermills/4', 'List/of/medical/schools/in/Malaysia/0', 'Au/Kin-Yee/0', 'List/of/Michelin/starred/restaurants/18', 'Television/in/South/Africa/0', '1999/World/Rhythmic/Gymnastics/Championships/5', '2013/FKF/Division/One/1', '2010/Torneo/Descentralizado/0', 'List/of/space/telescopes/0', '1929/International/Cross/Country/Championships/1', 'Canal/Hotel/bombing/0', '5000/metres/0', 'List/of/territorial/disputes/13', 'List/of/alumni/and/faculty/of/the/Royal/Melbourne/Institute/of/Technology/8', '2004/Campeonato/Paulista/0', 'Gary/Bartz/0', '2011/European/Team/Championships/Super/League/4', '2010/Nationwide/Tour/0', 'List/of/Iranian/submissions/for/the/Academy/Award/for/Best/Foreign/Language/Film/0', 'Gymnastics/at/the/2006/Commonwealth/Games/2', '1996/World/Junior/Ice/Hockey/Championships/0', 'SAS/Institute/0', 'Economy/of/Germany/1', 'Administrative/divisions/of/Uttar/Pradesh/0', '2011/European/Team/Championships/Super/League/26', 'FAM/Football/Awards/8', 'Charlie/Vox/0', 'List/of/NBC/television/affiliates/(table)/1', '2011/MLS/SuperDraft/1', 'Cardiff/City/F.C./Player/of/the/Year/0', 'Liga/ASOBAL/0', 'List/of/Roman/victory/columns/0', 'List/of/Virtual/Console/games/for/Nintendo/3DS/(North/America)/2', 'Asian/Club/Championship/and/AFC/Champions/League/records/and/statistics/0', 'List/of/Sinfonians/21', '2010/Major/League/Soccer/season/0', 'List/of/video/games/published/by/Aksys/Games/9', 'List/of/sports/teams/in/Florida/2', 'List/of/New/York/State/Historic/Markers/in/Bronx/County,/New/York/0', 'List/of/National/Historic/Sites/of/Canada/in/Prince/Edward/Island/0', '2004/AFL/Draft/1', 'Stephen/Hendry/1', '2009/Aerobic/Gymnastics/European/Championships/0', 'List/of/wealthiest/non-inflated/historical/figures/3', 'List/of/southpaw/stance/boxers/0', 'Swedish/Football/Division/1/1', 'List/of/mosques/in/the/United/Kingdom/0', '2011/in/the/Philippines/2', 'Los/Angeles/Galaxy/1', 'List/of/hoards/in/Britain/4', 'Sports/in/Chicago/1', "FIRS/Senior/Men's/Inline/Hockey/World/Championships/0", 'List/of/shopping/malls/in/the/Philippines/18', 'Canon/law/(Catholic/Church)/0', 'List/of/Florida/Agricultural/and/Mechanical/University/alumni/2', 'Major/crimes/in/the/United/Kingdom/11', '2011/NRL/season/0', 'List/of/largest/buildings/in/the/world/3', 'List/of/submissions/to/the/40th/Academy/Awards/for/Best/Foreign/Language/Film/0', 'List/of/maritime/boundary/treaties/4', 'NCAA/Division/I/conference/realignment/7', 'Anthony/M./Jones/0', 'MegaStructures/0', '2001/San/Marino/Grand/Prix/0', '1995/Skate/Canada/International/1', '1996/Skate/Canada/International/1', '2011/in/spaceflight/3', 'List/of/songs/in/The/Beatles:/Rock/Band/0', 'List/of/census/divisions/of/Alberta/0', 'List/of/ancient/watermills/4', '2011/Rugby/League/Four/Nations/1', 'List/of/the/busiest/airports/in/Italy/1', 'David/Morrissey/0', 'MegaStructures/0', 'List/of/institutions/of/higher/education/in/Uttar/Pradesh/1', 'Ranked/lists/of/Chilean/regions/7', 'List/of/museums/in/Metro/Manila/0', '2013/Charlotte/Eagles/season/0', '2008/Geylang/United/FC/season/0', 'River/island/1', 'Administrative/divisions/of/Uttar/Pradesh/0', "List/of/teams/and/cyclists/in/the/2010/Giro/d'Italia/1", 'List/of/submissions/to/the/60th/Academy/Awards/for/Best/Foreign/Language/Film/0', 'Eurogroup/0', 'List/of/programming/changes/on/Australian/television/in/2008/4', 'List/of/NCAA/conferences/3', "List/of/teams/and/cyclists/in/the/2011/Giro/d'Italia/1", 'List/of/Trabzonspor/players/0', 'List/of/cities/and/towns/in/India/0', 'European/Challenge/Cup/1', 'UEFA/Cup/and/Europa/League/records/and/statistics/0', 'List/of/the/busiest/airports/in/Asia/5', 'List/of/European/financial/services/companies/by/revenue/0', '1998/Luxembourg/Grand/Prix/0', '2005/IAAF/World/Half/Marathon/Championships/1', '1950/International/Cross/Country/Championships/1', 'List/of/former/national/capitals/13', 'List/of/airports/in/Zimbabwe/0', 'List/of/Christmas/television/specials/30', 'List/of/Malaysian/football/transfers/2014/1', 'List/of/Maccabiah/records/in/swimming/1', 'Major/party/0', 'Comparison/of/Asian/national/space/programs/5', '3000/metres/steeplechase/3', 'List/of/ship/launches/in/1944/1', 'List/of/honorary/British/knights/and/dames/7', 'Deloitte/Football/Money/League/8', 'Outline/of/Austria/5', 'List/of/ship/launches/in/1944/5', '2000/World/Junior/Figure/Skating/Championships/2', 'List/of/multiple/barrel/firearms/0', 'HK/Lida/0', 'Adrian/Smith/(architect)/0', 'Anti-Bullfighting/City/0', 'List/of/southpaw/stance/boxers/0', 'List/of/European/ultra/prominent/peaks/1', 'List/of/maritime/boundary/treaties/4', 'List/of/Trabzonspor/players/0', '2013/European/Team/Championships/Super/League/3', '1972/International/Cross/Country/Championships/3', '1999/World/Rhythmic/Gymnastics/Championships/5', 'AFL/Ontario/0', 'David/Morrissey/0', 'Jonathan/Taylor/Thomas/1', 'List/of/tallest/structures/in/Africa/0', 'David/Morrissey/0', 'Comparison/of/OLAP/Servers/0', '1500/metres/0', 'Algerian/Ligue/Professionnelle/1/4', 'List/of/European/Athletics/Championships/records/2', 'List/of/sterile/insect/technique/trials/0', 'List/of/programming/changes/on/Australian/television/in/2008/4', '2011/Asian/Rhythmic/Gymnastics/Championships/6', 'List/of/doping/cases/in/athletics/22', 'Edge/(magazine)/0', '1950/International/Cross/Country/Championships/1', '1996/Skate/Canada/International/1', 'FITA/Archery/World/Cup/4']



    # relevant_table_ids=set(relevant_table_ids)
    #relevant_table_ids = [(x.split("/")[-1]).split(".jpg")[0] for x in path_arr]
    #print(relevant_table_ids)
    # return
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.expand_frame_repr', False)

    # Path to save images
    BASE_DIR = "/home/suyash/final_repo/temp_test_table_images_new/"
    try:
        os.makedirs(os.path.join(BASE_DIR, DATASET), exist_ok=True)
    except:
        pass

    options = {
        "enable-local-file-access": None,
        "minimum-font-size": 24
    }

    # Read and process each line in JSONL file using multiprocessing
    with open(jsonl_file_path, 'r') as f:
        lines = f.readlines()

    with Pool() as pool:
        list(tqdm(pool.imap(process_line, lines), total=len(lines)))
