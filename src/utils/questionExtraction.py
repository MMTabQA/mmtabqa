import numpy as np
import pandas as pd
# from tqdm import tqdm
import os

# def explicit_for_all_rows(tsv_path):
#     # Load the TSV file into a DataFrame
#     try:
#         df = pd.read_csv(tsv_path, sep='\t')
#     except FileNotFoundError:
#         return "TSV file not found"

#     results = []
#     err_arr=[]
#     df.head()
#     # Iterate through each row in the DataFrame
#     for index, row in tqdm(df.iterrows()):
#         # print(index)
#         question_id = row['id']
#         question_lower = row['utterance'].lower()
        
#         context_path =   row['context'] #Double backslash is used to escape the single backslash

#         # Construct the full path to the CSV file
#         table_path = os.path.join("../WikiTableQuestions", context_path)
#         try:
#             # Check if the CSV file exists
#             if os.path.exists(table_path):
#                 # Load the CSV file into a DataFrame
#                 table = pd.read_csv(table_path)

#                 # Convert the table values to lowercase for case-insensitive comparison
#                 table_lower = table.apply(lambda x: x.lower() if type(x) == str else x)

#                 found_in_table = False
#                 found_word = ""
#                 cell_found = ""
#                 # Iterate through each word in the question
#                 for word in question_lower.split():
#                     # Iterate through each row and column in the table
#                     for _, table_row in table_lower.iterrows():
#                         for cell in table_row:
#                             if word == str(cell).lower():
#                                 found_in_table = True
#                                 found_word = word
#                                 cell_found=cell
#                                 results.append((question_id,found_word,question_lower,context_path))
#                                 break
#                         if found_in_table:
#                             break
#                     if found_in_table:
#                         break

#                 # Append the question ID to the results list
               
#             else:
#                 # If the CSV file does not exist, append None for the question ID
#                 # results.append(None)
#                 continue
#         except Exception as e:
#             print(question_id,e)
#             err_arr.append(question_id)
#     return results,err_arr

# # Example usage:
# tsv_path = "../WikiTableQuestions/data/training.tsv"
# results,err_arr = explicit_for_all_rows(tsv_path)


# # Assuming 'results' is a list of question IDs
# results_array = np.array(results)
# error_array = np.array(err_arr)
import pandas as pd
import os
from tqdm import tqdm

import pandas as pd
import os
from tqdm import tqdm

def explicit_for_all_rows(tsv_path, result_tsv_path):
    # Load the TSV file into a DataFrame
    try:
        df = pd.read_csv(tsv_path, sep='\t')
    except FileNotFoundError:
        return "TSV file not found"

    results = []
    err_arr = []

    # Iterate through each row in the DataFrame
    for index, row in tqdm(df.iterrows()):
        question_id = row['id']
        question_lower = row['utterance'].lower()
        context_path = row['context']  # Double backslash is used to escape the single backslash
        targetVal = row["targetValue"]

        # Construct the full path to the original and new CSV file
        original_table_path = os.path.join("../WikiTableQuestions", context_path)
        new_table_path = os.path.join("../WikiTableQuestions", context_path.replace('.csv', '-new.csv'))

        try:
            # Check if the original CSV file exists
            if os.path.exists(original_table_path):
                # Load the original CSV file into a DataFrame
                original_table = pd.read_csv(original_table_path)

                # Convert the table values to lowercase for case-insensitive comparison
                original_table_lower = original_table.apply(lambda x: x.astype(str).str.lower())

                print(f"Processing Question ID: {question_id}")

                # Check if the new CSV file exists
                if os.path.exists(new_table_path):
                    # Load the new CSV file into a DataFrame
                    new_table = pd.read_csv(new_table_path, sep='\t')  # Update here for tab-separated

                    # Convert the table values to lowercase for case-insensitive comparison
                    new_table_lower = new_table.apply(lambda x: x.astype(str).str.lower())

                    found_words = []
                    cells_found = []
                    original_rows_found = []

                    # Iterate through each row and column in the original table
                    for original_row_index, original_row in original_table_lower.iterrows():
                        for original_cell in original_row:
                            # Check if the original cell value appears in the question
                            if (str(original_cell).lower() in question_lower) or (str(original_cell).lower() in targetVal.lower()):
                                found_words.append(str(original_cell).lower())
                                cells_found.append(original_cell)
                                original_rows_found.append(original_row)

                    # Check if any of the found words is replaced by an image in the corresponding cell of the new table
                    for found_word, cell_found, original_row_found in zip(found_words, cells_found, original_rows_found):
                        # print(f"Found Word: {found_word}")
                        # print(f"Cell Found: {cell_found}")
                        # print(f"Original Row Found: {original_row_found}")
                        # print(new_table_lower.columns.get_loc(original_row_found[original_row_found == cell_found].index[0]))
                        corresponding_cell = new_table_lower.iloc[original_row_found.name, new_table_lower.columns.get_loc(original_row_found[original_row_found == cell_found].index[0])]
                        if "img-" in corresponding_cell.lower():
                            img_id = corresponding_cell  # Store the img_id
                            results.append((question_id, found_word, question_lower, context_path, img_id))
                            break

                            # print(f"Found Word: {found_word}")
                            # print(f"Cell Found: {cell_found}")
                            # print(f"Corresponding Cell: {corresponding_cell}")
                            

                else:
                    print(f"New CSV file not found for Question ID: {question_id}")
                    # If the new CSV file does not exist, append None for the question ID
                    # results.append(None)
                    continue
            else:
                print(f"Original CSV file not found for Question ID: {question_id}")
                # If the original CSV file does not exist, append None for the question ID
                # results.append(None)
                continue
        except Exception as e:
            print(f"Error processing question ID {question_id}: {str(e)}")
            err_arr.append(question_id)

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to a new TSV file
    results_df.to_csv(result_tsv_path, sep='\t', index=False)

    return f"Results saved to {result_tsv_path}"

# Example usage:
tsv_path = "../WikiTableQuestions/data/training.tsv"
result_tsv_path = "explicit_results.tsv"
# explicit_for_all_rows(tsv_path, result_tsv_path)

# # Save the NumPy array to a file
# np.save('results_array1.npy', results_array)
# np.save('error_array.npy', error_array)

# for i, question_id in enumerate(results, start=1):
#     print(f"Row {i}, Question ID {question_id}: Processing...")

# import pandas as pd

# def add_context_to_tsv(input_tsv_path, reference_tsv_path, output_tsv_path):
#     # Load the input TSV file into a DataFrame
#     try:
#         input_df = pd.read_csv(input_tsv_path, sep='\t')
#     except FileNotFoundError:
#         return "Input TSV file not found"

#     # Load the reference TSV file into a DataFrame
#     try:
#         reference_df = pd.read_csv(reference_tsv_path, sep='\t')
#     except FileNotFoundError:
#         return "Reference TSV file not found"

#     # Merge the input and reference DataFrames based on the 'id' column
#     merged_df = pd.merge(input_df, reference_df[['id']], how='inner', on='id')

#     # Save the merged DataFrame to a new TSV file
#     merged_df.to_csv(output_tsv_path, sep='\t', index=False)

#     # Return the number of unique contexts
#     num_unique_contexts = merged_df['context'].nunique()
#     return f"Context added and saved to {output_tsv_path}. Number of unique contexts: {num_unique_contexts}"

# # Example usage:
# input_tsv_path = "../WikiTableQuestions/data/training.tsv"
# reference_tsv_path = "../WikiTableQuestions/data/explicit_ques_ids.tsv"
# output_tsv_path = "../WikiTableQuestions/data/explicit_ques_ids_1.tsv"

# # result_message = add_context_to_tsv(input_tsv_path, reference_tsv_path, output_tsv_path)
# # print(result_message)

# import pandas as pd

# def get_rows_not_in_reference(input_tsv_path, reference_tsv_path, output_tsv_path):
#     # Load the input TSV file into a DataFrame
#     try:
#         input_df = pd.read_csv(input_tsv_path, sep='\t')
#     except FileNotFoundError:
#         return "Input TSV file not found"

#     # Load the reference TSV file into a DataFrame
#     try:
#         reference_df = pd.read_csv(reference_tsv_path, sep='\t')
#     except FileNotFoundError:
#         return "Reference TSV file not found"

#     # Identify rows in the input DataFrame that are not in the reference DataFrame
#     rows_not_in_reference_df = input_df[~input_df['id'].isin(reference_df['id'])]

#     # Replace '.csv' with '.table' in the 'context' column
#     rows_not_in_reference_df['context'] = rows_not_in_reference_df['context'].str.replace('.csv', '.table')

#     # Save the DataFrame to a new TSV file
#     rows_not_in_reference_df.to_csv(output_tsv_path, sep='\t', index=False)

#     return f"Rows not in reference saved to {output_tsv_path}"


# # output_tsv_path = "../WikiTableQuestions/data/implicit_ques_ids.tsv"

# # result_message = get_rows_not_in_reference(input_tsv_path, reference_tsv_path, output_tsv_path)
# # print(result_message)

input_tsv_path = "implicit_results.tsv"
output_tsv_path = "chatgptPrompt.tsv"
import pandas as pd
import os

def concatenate_question_and_table(input_tsv_path, output_tsv_path):
    # Load the input TSV file into a DataFrame
    try:
        input_df = pd.read_csv(input_tsv_path, sep='\t')
    except FileNotFoundError:
        return "Input TSV file not found"

    # Create a list to store the concatenated data
    concatenated_data = []
    print(input_df.columns)
    # Iterate through each row in the input DataFrame
    for index, row in input_df.iterrows():
        question_id = row['id']
        utterance = row['utterance']
        context_path = row['context']
        ans = row['targetValue']

        # Construct the full path to the CSV file
        table_path = os.path.join("../WikiTableQuestions", context_path)

        try:
            # Check if the CSV file exists
            if os.path.exists(table_path):
                # Read the CSV file with pipe as separator and handle Unnamed columns
                table = pd.read_csv(table_path, sep='|', engine='python')

                # Drop columns starting with 'Unnamed'
                table = table.loc[:, ~table.columns.str.startswith('Unnamed')]
                table.reset_index(inplace=True)
                table = table.transpose()
                table.reset_index(inplace=True)
                table = table.transpose()
                # Check if the evidence is present in the table
              
                evidence_text = "(Row: , Column: )"
                concatenated_text = f"""
   Given a table, a question and its answer, point out the evidence cells from the table where the answer is located, or the cells used to compute the answer. Check carefully all the cells. The evidence is most likely to be there. It can also involve calculation, like addition or difference or counting, and reasoning from multiple rows
If the evidence is not there, output No. Do not give wrong evidence. 
If evidence is there, only then output it in the form (Row number, and Column number). If multiple Rows or Columns are needed, output it.
The answer is {ans}
Here are a few examples:
Question - which team won previous to crettyard?
Answer - Wolfe Tones
Table -  0                   1         2      3           4
index  Team                County    Wins   Years won 
    0  Greystones          Wicklow       1        2011
    1  Ballymore Eustace   Kildare       1        2010
    2  Maynooth            Kildare       1        2009
    3  Ballyroan Abbey     Laois         1        2008
    4  Fingal Ravens       Dublin        1        2007
    5  Confey              Kildare       1        2006
    6  Crettyard           Laois         1        2005
    7  Wolfe Tones         Meath         1        2004
    8  Dundalk Gaels       Louth         1        2003
crettyard won in 2005, so the previous team would have won in 2004.
Evidence : (Row: 7, Column: 3)

Question - how many more passengers flew to los angeles than to saskatoon from manzanillo airport in 2013?
Answer - 12,467
Table - 
    0      1                            2            3         4                      5
index  Rank   City                         Passengers   Ranking   Airline              
    0      1  United States, Los Angeles   14,749                 Alaska Airlines      
    1      2  United States, Houston       5,465                  United Express       
    2      3  Canada, Calgary              3,761                  Air Transat, WestJet 
    3      4  Canada, Saskatoon            2,282        4                              
    4      5  Canada, Vancouver            2,103                  Air Transat          
    5      6  United States, Phoenix       1,829        1         US Airways           
    6      7  Canada, Toronto              1,202        1         Air Transat, CanJet  
    7      8  Canada, Edmonton             110                                         
    8      9  United States, Oakland       107                             
    Take the difference for Passengers columns in Saskatoon and Los Angeles
Evidence : (Row: 0,3, Column: 2,3)

Question - what was the last year where this team was a part of the usl a-league?
Answer : 2004
Table - Table - 
    0      1          2                     3                4                 5                 6                 7
index  Year   Division   League                Regular Season   Playoffs          Open Cup          Avg. Attendance 
    0   2001          2  USL A-League          4th, Western     Quarterfinals     Did not qualify   7,169           
    1   2002          2  USL A-League          2nd, Pacific     1st Round         Did not qualify   6,260           
    2   2003          2  USL A-League          3rd, Pacific     Did not qualify   Did not qualify   5,871           
    3   2004          2  USL A-League          1st, Western     Quarterfinals     4th Round         5,628           
    4   2005          2  USL First Division    5th              Quarterfinals     4th Round         6,028           
    5   2006          2  USL First Division    11th             Did not qualify   3rd Round         5,575           
    6   2007          2  USL First Division    2nd              Semifinals        2nd Round         6,851           
    7   2008          2  USL First Division    11th             Did not qualify   1st Round         8,567           
    8   2009          2  USL First Division    1st              Semifinals        3rd Round         9,734           
    9   2010          2  USSF D-2 Pro League   3rd, USL (3rd)   Quarterfinals     3rd Round         10,727          
Evidence: (Row 3, Column 1)

Question : how many people stayed at least 3 years in office?
Answer : 4
Table : Table - 
    0    1                    2                   3                4                       5                                                                   6
index       Name                 Took office         Left office      Party                   Notes/Events                                                      
    0   11  William McCreery     March 4, 1803       March 3, 1809    Democratic Republican                                                                     
    1   12  Alexander McKim      March 4, 1809       March 3, 1815    Democratic Republican                                                                     
    2   13  William Pinkney      March 4, 1815       April 18, 1816   Democratic Republican   Resigned to accept position as Minister Plenipotentiary to Russia 
    3   14  Peter Little         September 2, 1816   March 3, 1823    Democratic Republican                                                                     
    4   14  Peter Little         March 4, 1823       March 3, 1825    Jacksonian DR                                                                             
    5   14  Peter Little         March 4, 1825       March 3, 1829    Adams                                                                                     
    6   15  Benjamin C. Howard   March 4, 1829       March 3, 1833    Jacksonian   
Take the difference betweem Left Office and Took Office columns

Evidence : (Rows: 0,1,5,6, Columns: 2,3)
    """
                # Concatenate 'utterance', the extracted table, evidence, and answer dynamically
                concatenated_text += f"\nQuestion: {utterance}\nTable - \n{table.to_string(index=False)}\n Answer : {ans}\nEvidence: "

                # Append the concatenated data as a dictionary
                concatenated_data.append({'id': question_id, 'concatenated_text': concatenated_text})
            else:
                print(f"CSV file not found at {table_path} for question ID {question_id}")
        except Exception as e:
            print(f"Error processing question ID {question_id}: {str(e)}")

    # Convert the list of dictionaries to a DataFrame
    concatenated_df = pd.DataFrame(concatenated_data)

    # Save the concatenated data to a new TSV file
    concatenated_df.to_csv(output_tsv_path, sep='\t', index=False)

    return f"Concatenated data saved to {output_tsv_path}"


# result_message = concatenate_question_and_table(input_tsv_path, output_tsv_path)
# print(result_message)

# def extract_question_ids_with_empty_evidence(input_tsv_path, output_tsv_path):
#     # Load the input TSV file into a DataFrame
#     try:
#         input_df = pd.read_csv(input_tsv_path, sep='\t')
#     except FileNotFoundError:
#         return "Input TSV file not found"

#     # Create a DataFrame with only the 'id' column
#     question_ids_df = input_df[['id']]

#     # Add an empty 'evidence' column
#     question_ids_df['evidence'] = "(Row: , Column: )"

#     # Save the DataFrame to a new TSV file
#     question_ids_df.to_csv(output_tsv_path, sep='\t', index=False)

#     return f"Question IDs with empty evidence saved to {output_tsv_path}"



# # result_message = extract_question_ids_with_empty_evidence(input_tsv_path, output_tsv_path)
# # print(result_message)
# #
# import pandas as pd

# def change_csv_to_table(file_path):
#     # Load the DataFrame from the specified file
#     try:
#         df = pd.read_csv(file_path, sep='\t')
#     except FileNotFoundError:
#         return "File not found"

#     # Replace ".csv" with ".table" in the "context" column
#     df['context'] = df['context'].str.replace('.csv', '.table')

#     # Save the modified DataFrame back to the same file
#     df.to_csv(file_path, sep='\t', index=False)

#     return f"File updated: {file_path}"

# # Example usage:
# implicit_results_path = "implicit_results.tsv"
# # change_csv_to_table(implicit_results_path)

# import pandas as pd

# def rename_explicit_columns(explicit_results_path, renamed_explicit_path):
#     # Load explicit results into a DataFrame
#     try:
#         explicit_results_df = pd.read_csv(explicit_results_path, sep='\t')
#     except FileNotFoundError:
#         return "Explicit results file not found"

#     # Rename columns in explicit results
#     explicit_results_df.columns = ['id', 'matched_pharse', 'question', 'context','img-id']

#     # Save the renamed explicit results to a new TSV file
#     explicit_results_df.to_csv(renamed_explicit_path, sep='\t', index=False)

#     return f"Renamed explicit results saved to {renamed_explicit_path}"

# # Example usage:
# explicit_results_path = "explicit_results.tsv"
# renamed_explicit_path = "renamed_explicit_results.tsv"

# # rename_explicit_columns(explicit_results_path, renamed_explicit_path)

# import pandas as pd

# def get_implicit_question_rows(tsv_path, explicit_results_path, implicit_results_path):
#     # Load the TSV file into a DataFrame
#     try:
#         df = pd.read_csv(tsv_path, sep='\t')
#     except FileNotFoundError:
#         return "TSV file not found"

#     # Load explicit results into a DataFrame
#     try:
#         explicit_results_df = pd.read_csv(explicit_results_path, sep='\t')
#     except FileNotFoundError:
#         return "Explicit results file not found"

#     # Extract explicit question IDs
#     explicit_question_ids = set(explicit_results_df['id'])

#     # Filter implicit question rows
#     implicit_question_rows = df[~df['id'].isin(explicit_question_ids)]

#     # Save the implicit question rows to a new TSV file
#     implicit_question_rows.to_csv(implicit_results_path, sep='\t', index=False)

#     return f"Implicit results saved to {implicit_results_path}"

# # Example usage:
# tsv_path = "../WikiTableQuestions/data/training.tsv"
# explicit_results_path = "explicit_results.tsv"
# implicit_results_path = "implicit_results.tsv"

# get_implicit_question_rows(tsv_path, explicit_results_path, implicit_results_path)
