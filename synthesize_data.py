import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import random
import csv
from text_generation import Client
from io import StringIO

GREEN = '\033[38;5;35m'
RED = '\033[38;5;196m'
ORANGE = '\033[38;5;215m'
YELLOW = '\033[38;5;229m'
BLUE = '\033[0;34m'
PURPLE = '\033[38;5;90m'
RESET = '\033[0m'
COLOR = YELLOW

endpoint_url = "http://ice193:6300"
client = Client(endpoint_url, timeout = 60)

data_input = ""
data_input_list = []
df = pd.read_csv('use_cases_subset.csv')
for index, row in df.iterrows():
    if (pd.isna(row.use_case) == False): 
        data_input += "\n " + row.use_case
        data_input_list += [row.use_case]
    data_input += "\n"

labels = []
with open('labels.txt', 'r') as file:
    lines = file.readlines()
for line in lines:
    string = line.strip()
    labels.append(string)

# create synthetic data
print("\n\n------------------------------------------------------------")
print("\t::Beginning synthetic data creation::")
print("------------------------------------------------------------")

for i in range(1000):
    first = True
    if i != 0: first = False

    # with open("fake_data.txt", "r") as g:
    #     previous_entries = g.read()
    
    # randomly choose labels for LLM to base synthetic entry on
    num_labels = random.randint(3, 8)
    chosen_labels = set()
    for i in range(num_labels):
        chosen_string = random.choice([s for s in labels if s not in chosen_labels])
        chosen_labels.add(chosen_string)
    csv_labels = ", ".join(chosen_labels)
    chosen_labels = "\n".join(chosen_labels)

    # randomly choose entries from dataset for LLM to mimic
    # giving LLM entire dataset may flood it, and choosing only a subset may help with variety in synthetic entries
    chosen_entries = set()
    for i in range(40):
        chosen_string = random.choice([s for s in data_input_list if s not in chosen_entries])
        chosen_entries.add(chosen_string)
    chosen_entries = "\n".join(chosen_entries)

     #+ "You must also MAKE SURE you DO NOT REPEAT any previous synthetic data entry made in the past, which you can see here:\n" + previous_entries'''
    prompt = "<s> [INST] Your task is to create synthetic data based on a provided sample from a dataset and a list of labels. The data you create will be an entry that mimics on of the entries from the provided data. You will also be given a small list of labels, and your entry must be related to each of the labels provided. Respond in this format: \n\nEntry:\n(your fabricated entry here)@@\n\nBe sure to include the '@@' at the end of your entry, because it is important for formating. Here is the sample data; replicate the tone and writing style of these entries as closely as possible:\n"+ chosen_entries + "\n Now I will give you the labels that your entry must relate to. Here it is:\n" + chosen_labels + "\nRemeber, DO NOT specify when a label is mentioned in the entry. DO NOT INCLUDE (@label) OR ('label') OR ANY OTHER SPECIFIER WITHIN THE ENTRY\n\n[\INST] Okay, I will now create a fabricated data entry, consisting of no more than two paragraphs. I will relate it to EACH OF THE PROVIDED LABELS, I will place a '@@' immediately after my entry, I WILL NOT MENTION THE LABELS within the entry, and I will follow the specified format. Here it is:\n\nEntry:\n"
    # sythesized data entry
    fake_data = ""
    print("Number of labels: " + RED + str(num_labels) + RESET + "\n")
    print("Given labels:\n" + GREEN + chosen_labels + RESET + "\n")
    print("Synthetic entry:" + YELLOW)
    for response in client.generate_stream(prompt, max_new_tokens=500, repetition_penalty=1.4, stop_sequences=["<\s>", "@@", "\""]):
        if not response.token.special:
            fake_data += response.token.text
            print(response.token.text, end="")
    print(RESET + "\n------------------------------------------------------------")

    # write to csv
    fake_data = fake_data.rstrip("@@")
    #fake_data = fake_data.rstrip("\"@@")
    fake_data = fake_data.rstrip("(@@")
    with open('synth_data.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if first: 
            writer.writerow(['entry', 'labels'])
            first = False
        writer.writerow([fake_data, csv_labels])
    f.close()
    #f = open('fake_data.txt', 'a')
    #f.write("entry, labels")
    #f.write("\n" + fake_data + "; " + chosen_labels)
    #f.close()
