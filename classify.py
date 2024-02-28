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

# 1. make labels as needed for a given entry in the dataset
def generate_labels(case_notes):
    prompt = "<s> [INST] Your task is to look at an entry from a dataset and determine one or more labels that adequately describe the entry. Please respond in a format that can easily be converted to a python list by separating each label in your list with a ', '. Here is the entry: " + case_notes + "[\INST] Sure, here is a string with the labels separated only by a comma, and I WILL NOT CONTINUE SPEAKING after the list: \n\n\""
    generated_labels = ""
    for response in client.generate_stream(prompt, max_new_tokens=400, stop_sequences=["<\s>", "\""]):
        if not response.token.special:
            generated_labels += response.token.text
    print("Generated labels:\n" + generated_labels.rstrip("\"") + "\n")
    generated_labels = generated_labels.split(", ")
    # standardize labels
    for i in range(len(generated_labels)):
        generated_labels[i] = generated_labels[i].strip()
        generated_labels[i] = generated_labels[i].rstrip(".")
        generated_labels[i] = generated_labels[i].rstrip("\"")
        generated_labels[i] = generated_labels[i].rstrip("\"")
        generated_labels[i] = generated_labels[i].rstrip("\n")
        generated_labels[i] = generated_labels[i].lower()
    return generated_labels

# 2. check list    
# if one of the labels is not in the list, it appends it to the end of the list
# evaluates list afterwards
def check_list(generated_labels):
    for label in generated_labels:
        if label not in labels:
            # TODO (2): change to call evaluate list
            labels.append(label)
    evaluate_list(labels)

def evaluate_labels(given_labels):
    labels_string = "\n".join(labels)
    given_labels = "\n".join(given_labels)
    prompt = "<s> [INST] Your task is to evaluate whether or not a small set of labels should be added to a given large list of labels. You will be given both a large list of labels and a smaller list of labels to be potentially added to the larger list. You will evaluate each label in the smaller list and determine if it is worth adding to the larger list. If it is a redundant label that is too close to another label in the list, it should not be added. Keep in mind however, that it is possible for all the labels in the smaller list to be unique and worthwhile to add to the list. I will now provide the long list of labels:\n" + labels_string + "\nNow, I will provide you with the smaller list of labels that you will evaluate individually:\n" + given_labels + "Please respond in a format that can easily be converted to a python list by separating each label in your list with a ', '. [\INST] Sure, here is a string with all the non-rendundant labels that should be added to the longer list of labels, each separated by a comma, and I WILL NOT CONTINUE SPEAKING after the list:\n\n\""
    generated_labels = ""
    for response in client.generate_stream(prompt, max_new_tokens=100, repetition_penalty=1.2, stop_sequences=["<\s>", "\n", " \n"]):
        if not response.token.special:
            generated_labels += response.token.text
    print("Selected labels:\n" + GREEN + generated_labels + RESET)
    generated_labels = generated_labels.split(", ")
    for i in range(len(generated_labels)):
        generated_labels[i] = generated_labels[i].strip()
        generated_labels[i] = generated_labels[i].rstrip("\,")
        generated_labels[i] = generated_labels[i].rstrip(".")
        generated_labels[i] = generated_labels[i].rstrip("\"")
        generated_labels[i] = generated_labels[i].lstrip("\"")
        generated_labels[i] = generated_labels[i].rstrip("\n")
        generated_labels[i] = generated_labels[i].lower()
        if generated_labels[i] not in labels:
            labels.append(generated_labels[i])

# generate labels for each entry in the dataset
counter = 0
for index, row in df.iterrows():
    print("------------------------------------------------------------")
    print("Use Case:\n" + YELLOW + row['use_case'] + RESET + "\n")
    generated_labels = generate_labels(row['use_case'])
    #check_labels(generated_labels)
    evaluate_labels(generated_labels)
    print("\n\nCurrent list:" + ORANGE)
    for label in labels: print(label, end=", ")
    #print(", ".join(str(labels)))
    print(RESET + "\n")
    print("Number of elements: " + RED + str(len(labels)) + RESET)
    counter += 1
    #if counter > 5:
        #break

# save all labels to a separate file, used later for training
with open('labels.txt', 'w') as file:
    for label in labels:
        file.write(label + '\n')

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



        


    


    
    
    
