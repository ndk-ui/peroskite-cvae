import pandas as pd 
import numpy as np
import re, os

#Dataset Preparation
df = pd.read_csv('data/raw_data.csv')
replace_dict = {'CH3CH2NH3':'!', 'CH3CH2CH2NH3': '@','CH3CH2CH2CH2NH3':'#', 'CH3CH3CHNH3':'$', 'CH3CH3NH2':'%','CH3CNH2NH2':'^','CH3NH3':'-',
                'CNH2NH2NH2':'_','OHNH3':'+','HCNH2NH2':'{','CH3CH3CH3CH3N':'}','NH2NH3':'[','NH4':'&','CH3CH3CH3NH':'*','C3H6N2':'|',
                'C3H5N2':'/','Ge':']','Pb':',','Sn':'.','F3':'<','Cl3':'>','Br3':';','I3':'?'}

def processing_names(df, replacement):
    df = df.copy()
    pattern = re.compile('|'.join(map(re.escape, replacement.keys())))
    df["processed_names"] = df["Composition"].apply(lambda x: pattern.sub(lambda m: replacement[m.group(0)],x))
    unique_char = set("".join(df["processed_names"]))
    return df, unique_char
    
def one_hot_encode(composition, unique_char, maxlen = 3):
    comp_index = {char: index for index,char in enumerate(unique_char)}
    comp_len = min(len(composition), maxlen)
    
    indices = [comp_index[char] for char in composition[:comp_len]]
    comp_matrix = np.zeros((len(unique_char), maxlen), dtype = int)
    comp_matrix[indices, np.arange(len(indices))] = 1 
    return comp_matrix.flatten().tolist()

def processing_data(df, replacement=replace_dict):
    df, unique_char = processing_names(df, replacement)
    df['encoded_data'] = df['processed_names'].apply(lambda x: one_hot_encode(x, unique_char))
    return df, unique_char

def decode_one_hot(encoded_data, unique_char, maxlen = 3):
    comp_matrix = encoded_data.reshape((len(unique_char), maxlen))
    char_list = np.array(list(unique_char))
    indices = np.argmax(comp_matrix, axis = 0)
    mask = comp_matrix[indices, np.arange(maxlen)] == 1 
    decoded = "".join(char_list[indices][mask])
    return decoded 

def processing_char(decoded_str, replacement):
    reverse_dict = {value: key for key, value in replacement.items()}
    symbol = re.compile('|'.join(map(re.escape, reverse_dict.keys())))
    names = symbol.sub(lambda m: reverse_dict[m.group(0)],decoded_str)
    return names

def backprocess_one_hot(encoded_data, unique_char, maxlen = 3, replacement = replace_dict):
    decoded_str = decode_one_hot(encoded_data, unique_char, maxlen)
    return processing_char(decoded_str, replacement)

def save_data(dataframe, replacement = replace_dict):
    dataframe, unique_char = processing_data(dataframe, replacement)
    os.makedirs('data', exist_ok = True)
    dataframe.to_csv('data/processed_data.csv')
    with open('data/unique_char.txt', 'w') as f:
        for char in unique_char:
            f.write(f"{char}\n")
            
save_data(df)