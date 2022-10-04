import pandas as pd
from scipy.io import loadmat
from comtrade import Comtrade
import os


# Finding all the different type of files that needs to be converted
def files2convert(path):

    if not os.path.exists(path+"/Converted_files/"):
        os.makedirs(path+'/Converted_files/')

    com_files = []
    mat_files = []
    inf_files = []
    out_files = []

    for files in os.listdir(path):
        if files.endswith('.cfg'):  # Searching files which end with '.cfg' (comtrade files) extension and string them in a list
            com_files.append(files)
        elif files.endswith(('.mat', '.MAT')):  # Searching files which end with '.mat' (matlab files) extension and string them in a list
            mat_files.append(files)
        elif files.endswith('.inf'):  # Searching files which end with '.inf' (pscad info file) extension and string them in a list
            inf_files.append(files)
        elif files.endswith('.out'):  # Searching files which end with '.out' (pscad data files) extension and string them in a list
            out_files.append(files)

    return com_files, mat_files, inf_files, out_files


# Function to convert comtrade file to csv
def comtrade2csv(cfg_file, dat_file, folder_path):
    rec = Comtrade()  # Making an instance of Comtrade class
    rec.load(folder_path + '\\' + cfg_file, folder_path + '\\' + dat_file)  # Loading the data

    df_analog = pd.DataFrame(rec.analog, index=rec.analog_channel_ids).transpose()  # Creating dataframe for analog data
    df_digital = pd.DataFrame(rec.status, index=rec.status_channel_ids).transpose()  # Creating dataframe for digital data

    combined_df = pd.concat([df_analog, df_digital], axis=1)  # Combining the 2 dataframes into 1
    os.chdir(folder_path+'/Converted_files/')
    combined_df.to_csv(cfg_file[:-4]+'_csv.csv', index=False)  # Converting the dataframe into csv
    os.chdir(folder_path)

    print('{} successfully converted'.format(cfg_file))
    return None


# Function to convert matlab files to csv:
def mat2csv(file, folder_path):
    # Function to convert to '*.mat' files to '*.csv' file.

    # Changing the working directory to 'Output' folder
    os.chdir(folder_path + '/Converted_files/')

    file_path = folder_path + '\\' + file

    p = []
    x = loadmat(file_path)
    for i in list(x.keys())[3:]:
        if len(x[i]) > 1:
            p.append(i)

    df = pd.DataFrame()

    for i in p:
        df = pd.concat([df, pd.DataFrame(x[i], columns=[i])], axis=1)
        df.to_csv(file[:-4] + '_csv.csv', index=False)

    print(f'\'{file}\' successfully converted')
    os.chdir(folder_path)

    return None


# Convert pscad files (.inf, .out) files to csv:
def get_columns(inf_file):
    # Helper function for function 'pscad2csv', returns the column names using the '*.inf' file.

    complete_inf_data = []
    count1 = 0
    with open(inf_file) as file1:
        while file1.readline():
            count1 += 1
    with open(inf_file) as file1:
        for _ in range(count1):
            complete_inf_data.append(file1.readline().split())

    field_names = ['Time']
    for k in range(len(complete_inf_data)):
        field_names.append(complete_inf_data[k][2].split('"')[1])
    return field_names


def pscad2csv(inf_files, out_files, folder_path):
    # Function to convert '*.inf', '*.out' files to '*.csv' file.

    column_names = get_columns(inf_files)
    col = column_names.copy()

    data_files = []
    for files in out_files:
        if files[:-7] == inf_files[:-4]:
            data_files.append(files)

    count = 0
    x = len(column_names) % 11
    df = pd.DataFrame()
    for files in data_files:
        if count < x:
            df1 = pd.read_fwf(files, header=None)
            y = dict(zip(df.columns, col[0:11]))
            del col[1:11]
            df1.rename(columns=y, inplace=True)
            df = pd.concat([df, df1], axis=1)
            count += 1
        else:
            df1 = pd.read_fwf(files, header=None)
            y = dict(zip(df.columns, column_names[0:]))
            df1.rename(columns=y, inplace=True)
            df = pd.concat([df, df1], axis=1)
            count += 1
        print('\'{}\' successfully converted'.format(files))

    y = dict(zip(df.columns, column_names[0:11]))
    df.rename(columns=y, inplace=True)

    # Remove duplicate columns pandas DataFrame
    df2 = df.loc[:, ~df.columns.duplicated()]

    os.chdir(folder_path + '/Converted_files/')
    df2.to_csv(inf_files[:-4] + '_csv.csv', index=False)
    os.chdir(folder_path)

    print(f'All files converted for \'{inf_files}\', COMPLETE')
    return None
