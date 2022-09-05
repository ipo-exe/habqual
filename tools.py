import pandas as pd
import numpy as np


def convert_to_invest_tables(s_fpath_lulc_tbl, s_dir_output='C:/bin', b_random_fill=False, b_base=False, b_fut=False):
    """
    get InVEST-like tables
    :param s_fpath_lulc_tbl: string filepath to txt lulc table
    mandadory fields: Id, Name, Class (Threat)
    :param s_dir_output: string path to output directory
    :param b_random_fill: boolean to random fill
    :return:
    """

    df = pd.read_csv(s_fpath_lulc_tbl, sep=';')

    # get threats list
    df_lulc_threats = df.query('Class == "Threat"')
    print(df_lulc_threats.to_string())
    lst_threats = list(df_lulc_threats['Name'].values)
    lst_threats_upper = [t.upper().replace(' ', '_') for t in lst_threats]
    # deploy LULC dataframe
    df_lulc = pd.DataFrame(
        {
            'LULC': df['Id'].values,
            'NAME': df['Name'].values,
            'HABITAT': 1.0
        }
    )
    # append threats fields
    for t in lst_threats_upper:
        df_lulc[t] = 0.0

    # fill
    if b_random_fill:
        for i in range(len(df_lulc)):
            if i > 1 and i < 17:
                df_lulc['HABITAT'].values[i] = np.round(np.random.randint(0, 40) / 100, 1)
        for f in lst_threats_upper:
            df_lulc[f] = np.round(np.random.random(size=len(df_lulc)), 1)

    print(df_lulc.to_string())

    # deploy threats dataframe
    df_threats = pd.DataFrame(
        {
            'THREAT': lst_threats_upper,
            'MAX_DIST': 1,
            'WEIGHT': 1,
            'DECAY': 'exponential',
            'CUR_PATH': '',
        }
    )
    if b_base:
        df_threats['BASE_PATH'] = ''
    if b_fut:
        df_threats['FUT_PATH'] = ''
    for i in range(len(df_threats)):
        s_lcl_threat = df_threats['THREAT'].values[i].lower()
        df_threats['CUR_PATH'].values[i] = 'cur_{}.tif'.format(s_lcl_threat)
        if b_base:
            df_threats['BASE_PATH'].values[i] = 'base_{}.tif'.format(s_lcl_threat)
        if b_fut:
            df_threats['FUT_PATH'].values[i] = 'fut_{}.tif'.format(s_lcl_threat)
    print(df_threats.to_string())
    # export files
    df_lulc.to_csv('{}/lulc.csv'.format(s_dir_output), sep=',', index=False)
    df_threats.to_csv('{}/threats.csv'.format(s_dir_output), sep=',', index=False)


'''
get_invest_tables(
    s_fpath_lulc_tbl=r"C:\000_myFiles\myDrive\myProjects\124_consultoria_Teia_2022\outputs\basic\bahia_lulc_2021_table.txt",
    s_dir_output='C:/bin/teia',
    b_random_fill=True,
    b_base=False,
    b_fut=False
)
'''
