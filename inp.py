import numpy as np


def asc_raster(file, nan=False, dtype='int16'):
    """
    A function to import .ASC raster files
    :param file: string of file path with the '.asc' extension
    :param nan: boolean to convert nan values to np.nan
    :param dtype: string code to data type. Options: 'int16', 'int32', 'float32' etc
    :return: 1) metadata dictionary and 2) numpy 2d array
    """
    def_f = open(file)
    def_lst = def_f.readlines()
    def_f.close()
    #
    # get metadata constructor loop
    meta_lbls = ('ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'NODATA_value')
    meta_format = ('int', 'int', 'float', 'float', 'float', 'float')
    meta_dct = dict()
    for i in range(6):
        lcl_lst = def_lst[i].split(' ')
        lcl_meta_str = lcl_lst[len(lcl_lst) - 1].split('\n')[0]
        if meta_format[i] == 'int':
            meta_dct[meta_lbls[i]] = int(lcl_meta_str)
        else:
            meta_dct[meta_lbls[i]] = float(lcl_meta_str)
    #
    # array constructor loop:
    array_lst = list()
    for i in range(6, len(def_lst)):
        lcl_lst = def_lst[i].split(' ')[1:]
        lcl_lst[len(lcl_lst) - 1] = lcl_lst[len(lcl_lst) - 1].split('\n')[0]
        array_lst.append(lcl_lst)
    def_array = np.array(array_lst, dtype=dtype)
    #
    # replace NoData value by np.nan
    if nan:
        ndv = float(meta_dct['NODATA_value'])
        for i in range(len(def_array)):
            lcl_row_sum = np.sum((def_array[i] == ndv) * 1)
            if lcl_row_sum > 0:
                for j in range(len(def_array[i])):
                    if def_array[i][j] == ndv:
                        def_array[i][j] = np.nan
    return meta_dct, def_array