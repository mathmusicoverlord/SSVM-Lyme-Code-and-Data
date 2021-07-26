from calcom import calcom
import calcom.io
import pandas as pd
import string
import numpy as np
import re
import missingpy
from copy import deepcopy

import numpy as np


###############################################################################################################


class MedianPolish:
    """Fits an additive model using Tukey's median polish algorithm"""

    def __init__(self, array):
        """Get numeric data from numpy ndarray to self.tbl, keep the original copy in tbl_org"""
        if isinstance(array, np.ndarray):
            self.tbl_org = array
            self.tbl = self.tbl_org.copy()
        else:
            raise TypeError('Expected the argument to be a numpy.ndarray.')

    def median_polish(self, max_iterations=10, method='median'):
        """
            Implements Tukey's median polish alghoritm for additive models
            method - default is median, alternative is mean. That would give us result equal ANOVA.
        """

        grand_effect = 0
        median_row_effects = 0
        median_col_effects = 0
        row_effects = np.zeros(shape=self.tbl.shape[0])
        col_effects = np.zeros(shape=self.tbl.shape[1])

        for i in range(max_iterations):
            if method == 'median':
                row_medians = np.median(self.tbl, 1)
                row_effects += row_medians
                median_row_effects = np.median(row_effects)
            elif method == 'average':
                row_medians = np.average(self.tbl, 1)
                row_effects += row_medians
                median_row_effects = np.average(row_effects)
            grand_effect += median_row_effects
            row_effects -= median_row_effects
            self.tbl -= row_medians[:, np.newaxis]

            if method == 'median':
                col_medians = np.median(self.tbl, 0)
                col_effects += col_medians
                median_col_effects = np.median(col_effects)
            elif method == 'average':
                col_medians = np.average(self.tbl, 0)
                col_effects += col_medians
                median_col_effects = np.average(col_effects)

            self.tbl -= col_medians

            grand_effect += median_col_effects

        # return grand_effect, col_effects, row_effects, self.tbl, self.tbl_org
        return


#################################################################################################################

def plot_pandas(df: pd.DataFrame, figsize: tuple, grp_colors: str, grp_mrkrs: str, mrkr_list: list, title: str,
                dim: int, x_label: str, y_label: str, grp_label: str, mrkr_size:int, save_name=None):

    from matplotlib import pyplot as plt
    from matplotlib.pyplot import Axes
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns

    num_colors = len(df[grp_colors].unique())
    palette = sns.color_palette("Accent", num_colors)

    if dim == 3:
        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig)
        for i, grp0 in enumerate(df.groupby(grp_colors).groups.items()):
            grp_name0 = grp0[0]
            grp_idx0 = grp0[1]

            for j, grp1 in enumerate(df.groupby(grp_mrkrs).groups.items()):
                grp_name1 = grp1[0]
                grp_idx1 = grp1[1]

                grp_idx = list(set(grp_idx0).intersection(set(grp_idx1)))
                x = df.iloc[grp_idx, 0]
                y = df.iloc[grp_idx, 1]
                z = df.iloc[grp_idx, 2]
                label = df.loc[df.index[grp_idx], grp_label].unique().item()

                ax.scatter(x, y, z, label=label, c=np.array(palette[i]).reshape(1, -1), marker=mrkr_list[j], s=mrkr_size, edgecolors='k', linewidths=.5)

        ax.axis('off')
    elif dim == 2:
        fig, ax = plt.subplots(1, figsize=figsize)
        for i, grp0 in enumerate(df.groupby(grp_colors).groups.items()):
            grp_name0 = grp0[0]
            grp_idx0 = grp0[1]

            for j, grp1 in enumerate(df.groupby(grp_mrkrs).groups.items()):
                grp_name1 = grp1[0]
                grp_idx1 = grp1[1]

                grp_idx = list(set(grp_idx0).intersection(set(grp_idx1)))
                if len(grp_idx) > 0:
                    x = df.iloc[grp_idx, 0]
                    y = df.iloc[grp_idx, 1]
                    label = df.loc[df.index[grp_idx], grp_label].unique().item()

                    ax.scatter(x, y, label=label, c=np.array(palette[i]).reshape(1, -1), marker=mrkr_list[j], s=mrkr_size, edgecolors='k', linewidths=.5)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='best', bbox_to_anchor=(1, 0.5), fontsize=15, title=grp_colors)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    plt.show()
    if not (save_name is None):
        plt.savefig(fname=save_name+'.png', format='png')


#################################################################################################################
class CCLymeMetaboliteSet(calcom.io.CCDataSet):
    '''Subclass of CCDataSet for metabolomics data and calculations for Lyme Disease'''

    def __init__(self, data_file_path="", data_format="", metadata_file_path="", metadata_format="",
                 osmolality_file_path="", offset=11, data=None, metadata=None):
        '''
        Initializes the CCLymeMetaboliteSet class.

        Inputs: See load_csv.

        Outputs: Instance of CCLymeMetaboliteSet.
        '''
        super().__init__(fname="", preload=True)

        self.varattr = pd.DataFrame()
        self.name = ''
        self.imputation_method = None
        self.normalization_method = None
        self.new_data = np.array(self.data)
        self.missing_features = pd.DataFrame()
        self.removed_features = pd.DataFrame()
        self.path = '~/Dropbox/ELvsHC XCMS data/'
        self.offset = offset

        if data is not None:
            if metadata is not None:
                self.create(datapoints=data, metadata=metadata)
            else:
                self.create(datapoints=data, metadata=None)

        if data_file_path != "":
            self.load_csv(data_file_path, data_format, metadata_file_path, metadata_format, osmolality_file_path,
                          offset)

    def generate_data_matrix(self, **kwargs):
        '''
        THIS IS A FORK FROM CALCOMS generate_data_matrix USED TO GENERATE NORMALIZED DATA

        Outputs an numpy.ndarray of the data matrix, to be used in
        classification algorithms.

        Inputs: none
        Outputs: numpy.ndarray of two dimensions.

        Optional inputs (kwargs):
            idx: List of integers indicating the subset of the data
                to pull.
            features: String/List of integers, indicating the specific
                dimensions in the data to pull.
            attr_list: List of strings indicating attribute values to append to
                the raw data. The attributes are concatenated in the order they
                appear in the argument list, whether they are scalars or
                vectors.
                Default: empty list
            use_data: Boolean.If False, the CCDataPoint is ignored, and only
                the values referenced in attr_list are used.
                Default: True

        If a list of integers is specified, only the specified subset of the
        data is pulled.

        Examples:
            # Full dataset pulled
            data0 = ccd.generate_data_matrix()
            # Only data points 2,4,6 pulled
            data1 = ccd.generate_data_matrix(idx=[2,4,6])
        '''
        import numpy as np
        from calcom.io import CCList
        from calcom.utils import type_functions as tf

        from calcom.io.loadsave_hdf import load_CCDataPoint

        n = len(self.data)

        if ('idx_list' in list(kwargs.keys())) and ('idx' not in list(kwargs.keys())):
            # re-instate backwards support
            idx = kwargs['idx_list']
        else:
            idx = kwargs.get('idx', np.arange(0, len(self.data), dtype=np.int64))
        #

        if ('feature_set' in list(kwargs.keys())) and ('features' not in list(kwargs.keys())):
            # re-instate backwards support
            features = kwargs['feature_set']
        else:
            features = kwargs.get('features', [])
        #

        #        features = kwargs.get('features', [])

        attr_list = kwargs.get('attr_list', CCList())
        use_data = kwargs.get('use_data', True)

        #

        if tf.is_string_like(features):
            if len(features) > 0:
                if features not in self.feature_sets:
                    print('Warning: feature set %s not found in CCDataSet. Using all features.' % features)
                    features = []
                else:
                    features = self.feature_sets[features]
                #
            else:
                features = []
            #
        #

        output = CCList()

        for i in idx:
            elem = CCList()
            d = self.data[i]
            d_new = self.new_data[i, :]
            if not d._loaded:
                d = load_CCDataPoint(self.fname, d._id)
                d._loaded = True
                self.data[i] = d
            #

            if use_data:
                if len(features) == 0:

                    # elem.append( self.data[i].flatten() )   # Row-wise concatenation if it is a 2D array.
                    elem.append(d_new)
                else:
                    # Need to break into cases to do the slicing properly.
                    # Only two-dimensional arrays supported. TODO: is there a
                    # way to slice on the first dimension (python counting)
                    # if the order of the array is unknown? Probably won't be needed.
                    if len(np.shape(d)) == 2:
                        elem.append(d_new[:, features])
                    elif len(np.shape(d)) == 1:
                        elem.append(d_new[features])
                    #
                #
            #
            for attr in attr_list:
                elem.append(getattr(getattr(d, attr), 'value'))
            #

            # elem = np.hstack(elem)
            output.append(np.hstack(elem))
        #

        return np.array(output)

    def data_clear(self):
        '''
          Clears any normalization to the data and resets it to the original values.

          Input: None

          Output: Inplace method.
          '''
        self.imputation_method = None
        self.normalization_method = None
        self.new_data = np.array(self.data)

    def load_csv(self, data_file_path="", data_format="", metadata_file_path="", metadata_format="",
                 osmolality_file_path="", offset=11):

        '''
          Loads a metabolomics data set from a csv file along with an optional metadata csv file

          Input:
              data_file_path: string. File path of data set
              data_format: string. Indicates the format of the data. e.g. 'XCMS', 'MassHunter', 'Skyline', 'MZmine', 'Urine'
              metadata_file_path: string. File path of metadata (optional)
              metadata_format: string. Format of metadata. e.g. 0, 1
              osmolality_file_path: string. File path for osmolality data
              offset: int. Indicates the base-0 position of the first column with data in the data file
          Output:
             Inplace method.
          '''

        # Name the CCLymeMetaboliteSet
        from pathlib import Path
        self.name = Path(data_file_path).stem

        # Read in data csv as Pandas DataFrame
        if data_format == 'Skyline':
            if metadata_format == 1:
                df_data = pd.read_csv(data_file_path, index_col=False)
                df_data.rename(columns={'Replicate': 'CSU'}, inplace=True)
                df_data['CSU'] = df_data['CSU'].str.replace('Nurul072419_Urine_', '', regex=True)
                df_data['CSU'] = df_data['CSU'].str.replace('Pos', '', regex=True)
                df_data['CSU'] = df_data['CSU'].str.replace('_', '-', regex=True)
                df_data['CSU'] = df_data['CSU'].str.replace('LD', 'LD-', regex=True)
                # df_data.dropna(axis=0, how="any", inplace=True) # Drop rows with NA entries

                # Remove null rows for rt, mz, area
                df_data.drop(index=df_data[df_data['Product Mz'].isnull()].index.tolist(), inplace=True)
                df_data.drop(index=df_data[df_data['Retention Time'].isnull()].index.tolist(), inplace=True)
                df_data.drop(index=df_data[df_data['Area'].isnull()].index.tolist(), inplace=True)
            elif metadata_format == 0:
                df_data = pd.read_csv(data_file_path, index_col=False)
                df_data.rename(columns={'Replicate': 'CSU'}, inplace=True)
                df_data['CSU'] = df_data['CSU'].str.replace('g', '', regex=True)
                # Remove null rows for rt, mz, area
                df_data.drop(index=df_data[df_data['Product Mz'].isnull()].index.tolist(), inplace=True)
                df_data.drop(index=df_data[df_data['Retention Time'].isnull()].index.tolist(), inplace=True)
                df_data.drop(index=df_data[df_data['Area'].isnull()].index.tolist(), inplace=True)

            for column in df_data.columns:
                df_data.rename(columns={column: column.strip()}, inplace=True)

        elif data_format in ['XCMS', 'Urine']:
            df_data = pd.read_csv(data_file_path, index_col=0)

            # Fill in data
            data = np.array(df_data.values)[:, offset:]
            data = data.astype('float64')

        elif data_format == 'MassHunter':
            # Read in data file
            df_data = pd.read_csv(data_file_path, index_col=0)

            # Grab last row for metabolite names
            df_last_row = df_data.iloc[-1, :]
            var_names = []

            for entry in df_last_row:
                if (not pd.isnull(entry)) and ('Results' in entry):
                    var_names.append(entry.replace(' Results', ''))

            # Drop the last row
            df_data.drop(index=df_data.index[-1], inplace=True)

            # Remove rows that are repeats
            for idx in df_data['Name']:
                if 'Repeat' in idx:
                    label = df_data.index[df_data['Name'] == idx]
                    df_data.drop(index=label[0], inplace=True)

            # Reformat the sample labels
            df_data.rename(columns={'Name': 'CSU'}, inplace=True)
            df_data['CSU'] = df_data['CSU'].str.replace('Nurul072419_Urine_', '', regex=True)
            df_data['CSU'] = df_data['CSU'].str.replace('Pos', '', regex=True)
            df_data['CSU'] = df_data['CSU'].str.replace('_', '-', regex=True)
            df_data['CSU'] = df_data['CSU'].str.replace('LD', 'LD-', regex=True)
            for column in df_data.columns:
                df_data.rename(columns={column: column.strip()}, inplace=True)

        # Check for metadata
        if metadata_file_path != "":

            # Check format of metadata
            if metadata_format == 0:
                if data_format == 'XCMS':
                    # Create metadata DataFrame
                    df_metadata = pd.read_csv(metadata_file_path, index_col=6)

                    # Strip leading and trailing whitespace from column names
                    metadata_column_dict = dict()
                    for column in df_metadata.columns:
                        metadata_column_dict[column] = str(column).strip()
                    df_metadata.rename(columns=metadata_column_dict, inplace=True)

                    # Create metadata for samples and variables
                    df_metadata_ccd = pd.DataFrame(index=df_data.columns[offset:],
                                                   columns=['Generation', 'Culture Result', 'Age', 'Gender', 'Race',
                                                            'Previous Infection',
                                                            'Region', 'Presence of EM',
                                                            'Size of Largest EM', 'Duration of EM',
                                                            'Immunocomprimised',
                                                            'Temperature', 'Loss of Appetite',
                                                            'Joint Pain', 'Cough', 'Dizziness',
                                                            'Fatigue', 'Feverish', 'Headache',
                                                            'Muscle Pain / Body Aches',
                                                            'Nausea/Vomiting', 'Tingling / Numbeness',
                                                            'Stiff Neck /Neck Pain',
                                                            'Memory/Concentration Problems',
                                                            'Swollen Lymph Nodes',
                                                            'Blood Lyme CultureResult',
                                                            'Skin /Biopsy Lyme CultureResult',
                                                            'Blood Lyme PCR Result',
                                                            'Skin /Biopsy Lyme PCR Result',
                                                            'Outcome of Follow-up visit',
                                                            '2-Tier VIDAS / Marblot',
                                                            '2- Tier C6+ / Marblot',
                                                            '2-Tier VIDAS/C6+'])

                    for s in df_metadata_ccd.index:
                        if not (s in self.varattr.columns):

                            name_date = s.split(sep='.')

                            # Get Name for sample
                            df_metadata_ccd.at[s, 'Name'] = name_date[0]

                            # Get Date for sample
                            if len(name_date) > 1:
                                df_metadata_ccd.at[s, 'Date'] = re.search(r'\d+', name_date[1]).group()

                            # Get Health State
                            if 'L' in name_date[0]:
                                df_metadata_ccd.at[s, 'Health State'] = 'Lyme'
                            if 'H' in name_date[0]:
                                df_metadata_ccd.at[s, 'Health State'] = 'Healthy'

                            # Get Disease State
                            if 'ED' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'EDL'
                            if 'EL' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'ELL'
                            if 'CW' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'HCW'
                            if 'CN' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'HCN'

                            # Fill all other metadata
                            if metadata_file_path != "":

                                # Create dictionary between CCD metadata and clinical data
                                metadata_dict = dict(zip(['Generation', 'Culture Result', 'Age', 'Gender', 'Race',
                                                          'Previous Infection',
                                                          'Region', 'Presence of EM',
                                                          'Size of Largest EM', 'Duration of EM',
                                                          'Immunocomprimised',
                                                          'Temperature', 'Loss of Appetite',
                                                          'Joint Pain', 'Cough', 'Dizziness',
                                                          'Fatigue', 'Feverish', 'Headache',
                                                          'Muscle Pain / Body Aches',
                                                          'Nausea/Vomiting', 'Tingling / Numbeness',
                                                          'Stiff Neck /Neck Pain',
                                                          'Memory/Concentration Problems',
                                                          'Swollen Lymph Nodes',
                                                          'Blood Lyme CultureResult',
                                                          'Skin /Biopsy Lyme CultureResult',
                                                          'Blood Lyme PCR Result',
                                                          'Skin /Biopsy Lyme PCR Result',
                                                          'Outcome of Follow-up visit',
                                                          '2-Tier VIDAS / Marblot',
                                                          '2- Tier C6+ / Marblot',
                                                          '2-Tier VIDAS/C6+'],
                                                         ['Generation',
                                                          'Lyme Culture Result (Neg or at least one sample type pos)',
                                                          'Age', 'Gender', 'Race', 'Lyme Disease', 'State of exposure',
                                                          'Presence of EM', 'Size of Largest EM (cm2)',
                                                          'Duration of EM (days)', 'Immunocomprimised',
                                                          'Baseline Temp / High Temp', 'Loss of Appetite',
                                                          'Joint Pain', 'Cough', 'Dizziness',
                                                          'Fatigue', 'Feverish', 'Headache',
                                                          'Muscle Pain / Body Aches',
                                                          'Nausea/Vomiting', 'Tingling / Numbeness',
                                                          'Stiff Neck /Neck Pain',
                                                          'Memory/Concentration Problems',
                                                          'Swollen Lymph Nodes',
                                                          'Blood Lyme CultureResult',
                                                          'Skin /Biopsy Lyme CultureResult',
                                                          'Blood Lyme PCR Result',
                                                          'Skin /Biopsy Lyme PCR Result',
                                                          'Outcome of Follow-up visit',
                                                          '2-Tier VIDAS / Marblot',
                                                          '2- Tier C6+ / Marblot',
                                                          '2-Tier VIDAS/C6+']))

                                # Check format of number
                                if name_date[0] in df_metadata.index:
                                    index = name_date[0]
                                else:
                                    index = df_metadata_ccd.at[s, 'Disease State'] + '0' + str(
                                        int(re.search(r'\d+', s).group()))

                                for key in metadata_dict:
                                    df_metadata_ccd.at[s, key] = df_metadata.at[index, metadata_dict[key]]

                    # Remove inconsistencies of names
                    df_metadata_ccd = df_metadata_ccd.replace({'No': 'no', 'Yes': 'yes', 'No ': 'no', 'Yes ': 'yes',
                                                               'Caucasion': 'Caucasian', 'nan ': 'nan'})

                    # Strip leading and trailing whitespace
                    df_metadata_ccd = df_metadata_ccd.astype(str)
                    for column in df_metadata_ccd.columns:
                        df_metadata_ccd.loc[:, column] = df_metadata_ccd.loc[:, column].str.strip()
                        df_metadata_ccd.loc[:, column] = df_metadata_ccd.loc[:, column].str.lower()

                    # Fill-in Date
                    df_metadata_ccd = df_metadata_ccd.dropna(subset=['Date'])
                    df_metadata_ccd.loc[:, 'Date'] = df_metadata_ccd['Date'].value_counts().idxmax()

                    # Convert DataFrame to numpy array
                    metadata_ccd = np.array(df_metadata_ccd.columns).reshape(1, len(df_metadata_ccd.columns))
                    metadata_ccd = np.concatenate([metadata_ccd, np.array(df_metadata_ccd)], axis=0)

                elif data_format == 'Skyline':
                    # Create metadata DataFrame
                    df_metadata = pd.read_csv(metadata_file_path, index_col=6)

                    # Strip leading and trailing whitespace from column names
                    metadata_column_dict = dict()
                    for column in df_metadata.columns:
                        metadata_column_dict[column] = str(column).strip()
                    df_metadata.rename(columns=metadata_column_dict, inplace=True)

                    # Create metadata for samples and variables
                    df_metadata_ccd = pd.DataFrame(index=set(df_data['CSU']),
                                                   columns=['Generation', 'Culture Result', 'Age', 'Gender', 'Race',
                                                            'Previous Infection',
                                                            'Region', 'Presence of EM',
                                                            'Size of Largest EM', 'Duration of EM',
                                                            'Immunocomprimised',
                                                            'Temperature', 'Loss of Appetite',
                                                            'Joint Pain', 'Cough', 'Dizziness',
                                                            'Fatigue', 'Feverish', 'Headache',
                                                            'Muscle Pain / Body Aches',
                                                            'Nausea/Vomiting', 'Tingling / Numbeness',
                                                            'Stiff Neck /Neck Pain',
                                                            'Memory/Concentration Problems',
                                                            'Swollen Lymph Nodes',
                                                            'Blood Lyme CultureResult',
                                                            'Skin /Biopsy Lyme CultureResult',
                                                            'Blood Lyme PCR Result',
                                                            'Skin /Biopsy Lyme PCR Result',
                                                            'Outcome of Follow-up visit',
                                                            '2-Tier VIDAS / Marblot',
                                                            '2- Tier C6+ / Marblot',
                                                            '2-Tier VIDAS/C6+'])

                    for s in df_metadata_ccd.index:
                        if not (s in self.varattr.columns):

                            name_date = s.split(sep='-')

                            # Get Name for sample
                            df_metadata_ccd.at[s, 'Name'] = name_date[0]

                            # Get Date for sample
                            if len(name_date) > 1:
                                df_metadata_ccd.at[s, 'Date'] = re.search(r'\d+', name_date[1]).group()
                            else:
                                df_metadata_ccd.at[s, 'Date'] = 'None'

                            # Get Health State
                            if 'L' in name_date[0]:
                                df_metadata_ccd.at[s, 'Health State'] = 'Lyme'
                            if 'H' in name_date[0]:
                                df_metadata_ccd.at[s, 'Health State'] = 'Healthy'

                            # Get Disease State
                            if 'ED' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'EDL'
                            if 'EL' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'ELL'
                            if 'CW' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'HCW'
                            if 'CN' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'HCN'
                            if 'Hu' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'HCHu'

                            # Fill all other metadata
                            if metadata_file_path != "":

                                # Create dictionary between CCD metadata and clinical data
                                metadata_dict = dict(zip(['Generation', 'Culture Result', 'Age', 'Gender', 'Race',
                                                          'Previous Infection',
                                                          'Region', 'Presence of EM',
                                                          'Size of Largest EM', 'Duration of EM',
                                                          'Immunocomprimised',
                                                          'Temperature', 'Loss of Appetite',
                                                          'Joint Pain', 'Cough', 'Dizziness',
                                                          'Fatigue', 'Feverish', 'Headache',
                                                          'Muscle Pain / Body Aches',
                                                          'Nausea/Vomiting', 'Tingling / Numbeness',
                                                          'Stiff Neck /Neck Pain',
                                                          'Memory/Concentration Problems',
                                                          'Swollen Lymph Nodes',
                                                          'Blood Lyme CultureResult',
                                                          'Skin /Biopsy Lyme CultureResult',
                                                          'Blood Lyme PCR Result',
                                                          'Skin /Biopsy Lyme PCR Result',
                                                          'Outcome of Follow-up visit',
                                                          '2-Tier VIDAS / Marblot',
                                                          '2- Tier C6+ / Marblot',
                                                          '2-Tier VIDAS/C6+'],
                                                         ['Generation',
                                                          'Lyme Culture Result (Neg or at least one sample type pos)',
                                                          'Age', 'Gender', 'Race', 'Lyme Disease', 'State of exposure',
                                                          'Presence of EM', 'Size of Largest EM (cm2)',
                                                          'Duration of EM (days)', 'Immunocomprimised',
                                                          'Baseline Temp / High Temp', 'Loss of Appetite',
                                                          'Joint Pain', 'Cough', 'Dizziness',
                                                          'Fatigue', 'Feverish', 'Headache',
                                                          'Muscle Pain / Body Aches',
                                                          'Nausea/Vomiting', 'Tingling / Numbeness',
                                                          'Stiff Neck /Neck Pain',
                                                          'Memory/Concentration Problems',
                                                          'Swollen Lymph Nodes',
                                                          'Blood Lyme CultureResult',
                                                          'Skin /Biopsy Lyme CultureResult',
                                                          'Blood Lyme PCR Result',
                                                          'Skin /Biopsy Lyme PCR Result',
                                                          'Outcome of Follow-up visit',
                                                          '2-Tier VIDAS / Marblot',
                                                          '2- Tier C6+ / Marblot',
                                                          '2-Tier VIDAS/C6+']))

                                # Check format of number
                                if name_date[0] in df_metadata.index:
                                    index = name_date[0]
                                else:
                                    index = df_metadata_ccd.at[s, 'Disease State'] + '0' + str(
                                        int(re.search(r'\d+', s).group()))

                                for key in metadata_dict:
                                    df_metadata_ccd.at[s, key] = df_metadata.at[index, metadata_dict[key]]

                    # Remove inconsistencies of names
                    df_metadata_ccd = df_metadata_ccd.replace({'No': 'no', 'Yes': 'yes', 'No ': 'no', 'Yes ': 'yes',
                                                               'Caucasion': 'Caucasian', 'nan ': 'nan'})

                    # Order by Name
                    df_metadata_ccd.sort_index(inplace=True)

                    # Strip leading and trailing whitespace
                    df_metadata_ccd = df_metadata_ccd.astype(str)
                    for column in df_metadata_ccd.columns:
                        df_metadata_ccd.loc[:, column] = df_metadata_ccd.loc[:, column].str.strip()
                        df_metadata_ccd.loc[:, column] = df_metadata_ccd.loc[:, column].str.lower()

                    # Convert DataFrame to numpy array
                    metadata_ccd = np.array(df_metadata_ccd.columns).reshape(1, len(df_metadata_ccd.columns))
                    metadata_ccd = np.concatenate([metadata_ccd, np.array(df_metadata_ccd)], axis=0)

            elif metadata_format == 1:
                # Create metadata DataFrame
                df_metadata = pd.read_csv(metadata_file_path, index_col=0)
                df_metadata.index.str.strip()

                # Create ccd metadata dataframe
                if data_format in ['Urine', 'MassHunter']:
                    df_metadata_ccd = pd.DataFrame(index=df_data['CSU'], columns=df_metadata.columns)
                elif data_format == 'Skyline':
                    df_metadata_ccd = pd.DataFrame(index=set(df_data['CSU']), columns=df_metadata.columns)

                df_metadata_ccd.index.str.strip()

                # Strip whitespace and make all lowercase
                for column in df_metadata.columns:
                    for sample in df_metadata.index:
                        s = df_metadata.at[sample, column]
                        if type(s) == str:
                            if column in ['Site', 'Current Label']:
                                s = s.upper()
                            elif column == 'Race':
                                s = string.capwords(s, sep='/')
                            else:
                                s = string.capwords(s)

                            df_metadata.at[sample, column] = s.strip()

                # Fill metadata for ccd
                for sample in (set(df_metadata.index) & set(df_metadata_ccd.index)):
                    df_metadata_ccd.loc[sample, :] = df_metadata.loc[sample, :]

                if osmolality_file_path != "":
                    # Fill in osmolality
                    df_osmolality = pd.read_csv(osmolality_file_path, index_col=6)
                    df_metadata_ccd.loc[:, 'mOSM/ kg H2O #1'] = 'nan'
                    df_metadata_ccd.loc[:, 'mOSM/ kg H2O #2'] = 'nan'
                    df_metadata_ccd.loc[:, 'mOSM/ kg H2O #3'] = 'nan'

                    for sample in (set(df_osmolality.index) & set(df_metadata_ccd.index)):
                        df_metadata_ccd.at[sample, 'mOSM/ kg H2O #1'] = df_osmolality.at[sample, 'mOSM/ kg H2O #1']
                        df_metadata_ccd.at[sample, 'mOSM/ kg H2O #2'] = df_osmolality.at[sample, 'mOSM/ kg H2O #2']
                        df_metadata_ccd.at[sample, 'mOSM/ kg H2O #3'] = df_osmolality.at[sample, 'mOSM/ kg H2O #3']

                df_metadata_ccd.reset_index(inplace=True)

                if data_format in ['Urine', 'MassHunter']:
                    df_metadata_ccd.rename(columns={'CSU': 'CSU_ID'}, inplace=True)
                elif data_format == 'Skyline':
                    df_metadata_ccd.rename(columns={'index': 'CSU_ID'}, inplace=True)

                # Sort Metadata by CSU_ID
                df_metadata_ccd.sort_values(by=['CSU_ID'], inplace=True)

                # Convert DataFrame to numpy array
                metadata_ccd = np.array(df_metadata_ccd.columns).reshape(1, len(df_metadata_ccd.columns))
                metadata_ccd = np.concatenate([metadata_ccd, np.array(df_metadata_ccd)], axis=0)

        else:
            if metadata_format == 0:
                if data_format == 'XCMS':
                    # Create metadata for samples and variables
                    df_metadata_ccd = pd.DataFrame(index=df_data.columns[offset:],
                                                   columns=['Name', 'Date', 'Health State', 'Disease State'])
                    for s in df_metadata_ccd.index:
                        if not (s in self.varattr.columns):

                            name_date = s.split(sep='.')

                            # Get Name for sample
                            df_metadata_ccd.at[s, 'Name'] = name_date[0]

                            # Get Date for sample
                            if len(name_date) > 1:
                                df_metadata_ccd.at[s, 'Date'] = re.search(r'\d+', name_date[1]).group()

                            # Get Health State
                            if 'L' in name_date[0]:
                                df_metadata_ccd.at[s, 'Health State'] = 'Lyme'
                            if 'H' in name_date[0]:
                                df_metadata_ccd.at[s, 'Health State'] = 'Healthy'

                            # Get Disease State
                            if 'ED' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'EDL'
                            if 'EL' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'ELL'
                            if 'CW' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'HCW'
                            if 'CN' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'HCN'
                            if 'Hu' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'HCHu'
                                # Remove inconsistencies of names
                                df_metadata_ccd = df_metadata_ccd.replace(
                                    {'No': 'no', 'Yes': 'yes', 'No ': 'no', 'Yes ': 'yes',
                                     'Caucasion': 'Caucasian', 'nan ': 'nan'})

                elif data_format == 'Skyline':
                    # Create metadata for samples and variables
                    df_metadata_ccd = pd.DataFrame(index=set(df_data['CSU']),
                                                   columns=['Name', 'Date', 'Health State', 'Disease State'])
                    for s in df_metadata_ccd.index:
                        if not (s in self.varattr.columns):

                            name_date = s.split(sep='-')

                            # Get Name for sample
                            df_metadata_ccd.at[s, 'Name'] = name_date[0]

                            # Get Date for sample
                            if len(name_date) > 1:
                                df_metadata_ccd.at[s, 'Date'] = re.search(r'\d+', name_date[1]).group()
                            else:
                                df_metadata_ccd.at[s, 'Date'] = 'None'

                            # Get Health State
                            if 'L' in name_date[0]:
                                df_metadata_ccd.at[s, 'Health State'] = 'Lyme'
                            if 'H' in name_date[0]:
                                df_metadata_ccd.at[s, 'Health State'] = 'Healthy'

                            # Get Disease State
                            if 'ED' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'EDL'
                            if 'EL' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'ELL'
                            if 'CW' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'HCW'
                            if 'CN' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'HCN'
                            if 'Hu' in name_date[0]:
                                df_metadata_ccd.at[s, 'Disease State'] = 'HCHu'

                # Strip leading and trailing whitespace
                df_metadata_ccd = df_metadata_ccd.astype(str)

                # Order by Name
                df_metadata_ccd.sort_index(inplace=True)

                # Convert DataFrame to numpy array
                metadata_ccd = np.array(df_metadata_ccd.columns).reshape(1, len(df_metadata_ccd.columns))
                metadata_ccd = np.concatenate([metadata_ccd, np.array(df_metadata_ccd)], axis=0)
            else:
                metadata_ccd = None

        # Check format of data
        if data_format == "XCMS":
            # Fill-in variable attributes
            self.varattr = df_data[['mz', 'rt', 'mzmin', 'mzmax', 'rtmin', 'rtmax', 'npeaks']]
            self.varattr.loc[:, 'Average Value(non-missing)'] = np.true_divide(data.sum(1), (data != 0).sum(1))

            # Load data points
            self.create(datapoints=data.transpose(), metadata=metadata_ccd)

        elif data_format == "Urine":
            # Set index for variable attributes
            self.varattr = pd.DataFrame(index=np.arange(len(df_data.columns[offset:])))

            # Fill in mz and rt values
            mz_rt = [list(map(float, s.strip('M').split('_'))) for s in df_data.columns[offset:]]
            self.varattr.loc[:, 'mz'] = 'nan'
            self.varattr.loc[:, 'rt'] = 'nan'
            self.varattr.loc[:, 'mz'] = np.array(mz_rt)[:, 0]
            self.varattr.loc[:, 'rt'] = np.array(mz_rt)[:, 1]

            # Fill in remaining case specific attributes
            self.varattr.loc[:, 'Average Value(non-missing)'] = np.true_divide(data.sum(0), (data != 0).sum(0))

            # Load data points
            self.create(datapoints=data, metadata=metadata_ccd)

        elif data_format == "Skyline":
            # Grab m/z values
            mz_vals = set(df_data['Product Mz'])
            mz_vals = sorted(mz_vals)

            # Initialize Feature Set
            features = []

            # Fill in Feature List
            for mz in mz_vals:
                df = df_data[df_data['Product Mz'] == mz]
                rt = df['Retention Time'].mean()
                rtmin = df['Retention Time'].min()
                rtmax = df['Retention Time'].max()
                protein = df['Protein'].iloc[0]
                peptide = df['Peptide'].iloc[0]
                features.append([mz, rt, rtmin, rtmax, protein, peptide])

            # Create variable attributes
            self.varattr = pd.DataFrame(np.array(features),
                                        columns=['mz', 'rt', 'rtmin', 'rtmax', 'Protein', 'Peptide'])

            # Initialize data values
            df_data_vals = pd.DataFrame(index=set(df_data['CSU']), columns=mz_vals)

            # Fill in data values
            for sample in df_data.index:
                csu_id = df_data.at[sample, 'CSU']
                mz = df_data.at[sample, 'Product Mz']
                df_data_vals.at[csu_id, mz] = df_data.at[sample, 'Area']

            # Sort data set by CSU ID
            df_data_vals.sort_index(inplace=True)

            data = df_data_vals.values.astype('float64')
            data = np.nan_to_num(data)

            # Fill in remaining case specific attributes
            self.varattr.loc[:, 'Average Value(non-missing)'] = np.true_divide(data.sum(0), (data != 0).sum(0))

            # Load data points
            self.create(datapoints=data, metadata=metadata_ccd)

        elif data_format == 'MassHunter':
            # Set index for variable attributes
            self.varattr = pd.DataFrame(index=np.arange(len(var_names)))
            self.varattr['Name'] = var_names

            # Fill in retention times
            self.varattr.loc[:, 'rt'] = np.nan
            self.varattr.loc[:, 'rtmin'] = np.nan
            self.varattr.loc[:, 'rtmax'] = np.nan
            i = 0
            for column in df_data.columns:
                if 'RT' in column:
                    self.varattr.at[i, 'rt'] = df_data[column].astype(float).mean()
                    self.varattr.at[i, 'rtmin'] = df_data[column].astype(float).min()
                    self.varattr.at[i, 'rtmax'] = df_data[column].astype(float).max()
                    i = i + 1

            # Fill in data values
            for column in df_data.columns:
                if not ('Area' in column):
                    df_data.drop(column, axis=1, inplace=True)

            data = df_data.values
            data = np.nan_to_num(data).astype('float64')

            # Fill in remaining case specific attributes
            self.varattr.loc[:, 'Average Value(non-missing)'] = np.true_divide(data.sum(0), (data != 0).sum(0))

            # Load data points
            self.create(datapoints=data, metadata=metadata_ccd)

        # Fill in generic variable attributes
        self.varattr.loc[:, 'Common Name'] = 'nan'
        self.varattr.loc[:, 'Classification Rate'] = 'nan'
        self.varattr.index = np.arange(len(self.varattr))
        self.varattr.loc[:, 'id'] = self.varattr.index
        # Give Common Name to known metabolites
        # self.varattr.at[self.find_feature(235.179156, 556.407), 'Common Name'] = 'Lidocaine'

        # Fill new_data
        self.new_data = np.array(self.data)

        # Fill in remaining varattrs
        self.varattr.loc[:, 'Proportion Missing'] = self.proportion_missing(np.arange(len(self.varattr)),
                                                                            axis='feature')
        self.varattr.loc[:, 'Imputed'] = 'no'

        # Make sure rt and mz are float
        convert_dict = {'rt': float, 'mz': float}

        self.varattr = self.varattr.astype(convert_dict)

    def generate_metadata(self, **kwargs):
        '''
        Generates a table of the metadata associated with all data.
        Missing values are replaced with "None".

        Inputs:
            None
        Optional inputs:
            format: string; one of 'df', 'str', 'tuple'. Modifies the output. (default: 'df')

            verbosity: level of output. (default: 0)
            idx: pointers of data to generate metadata for (default: all data)
            sortby: string, or list of strings, indicating attributes to pre-sort
                the data. (default: '_id'). Priority reads from left to right
                (sorted first by sortby[0], then sortby[1], and so on).
                Powered by a call to self.sort_by_attrvalues().
            save_to_disk: Boolean. Directly saves the generated table to h5 file.
                Overwrites anything else there. (default: False)

            In the case format=='str', arguments for the type of output are considered:
                delimiter: string; what type of delimiter to be used in the case that
                    format=='str'. (default: '\t'; i.e., tab-separated)
                newline: string; how to create a newline. (default: '\n').

        Outputs:
            A table of the metadata. Its format depends on optional inputs:
                'df': A pandas DataFrame object, whose rows correspond to
                    CCDataPoint()s and columns correspond to attributes/metadata.
                'string': A raw string which can be exported to file via:
                    output = self.generate_metadata(...)
                    f = open(fname,'w')
                    f.write(output)
                    f.close()
                'tuple': A tuple of the raw output needed to reconstruct the table;
                        output[0]: row labels (sorted idx)
                        output[1]: column labels (self.attrnames, leading with id)
                        output[2]: interior of table, as a list of lists.

                    Primary use is to populate self._metadata_table. This
                    functionality is for user to load metadata and access
                    subsets of data without loading the entire CCDataSet.
                    (We don't want pandas as a hard dependency and we
                    need a raw output for HDF anyways.)

        '''
        import numpy as np

        format = kwargs.get('format', 'df')
        verb = kwargs.get('verbosity', 0)

        sortby = kwargs.get('sortby', '_id')
        save_to_disk = kwargs.get('save_to_disk', False)

        if ('idx_list' in kwargs) and ('idx' not in kwargs):
            # just keep it for backward compatibility; don't advertise it.
            idx = kwargs.get('idx_list', np.arange(0, len(self.data), dtype=np.int64))
        else:
            idx = kwargs.get('idx', np.arange(0, len(self.data), dtype=np.int64))
        #

        if type(sortby) == str:
            sortby = [sortby]
        #
        order = self.lexsort(sortby)

        idx = np.array(idx)[order]

        # What are our delimiter and newline characters for string output?
        dl = kwargs.get('delimiter', '\t')
        nl = kwargs.get('newline', '\n')

        columns = self.attrnames
        try:
            columns.remove('_id')
        except:
            pass
        #

        # This should be gone by now... but just in case...
        try:
            columns.remove('id')
        except:
            pass
        #

        # Generate the raw table
        table = []
        for i in idx:
            d = self.data[i]
            row = [d._id] + [getattr(d, attr).value for attr in columns]
            table.append(row)
        #

        if format == 'df':
            # EZ
            import pandas
            output = pandas.DataFrame(data=table, index=idx, columns=np.append('_id', columns))
        elif format == 'str':
            # Only reason to do this ourselves is if the user doesn't want to
            # deal with pandas at all; since pandas DataFrames have their own
            # export to csv option.
            output = dl + '_id' + dl
            output += dl.join(columns) + nl

            for i, row in enumerate(table):
                strrow = str(idx[i])
                for j, e in enumerate(row):
                    try:
                        strrow += dl + str(e)
                    except:
                        raise TypeError(
                            'ERROR: Unable to cast attribute %s for datapoint %s to string.' % (columns[j], d._id))
                    #
                #
                strrow += nl
                output += strrow
            #
        elif format == 'tuple':
            # User can call this function with extra arguments if they prefer
            # a particular ordering of the data. Otherwise save_CCDataSet calls
            # the function with this flag just to populate the thing.
            output = (idx, ['_id'] + list(columns), table)
            self._metadata_table = output
        #

        if save_to_disk:
            # TODO - MOVE THIS OVER TO loadsave_hdf.py AS ITS OWN FUNCTION
            import h5py
            h5f = h5py.File(self.fname)
            h5f_ccd = h5f['CCDataSet']
            if 'metadata_table' not in h5f_ccd.keys():
                h5_metadata_table = h5f_ccd.create_group('metadata_table')
                h5_metadata_table.create_dataset('rows', data=idx, compression="gzip")
                h5_metadata_table.create_dataset('columns', data=np.array(['_id'] + columns, dtype=np.string_),
                                                 compression="gzip")
                h5_metadata_table.create_dataset('metadata', data=np.array(table, dtype=np.string_), compression="gzip")
            else:
                # BUG: CANNOT OVERWRITE OLD TABLE. DO NOT KNOW THE FUNCTIONALITY;
                # CAN'T BE BOTHERED TO LOOK IT UP RIGHT NOW. TOO LATE AT NIGHT.
                h5_metadata_table = h5f_ccd['metadata_table']
                h5_metadata_table['rows'] = idx
                h5_metadata_table['columns'] = np.array(['_id'] + columns, dtype=np.string_)
                h5_metadata_table['metadata'] = np.array(table, dtype=np.string_)
            #

            h5f.close()
        #

        return output

    def plot(self, identifier=None, style='contour', show_missing=False, s=2):
        '''
       Plots a CCDataPoint intensity within a CCLymeMetaboliteSet instance with respect to (mz, rt) axes

       Input:
            identifier: integer. Indicates the position of the CCDataPoint in the CCList.
            style: string. '3d' for 3-dimensional plot, 'contour' for 2-dimensional contour plot

            s: integer. Size of markers in plot.

            show_missing: bool. Indicates whether or not to plot missing data along side non-missing data

       Output:
           Labelled 3D or 2D (mz, rt) intensity plot of a particular CCDataPoint
       '''

        if identifier is None:
            raise TypeError("No identifier given for CCDataPoint")

        if (style == '3d') and show_missing:
            raise ValueError("plot() cannot produce multiple 3d subplots. Use style='contour'")

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from numpy import errstate, isneginf
        from datetime import datetime

        if identifier < 0:
            raise TypeError("Unrecognized type: Please input non-negative integer.")

        z = np.array(self.new_data[identifier])

        # Take log of intensity to visualize dynamic range
        with errstate(divide='ignore'):
            z = np.log(z)
        z[isneginf(z)] = 0

        # Get name, date of sample
        name = self.data[identifier].Name.value
        date = self.data[identifier].Date.value
        date = datetime.strptime(date, '%m%d%y').strftime('%m/%d/%y')

        x = np.array(self.varattr['mz'])
        y = np.array(self.varattr['rt'])

        # Plot
        c = z
        cm = plt.get_cmap("jet")

        if not show_missing:
            fig = plt.figure(figsize=(10, 7))

            if style == '3d':
                ax = fig.add_subplot(111, projection='3d')
                cax = ax.scatter(xs=x, ys=y, zs=z, s=10, c=c, cmap=cm)
                ax.set_zlabel('Log(Intensity)')
            if style == 'contour':
                ax = fig.add_subplot(111)
                cax = ax.scatter(x, y, s=s, linewidths=0, c=c, cmap=cm)

            ax.set_title('Log intensity plot of m/z vs renention time\n for Lyme sample ' + name + ' on ' + date +
                         '\nData Set: ' + self.name)
            ax.set_xlabel('m/z')
            ax.set_ylabel('Retention time (sec)')
            cbar = fig.colorbar(cax)
            cbar.set_label('Log(Intensity)', rotation=270, labelpad=13)
            plt.show()

        else:

            fig = plt.figure(figsize=(15, 10))

            # Locate missing data
            missing = (z == 0)
            not_missing = np.logical_not(missing)

            # Plot non-missing
            ax1 = fig.add_subplot(1, 2, 1)
            cax1 = ax1.scatter(x[not_missing], y[not_missing], s=s, linewidths=0, c=c[not_missing], cmap=cm)
            ax1.set_title('Log intensity plot of m/z vs renention time\n for Lyme sample '
                          + name + ' on ' + date + ' (non-missing)' + '\nData Set: ' + self.name)
            ax1.set_xlabel('m/z')
            ax1.set_ylabel('Retention time (sec)')
            cbar1 = fig.colorbar(cax1)
            cbar1.set_label('Log(Intensity)', rotation=270, labelpad=13)

            # Plot missing
            ax2 = fig.add_subplot(1, 2, 2)
            cax2 = ax2.scatter(x[missing], y[missing], s=s, linewidths=0, c='k')
            ax2.set_title('Log intensity plot of m/z vs renention time\n for Lyme sample '
                          + name + ' on ' + date + ' (missing)' + '\nData Set: ' + self.name)
            ax2.set_xlabel('m/z')
            ax2.set_ylabel('Retention time (sec)')

            plt.show()
        return

    def proportion_missing(self, identifier=None, axis='feature'):
        '''
          Calculates the proportion of missing data for a given CCDataPoint or feature, if no identifier is given then
          this function calculates the average of missing data over CCDataPoints in the CCLymeMetaboliteSet. This is equivalent to
          calculating the proportion of missing data in the entire set.

          Input:
              identifier: list of integers. integer value indicates the position of the CCDataPoint in the CCList or feature
                          number in the feature list.

              axis: string. string value can either be sample or feature.

          Output:
              float. Proportion of missing data
          '''

        proportion = 0

        data = deepcopy(self.new_data)
        m, n = data.shape

        if identifier is None:
            return np.count_nonzero(data == 0) / (m * n)

        if axis == 'sample':
            return np.count_nonzero(data[identifier, :] == 0, axis=1) / n
        elif axis == 'feature':
            return np.count_nonzero(data[:, identifier] == 0, axis=0) / m

    def locate_missing(self, threshold=0, axis='feature', attrname=None):
        '''
          Locates features or samples by id in the CCLymeMetaboliteSet which have missing data above
          a given threshold. The user can specify an attribute name which indicates that the data must
          be missing from each equivalence class under that attribute above the threshold

          Input:
              threshold: 0<float<1. Look for features or sample which have missing data above this threshold
              axis: string. Indicates to look for missingness in features or samples
              attrname: string. Indicates which attribute to bin by for a specific feature.

          Output:
              list. list of ids of features or samples above threshold
          '''

        if (threshold < 0) or (threshold > 1):
            raise ValueError("Invalid threshold. Please input a threshold value between 0 and 1")

        data = deepcopy(self.new_data)
        m, n = data.shape

        if attrname is None:
            if axis == 'feature':
                loc = 0
                threshold = threshold * m
            elif axis == 'sample':
                loc = 1
                threshold = threshold * n

            num_zeros = np.count_nonzero(data == 0, axis=loc)

            return np.where(num_zeros > threshold)[0]

        else:
            if axis == 'sample':
                raise ValueError("attrname must be used with features and not samples")
            else:
                attr_classes = self.partition(attrname)
                index = np.arange(n)
                for equiv_class in attr_classes.values():
                    data_i = data[equiv_class, :]
                    [m_i, n_i] = data_i.shape
                    threshold_i = threshold * m_i
                    num_zeros = np.count_nonzero(data_i == 0, axis=0)
                    index = np.intersect1d(index, np.where(num_zeros > threshold_i)[0])
            return index

    def remove_feature(self, identifier=None):
        '''
          Removes feature from the CCLymeMetaboliteSet. Stores list of removed features into self.removed_features

          Input:
              identifier: integer or list of integers. index(s) of feature(s) to be removed

          Output:
              None. Updates self with removed features
          '''

        if identifier is None:
            raise ValueError('Please input a list of features to be removed.')
        elif type(identifier) == int:
            identifier = [identifier]

        # Remove Data and then store it elsewhere
        self.removed_features = pd.concat([self.removed_features, self.varattr.loc[identifier]])
        self.removed_features.index = np.arange(len(self.removed_features))
        self.varattr.drop(identifier, inplace=True)
        self.varattr.index = np.arange(len(self.varattr))

        self.new_data = np.delete(arr=self.new_data, obj=identifier, axis=1)

        # for j in identifier:
        # self.variable_names.remove(self.variable_names[j])

        return

    def remove_missing(self, threshold=0.8, attrname=None):
        '''
          Removes features from the CCLymeMetaboliteSet which have missing data above
          a given threshold with respects to an attribute. i.e. missing above a threshold in every
          equivalence class of the attribute

          Input:
              threshold: 0<float<1. Look for features or samples which have missing data above this threshold
              attrname: string. Indicates which attribute to bin by for a specific feature.

          Output:
              None. Updates self with removed features
          '''

        # Locate data above missing threshold
        location = self.locate_missing(threshold, 'feature', attrname)

        # Remove Data and then store it elsewhere
        self.remove_feature(identifier=location)

        return

    def plot_missing(self, threshold=0.8, attrname='Date'):
        '''
          Plots a contour plot of features in the CCLymeMetaboliteSet which have missing data above
          a given threshold with respects to an attribute. i.e. missing above a threshold in every
          equivalence class of the attribute. Axes are x:'m/z', y:'rt', z:'average value over non-missing'

          Input:
              threshold: 0<float<1. Look for features or samples which have missing data above this threshold
              attrname: string. Indicates which attribute to bin by for a specific feature.

          Output:
              Plot.
          '''

        import matplotlib.pyplot as plt
        from numpy import errstate, isneginf

        # Find missing features
        location = self.locate_missing(threshold=threshold, attrname=attrname)

        # Fill x,y,z axes
        x = np.array(self.varattr.loc[location, 'mz'])
        y = np.array(self.varattr.loc[location, 'rt'])
        z = np.array(self.varattr.loc[location, 'Average Value(non-missing)'])

        # Take log of intensity to visualize dynamic range
        with errstate(divide='ignore'):
            z = np.log(z)
        z[isneginf(z)] = 0

        # Plot
        c = z
        cm = plt.get_cmap("jet")
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        cax = ax.scatter(x, y, s=10, c=c, cmap=cm)
        ax.set_title('Log intensity plot of m/z vs renention time for features\n missing in greater than '
                     + str(threshold * 100) + '% of samples across ' + attrname + ' groups \nData Set: ' + self.name)

        ax.set_xlabel('m/z')
        ax.set_ylabel('Retention time (min)')
        cbar = fig.colorbar(cax)
        cbar.set_label('Log(Average Intensity) over samples with non-missing feature', rotation=270, labelpad=13)
        plt.show()

    def plot_missing_vs_threshold(self, threshold_step, axis='sample'):
        '''
          Plots the proportion of missing data in features or samples with respect to the threshold.i.e.
          Plots the function f(threshold) = proportion of features/samples with missing data > threshold

          Input:
              threshold_step: float. Step size for threshold-axis
              axis: string. Indicates to plot with respect to features or samples

          Output:
              plot
          '''

        m, n = np.shape(self.data)

        if axis == 'sample':
            total = m
        elif axis == 'feature':
            total = n

        threshold_axis = np.arange(0, 1, threshold_step)

        # Calculate proportions of missing data
        proportions = [self.locate_missing(threshold=t, axis=axis).size / total for t in threshold_axis]

        # Plot
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 7))
        plot = plt.plot(threshold_axis, proportions, c='k')
        plt.title(
            'Proportion of ' + axis.capitalize() + 's with Proportion of Missing Data Strictly above a given'
                                                   ' Threshold\nData Set: ' + self.name)
        plt.xlabel('Threshold for Proportion of Missing Data')
        plt.ylabel('Proportion of ' + axis.capitalize() + 's')
        plt.xticks(np.arange(0, 1 + 5 * threshold_step, 5 * threshold_step))
        plt.yticks(np.arange(0, 1 + 5 * threshold_step, 5 * threshold_step))
        plt.show()

    def find_feature(self, mz=None, rt=None, ppm_tol=30, rt_tol=20, attrname=None, attrvalue=None):
        '''
          This function finds a features index in self.varattr based on the given mz and rt value by default. Also finds
          set of features by varattr name

          Input:
              mz: float. m/z value
              rt: float. rentention time
              attrname: string. varattr name
              ppm_tol: integer. ppm tolerance for m/z window
              rt_tol: integer. Retention time tolerance for retention time window
              attrvalue: variable-type. value of varattr name

          Output: int. Index of feature in self.varattr
          '''

        if (mz is None) and (rt is None):
            if (attrname is None) or (attrvalue is None):
                raise ValueError('Need to input both attribute name and value.')
            else:
                return list(self.varattr.loc[self.varattr[attrname] == attrvalue].index)

        elif not (mz is None) and not (rt is None):
            # Convert pandas attrs to numpy attrs
            ppm_array = (np.abs(np.array(self.varattr['mz']) - mz) / np.array(self.varattr['mz'])) * (10 ** 6)
            rt_array = np.abs(np.array(self.varattr['rt']) - rt)

            # Find feature
            mz_loc = ppm_array <= ppm_tol
            rt_loc = rt_array <= rt_tol
            feature_loc = np.logical_and(mz_loc, rt_loc)

            return np.nonzero(feature_loc)[0]

            return closest_rt

        elif rt is None:
            # Convert pandas attrs to numpy attrs
            ppm_array = (np.abs(np.array(self.varattr['mz']) - mz) / np.array(self.varattr['mz'])) * (10 ** 6)

            # Find feature
            mz_loc = ppm_array <= ppm_tol

            return np.nonzero(mz_loc)[0]

        else:
            raise ValueError('Must input either (mz,rt) pair or (attrname, attrvalue) pair')

    def skew_features(self, skewness=1, attrname='Date'):
        '''
          Locates features in the CCLymeMetaboliteSet which have skewness above
          a given threshold.

          Input:
              skewness: 0<float<1. Look for features which have skewness above this threshold
              attrname: string. Indicates which attribute to bin by for a specific feature.

          Output:
              list. list of ids of features above threshold
          '''

        # Imports
        import scipy.stats as stats

        # Gather data
        data = deepcopy(self.new_data)
        n = data.shape[1]

        # Define classes
        attr_classes = self.partition(attrname)
        index = np.array([])

        # Calculate skewness
        for equiv_class in attr_classes.values():
            data_i = data[equiv_class, :]
            skew_values = stats.skew(data_i, axis=0, nan_policy='omit')
            index = np.union1d(index, np.where(skew_values >= skewness)[0]).astype(int)

        return index

    def feature_histogram(self, identifier=None, attrname='Health State', bins=20, nanpolicy='keep'):
        '''
          Plot histograms of a feature intensity with respect to samples in the CCLymeMetaboliteSet
          across equivalence classes of an atrribute

          Input:
              attrname: string. Indicates which attribute to bin by for a specific feature.
              bins: Number of bins for histogram
          Output:
              plot.
          '''

        if identifier is None:
            raise ValueError('Must input feature number!')

        # Get feature name
        feature_name = '(' + str(self.varattr.loc[identifier, 'mz']) + ',' + str(
            self.varattr.loc[identifier, 'rt']) + ')'

        # Imports
        import matplotlib.pyplot as plt

        # Load Data
        data = deepcopy(self.new_data)

        if nanpolicy == 'omit':
            data[data == 0] = np.nan

        # Create DataFrame
        attr_classes = self.partition(attrname)
        df = pd.DataFrame()

        for i in range(len(attr_classes)):
            key = list(attr_classes.keys())[i].upper()
            value = data[list(attr_classes.values())[i], identifier]
            df_i = pd.DataFrame({key: value})
            df = pd.concat([df, df_i], axis=1)

        fig, ax = plt.subplots(figsize=(10, 7))
        df.plot.hist(density=False, ax=ax, bins=bins, alpha=.5, legend=True,
                     title='Histogram of feature (mz,rt)=' + feature_name + ' by ' + attrname +
                           '\nCommon Name: ' + self.varattr.loc[identifier, 'Common Name'] + '\nData Set: ' + self.name)
        ax.set_ylabel('# of Samples', fontsize=14)
        ax.grid(axis='y')
        ax.set_xlabel('Feature Intensity', fontsize=14)
        ax.set_facecolor('#d8dcd6')

    def impute(self, algorithm='auto', **kwargs):
        '''
                 Imputes data according to chosen algorithm across chosen features

                  Input:
                      algorithm: string. Imputation algorithm name. e.g. 'knn', 'missforest', 'median', 'mean', 'half-minimum'
                      kwargs: Arguments specific to choice of algorithm

                  Output: In-place method
        '''

        # List supported algorithms
        algorithms = ['knn', 'missforest', 'median', 'mean', 'half-minimum']

        if not (algorithm in algorithms):
            raise ValueError(
                'Algorithm mispelled or not supported.\n '
                'Please input either: knn, missforest, median, mean, or half-minimum')

        # Find all missing data
        loc = self.locate_missing()
        if loc.size == 0:
            print('No data is missing!')
            return

        # Grab data
        data = self.new_data

        # Locate missing and label imputed features
        loc = self.locate_missing()

        if algorithm == 'knn':
            from sklearn.impute import KNNImputer
            '''
                missingpy KNN imputer doc
                
                Parameters
                ----------
                missing_values : integer or "NaN", optional (default = "NaN")
                    The placeholder for the missing values. All occurrences of
                    `missing_values` will be imputed. For missing values encoded as
                    ``np.nan``, use the string value "NaN".
                
                n_neighbors : int, optional (default = 5)
                    Number of neighboring samples to use for imputation.
                
                weights : str or callable, optional (default = "uniform")
                    Weight function used in prediction.  Possible values:
                
                    - 'uniform' : uniform weights.  All points in each neighborhood
                      are weighted equally.
                    - 'distance' : weight points by the inverse of their distance.
                      in this case, closer neighbors of a query point will have a
                      greater influence than neighbors which are further away.
                    - [callable] : a user-defined function which accepts an
                      array of distances, and returns an array of the same shape
                      containing the weights.
                
                metric : str or callable, optional (default = "masked_euclidean")
                    Distance metric for searching neighbors. Possible values:
                    - 'masked_euclidean'
                    - [callable] : a user-defined function which conforms to the
                    definition of _pairwise_callable(X, Y, metric, **kwds). In other
                    words, the function accepts two arrays, X and Y, and a
                    ``missing_values`` keyword in **kwds and returns a scalar distance
                    value.
                
                row_max_missing : float, optional (default = 0.5)
                    The maximum fraction of columns (i.e. features) that can be missing
                    before the sample is excluded from nearest neighbor imputation. It
                    means that such rows will not be considered a potential donor in
                    ``fit()``, and in ``transform()`` their missing feature values will be
                    imputed to be the column mean for the entire dataset.
                
                col_max_missing : float, optional (default = 0.8)
                    The maximum fraction of rows (or samples) that can be missing
                    for any feature beyond which an error is raised.
                
                copy : boolean, optional (default = True)
                    If True, a copy of X will be created. If False, imputation will
                    be done in-place whenever possible. Note that, if metric is
                    "masked_euclidean" and copy=False then missing_values in the
                    input matrix X will be overwritten with zeros.
                
                Attributes
                ----------
                statistics_ : 1-D array of length {n_features}
                    The 1-D array contains the mean of each feature calculated using
                    observed (i.e. non-missing) values. This is used for imputing
                    missing values in samples that are either excluded from nearest
                    neighbors search because they have too many ( > row_max_missing)
                    missing features or because all of the sample's k-nearest neighbors
                    (i.e., the potential donors) also have the relevant feature value
                    missing.
                
                Methods
                -------
                fit(X, y=None):
                    Fit the imputer on X.
                
                    Parameters
                    ----------
                    X : {array-like}, shape (n_samples, n_features)
                        Input data, where ``n_samples`` is the number of samples and
                        ``n_features`` is the number of features.
                
                    Returns
                    -------
                    self : object
                        Returns self.
                
                
                transform(X):
                    Impute all missing values in X.
                
                    Parameters
                    ----------
                    X : {array-like}, shape = [n_samples, n_features]
                        The input data to complete.
                
                    Returns
                    -------
                    X : {array-like}, shape = [n_samples, n_features]
                        The imputed dataset.
                
                
                fit_transform(X, y=None, **fit_params):
                    Fit KNNImputer and impute all missing values in X.
                
                    Parameters
                    ----------
                    X : {array-like}, shape (n_samples, n_features)
                        Input data, where ``n_samples`` is the number of samples and
                        ``n_features`` is the number of features.
                
                    Returns
                    -------
                    X : {array-like}, shape (n_samples, n_features)
                        Returns imputed dataset.'''

            # Label imputed features
            self.varattr.loc[loc, 'Imputed'] = algorithm

            # Grab arguments
            missing_values = kwargs.get('missing_values', np.nan)
            n_neighbors = kwargs.get('n_neighbors', 5)
            weights = kwargs.get('weights', "uniform")
            metric = kwargs.get('metric', "nan_euclidean")
            copy = kwargs.get('copy', True)

            # Initialize Imputer
            imputer = KNNImputer(missing_values=missing_values, n_neighbors=n_neighbors, weights=weights,
                                 metric=metric, copy=copy)
            # Impute data
            data[data == 0] = np.nan
            data = imputer.fit_transform(data)

            # Revert back to zeros
            data[np.isnan(data)] = 0

        elif algorithm == 'missforest':
            '''Parameters
                ----------
                NOTE: Most parameter definitions below are taken verbatim from the
                Scikit-Learn documentation at [2] and [3].
                
                max_iter : int, optional (default = 10)
                    The maximum iterations of the imputation process. Each column with a
                    missing value is imputed exactly once in a given iteration.
                
                decreasing : boolean, optional (default = False)
                    If set to True, columns are sorted according to decreasing number of
                    missing values. In other words, imputation will move from imputing
                    columns with the largest number of missing values to columns with
                    fewest number of missing values.
                
                missing_values : np.nan, integer, optional (default = np.nan)
                    The placeholder for the missing values. All occurrences of
                    `missing_values` will be imputed.
                
                copy : boolean, optional (default = True)
                    If True, a copy of X will be created. If False, imputation will
                    be done in-place whenever possible.
                
                criterion : tuple, optional (default = ('mse', 'gini'))
                    The function to measure the quality of a split.The first element of
                    the tuple is for the Random Forest Regressor (for imputing numerical
                    variables) while the second element is for the Random Forest
                    Classifier (for imputing categorical variables).
                
                n_estimators : integer, optional (default=100)
                    The number of trees in the forest.
                
                max_depth : integer or None, optional (default=None)
                    The maximum depth of the tree. If None, then nodes are expanded until
                    all leaves are pure or until all leaves contain less than
                    min_samples_split samples.
                
                min_samples_split : int, float, optional (default=2)
                    The minimum number of samples required to split an internal node:
                    - If int, then consider `min_samples_split` as the minimum number.
                    - If float, then `min_samples_split` is a fraction and
                      `ceil(min_samples_split * n_samples)` are the minimum
                      number of samples for each split.
                
                min_samples_leaf : int, float, optional (default=1)
                    The minimum number of samples required to be at a leaf node.
                    A split point at any depth will only be considered if it leaves at
                    least ``min_samples_leaf`` training samples in each of the left and
                    right branches.  This may have the effect of smoothing the model,
                    especially in regression.
                    - If int, then consider `min_samples_leaf` as the minimum number.
                    - If float, then `min_samples_leaf` is a fraction and
                      `ceil(min_samples_leaf * n_samples)` are the minimum
                      number of samples for each node.
                
                min_weight_fraction_leaf : float, optional (default=0.)
                    The minimum weighted fraction of the sum total of weights (of all
                    the input samples) required to be at a leaf node. Samples have
                    equal weight when sample_weight is not provided.
                
                max_features : int, float, string or None, optional (default="auto")
                    The number of features to consider when looking for the best split:
                    - If int, then consider `max_features` features at each split.
                    - If float, then `max_features` is a fraction and
                      `int(max_features * n_features)` features are considered at each
                      split.
                    - If "auto", then `max_features=sqrt(n_features)`.
                    - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
                    - If "log2", then `max_features=log2(n_features)`.
                    - If None, then `max_features=n_features`.
                    Note: the search for a split does not stop until at least one
                    valid partition of the node samples is found, even if it requires to
                    effectively inspect more than ``max_features`` features.
                
                max_leaf_nodes : int or None, optional (default=None)
                    Grow trees with ``max_leaf_nodes`` in best-first fashion.
                    Best nodes are defined as relative reduction in impurity.
                    If None then unlimited number of leaf nodes.
                
                min_impurity_decrease : float, optional (default=0.)
                    A node will be split if this split induces a decrease of the impurity
                    greater than or equal to this value.
                    The weighted impurity decrease equation is the following::
                        N_t / N * (impurity - N_t_R / N_t * right_impurity
                                            - N_t_L / N_t * left_impurity)
                    where ``N`` is the total number of samples, ``N_t`` is the number of
                    samples at the current node, ``N_t_L`` is the number of samples in the
                    left child, and ``N_t_R`` is the number of samples in the right child.
                    ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
                    if ``sample_weight`` is passed.
                
                bootstrap : boolean, optional (default=True)
                    Whether bootstrap samples are used when building trees.
                
                oob_score : bool (default=False)
                    Whether to use out-of-bag samples to estimate
                    the generalization accuracy.
                
                n_jobs : int or None, optional (default=None)
                    The number of jobs to run in parallel for both `fit` and `predict`.
                    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
                    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
                    for more details.
                
                random_state : int, RandomState instance or None, optional (default=None)
                    If int, random_state is the seed used by the random number generator;
                    If RandomState instance, random_state is the random number generator;
                    If None, the random number generator is the RandomState instance used
                    by `np.random`.
                
                verbose : int, optional (default=0)
                    Controls the verbosity when fitting and predicting.
                
                warm_start : bool, optional (default=False)
                    When set to ``True``, reuse the solution of the previous call to fit
                    and add more estimators to the ensemble, otherwise, just fit a whole
                    new forest. See :term:`the Glossary <warm_start>`.
                
                class_weight : dict, list of dicts, "balanced", "balanced_subsample" or \
                None, optional (default=None)
                    Weights associated with classes in the form ``{class_label: weight}``.
                    If not given, all classes are supposed to have weight one. For
                    multi-output problems, a list of dicts can be provided in the same
                    order as the columns of y.
                    Note that for multioutput (including multilabel) weights should be
                    defined for each class of every column in its own dict. For example,
                    for four-class multilabel classification weights should be
                    [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
                    [{1:1}, {2:5}, {3:1}, {4:1}].
                    The "balanced" mode uses the values of y to automatically adjust
                    weights inversely proportional to class frequencies in the input data
                    as ``n_samples / (n_classes * np.bincount(y))``
                    The "balanced_subsample" mode is the same as "balanced" except that
                    weights are computed based on the bootstrap sample for every tree
                    grown.
                    For multi-output, the weights of each column of y will be multiplied.
                    Note that these weights will be multiplied with sample_weight (passed
                    through the fit method) if sample_weight is specified.
                    NOTE: This parameter is only applicable for Random Forest Classifier
                    objects (i.e., for categorical variables).
                
                Attributes
                ----------
                statistics_ : Dictionary of length two
                    The first element is an array with the mean of each numerical feature
                    being imputed while the second element is an array of modes of
                    categorical features being imputed (if available, otherwise it
                    will be None).
                
                Methods
                -------
                fit(self, X, y=None, cat_vars=None):
                    Fit the imputer on X.
                
                    Parameters
                    ----------
                    X : {array-like}, shape (n_samples, n_features)
                        Input data, where ``n_samples`` is the number of samples and
                        ``n_features`` is the number of features.
                
                    cat_vars : int or array of ints, optional (default = None)
                        An int or an array containing column indices of categorical
                        variable(s)/feature(s) present in the dataset X.
                        ``None`` if there are no categorical variables in the dataset.
                
                    Returns
                    -------
                    self : object
                        Returns self.
                
                
                transform(X):
                    Impute all missing values in X.
                
                    Parameters
                    ----------
                    X : {array-like}, shape = [n_samples, n_features]
                        The input data to complete.
                
                    Returns
                    -------
                    X : {array-like}, shape = [n_samples, n_features]
                        The imputed dataset.
                
                
                fit_transform(X, y=None, **fit_params):
                    Fit MissForest and impute all missing values in X.
                
                    Parameters
                    ----------
                    X : {array-like}, shape (n_samples, n_features)
                        Input data, where ``n_samples`` is the number of samples and
                        ``n_features`` is the number of features.
                
                    Returns
                    -------
                    X : {array-like}, shape (n_samples, n_features)
                        Returns imputed dataset.'''

            # Label imputed features
            self.varattr.loc[loc, 'Imputed'] = algorithm

            # Grab arguments
            max_iter = kwargs.get('max_iter', 10)
            decreasing = kwargs.get('decreasing', False)
            missing_values = kwargs.get('missing_values', 0)
            copy = kwargs.get('copy', True)
            n_estimators = kwargs.get('n_estimators', 100)
            criterion = kwargs.get('criterion', ('mse', 'gini'))
            max_depth = kwargs.get('max_depth', None)
            min_samples_split = kwargs.get('min_samples_split', 2)
            min_samples_leaf = kwargs.get('min_samples_leaf', 1)
            min_weight_fraction_leaf = kwargs.get('min_weight_fraction_leaf', 0.0)
            max_features = kwargs.get('max_features', 'auto')
            max_leaf_nodes = kwargs.get('max_leaf_nodes', None)
            min_impurity_decrease = kwargs.get('mmin_impurity_decrease', 0.0)
            bootstrap = kwargs.get('bootstrap', True)
            oob_score = kwargs.get('oob_score', False)
            n_jobs = kwargs.get('n_jobs', -1)
            random_state = kwargs.get('random_state', None)
            verbose = kwargs.get('verbose', 0)
            warm_start = kwargs.get('warm_start', False)
            class_weight = kwargs.get('class_weight', None)

            # Initialize Imputer
            imputer = missingpy.MissForest(max_iter=max_iter, decreasing=decreasing, missing_values=missing_values,
                                           copy=copy, n_estimators=n_estimators, criterion=criterion,
                                           max_depth=max_depth, min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf,
                                           min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                           max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                           bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs,
                                           random_state=random_state,
                                           verbose=verbose, warm_start=warm_start, class_weight=class_weight)
            # Impute Data
            data = imputer.fit_transform(data)

            # Revert back to zeros
            data[np.isnan(data)] = 0

        elif algorithm == 'median':

            # Get row dim of data
            m = data.shape[0]

            # Covert to nan
            data[data == 0] = np.nan

            # Feature independent LOD
            LOD = kwargs.get('LOD', None)

            if LOD is None:
                # Calculate feature dependent LOD based on skewness
                attrname = kwargs.get('attrname', 'Disease State')
                skewness = kwargs.get('skewness', -1000)
                index = np.intersect1d(loc,
                                       self.skew_features(skewness=skewness, attrname=attrname)).astype(int)
            else:
                index = np.where(np.nanmin(data, axis=0) < LOD)[0]

            # Label features with Imputed type
            self.varattr.loc[index, 'Imputed'] = algorithm

            # Convert index to bool type
            index_bool = np.full(data.shape, False)
            index_bool[:, index] = True
            index = np.logical_and(np.isnan(data), index_bool)

            # Compute median
            median = np.nanmedian(data, axis=0)
            median = np.tile(median, (m, 1))

            # Impute data
            data[index] = median[index]

            # Revert back to zeros
            data[np.isnan(data)] = 0

        elif algorithm == 'mean':

            # Get row dim of data
            m = data.shape[0]

            # Covert to nan
            data[data == 0] = np.nan

            # Feature independent LOD
            LOD = kwargs.get('LOD', None)

            if LOD is None:
                # Calculate feature dependent LOD based on skewness
                attrname = kwargs.get('attrname', 'Disease State')
                skewness = kwargs.get('skewness', -1000)
                index = np.intersect1d(loc,
                                       self.skew_features(skewness=skewness, attrname=attrname)).astype(int)
            else:
                index = np.where(np.nanmin(data, axis=0) < LOD)[0]

            # Label features with Imputed type
            self.varattr.loc[index, 'Imputed'] = algorithm

            # Convert index to bool type
            index_bool = np.full(data.shape, False)
            index_bool[:, index] = True
            index = np.logical_and(np.isnan(data), index_bool)

            # Compute mean
            mean = np.nanmean(data, axis=0)
            mean = np.tile(mean, (m, 1))

            # Impute data
            data[index] = mean[index]

            # Revert back to zeros
            data[np.isnan(data)] = 0

        elif algorithm == 'half-minimum':

            # Get row dim of data
            m = data.shape[0]

            # Covert to nan
            data[data == 0] = np.nan

            # Feature independent LOD
            LOD = kwargs.get('LOD', None)

            if LOD is None:
                # Calculate feature dependent LOD based on skewness
                attrname = kwargs.get('attrname', 'Disease State')
                skewness = kwargs.get('skewness', -1000)
                index = np.intersect1d(loc,
                                       self.skew_features(skewness=skewness, attrname=attrname)).astype(int)
            else:
                index = np.where(np.nanmin(data, axis=0) < LOD)[0]

            # Label features with Imputed type
            self.varattr.loc[index, 'Imputed'] = algorithm

            # Convert index to bool type
            index_bool = np.full(data.shape, False)
            index_bool[:, index] = True
            index = np.logical_and(np.isnan(data), index_bool)

            # Compute half-minimum
            hm = .5 * np.nanmin(data, axis=0)
            hm = np.tile(hm, (m, 1))

            # Impute data
            data[index] = hm[index]

            # Revert back to zeros
            data[np.isnan(data)] = 0

        # Replace data in LymeMetaboliteSet with imputed data
        self.new_data = data

        # Update Imputation method
        if self.imputation_method is None:
            self.imputation_method = algorithm
        else:
            self.imputation_method = self.imputation_method + '_' + algorithm

        return

    def normalize(self, algorithm='log', controls=[], idx_list=None, feature_set=None, **kwargs):
        '''
                 Imputes data according to chosen algorithm across chosen features

                  Input:
                      algorithm: string. Imputation algorithm name. e.g. 'standard', 'log',
                      'quantile', 'median-polish', 'cyclic lowess', 'median-fold change',
                      'osmolality'

                      kwargs: Arguments specific to choice of algorithm

                  Output: In-place method
        '''

        # List supported algorithms
        algorithms = ['standard', 'log', 'quantile', 'median-polish', 'cyclic lowess', 'median-fold change',
                      'osmolality']

        if not (algorithm in algorithms):
            raise ValueError(
                'Algorithm mispelled or not supported.\n '
                'Please input either: standard', 'log', 'quantile', 'median-polish', 'cyclic lowess',
                'median-fold change', 'osmolality')

        # Grab data
        data = deepcopy(self.new_data)

        # Restrict to idx_list and feature_set
        if idx_list is None:
            idx_list = np.arange(data.shape[0])
        if feature_set is None:
            feature_set = np.arange(data.shape[1])

        idx_list = np.array(idx_list)
        feature_set = np.array(feature_set)
        data = data[idx_list, :]
        data = data[:, feature_set]

        if algorithm == 'log':
            # Take log of intensity to visualize dynamic range
            with np.errstate(divide='ignore'):
                data = np.log(data)
            data[np.isneginf(data)] = 0

        if algorithm == 'quantile':
            m, n = data.shape
            data_copy = deepcopy(np.sort(data))
            data_sort = np.argsort(deepcopy(data))
            mean = np.mean(data_copy, axis=0)
            for i in range(m):
                data[i, data_sort[i, :]] = mean

        if algorithm == 'median-polish':
            median_polish = MedianPolish(data)
            median_polish.median_polish(method='median')
            data = median_polish.tbl

        if algorithm == 'standard':
            # Find missing data
            missing = (data == 0)
            data[missing] = np.nan

            # Compute means and stds
            mean = np.nanmean(data, axis=0)
            mean = mean.reshape(1, len(mean))
            std = np.nanstd(data, axis=0)
            std = std.reshape(1, len(std))

            # Shift by mean and scale by std
            data = data - mean
            data = data * (1 / std)
            data[missing] = 0

        if algorithm == 'median-fold change':
            # Find missing data and change zero to nan
            missing = (data == 0)
            data[missing] = np.nan

            # Integral normalize
            integral = np.nansum(data, axis=1)
            integral = integral.reshape(-1, 1)
            data = 100 * (data / integral)

            # Calculate medians of controls for "Golden Reference"
            median = np.nanmedian(data[controls, :], axis=0)

            # Calculate fold changes from median
            quotients = data / median

            # Calculate median of fold changes
            median_quotients = np.nanmedian(quotients, axis=1)
            median_quotients = median_quotients.reshape(-1, 1)

            # Scale each sample by median of the fold changes
            data = data / median_quotients
            data[missing] = 0

        if algorithm == 'osmolality':
            df = self.generate_metadata()
            df.sort_values(by='CSU_ID', inplace=True)
            for i, sample in enumerate(df.index):
                osmo = df.loc[sample, ['mOSM/ kg H2O #1', 'mOSM/ kg H2O #2', 'mOSM/ kg H2O #3']].astype(float).mean()
                if not (osmo is np.nan):
                    data[i, :] = data[i, :] / osmo

        # Replace data in LymeMetaboliteSet with normalized data
        self.new_data[idx_list[:, np.newaxis], feature_set] = data

        # Update Imputation method
        if self.normalization_method is None:
            self.normalization_method = algorithm
        else:
            self.normalization_method = self.normalization_method + '_' + algorithm

        return

    def visualize(self, method='umap', dimension=2, attrname='Health State', features=None, idx_list=None, save=False,
                  **kwargs):
        '''
        Visualize samples by attribute name in 2D or 3D using a particular visualiztion method

        Input:
            method: string. Name of method to visualize the data.
                e.g.    'umap': 'UMAP', 'pca': 'Principal Component Analysis',
                        'isomap': 'Isometric Mapping',
                        'lle': 'Local Linear Embedding',
                        'laplacian': 'Laplacian Eigenmaps',
                        'mds': 'Multidimensional Scaling', 'none': 'Generic Projection'

            dimension: 2 or 3. Dimension of space to embed data into.

            attrname: string. Attribute to visualize equivalence classes on.

            features: list of integers. Features to restrict data to.

            idx_list: list of integers. Samples to restrict data to.

            save: Boolean. Flag for determining whether or not to save the plot.
                           Saves to self.path


            kwargs: Arguments specific to choice of algorithm

        Output: Plot.
        '''
        # Imports
        import matplotlib.pyplot as plt
        import plotly
        import plotly.express as px


        # Load data
        if features is None:
            features = np.arange(self.new_data.shape[1])
            feature_label = ''
        else:
            feature_label = 'PFS'

        if idx_list is None:
            idx_list = np.arange(self.new_data.shape[0])

        data = deepcopy(self.new_data[idx_list, :])
        data = data[:, features]
        labels = kwargs.get('labels', self.generate_labels(attrname))
        labels = labels[idx_list].astype(str)
        n_samples = len(idx_list)

        k = kwargs.get('k', 1)
        mrkr_list = kwargs.get('mrkr_list', ['.', '+'])


        # Create method dictionary  and choose method type
        method_dict = {'umap': 'UMAP', 'pca': 'Principal Component Analysis', 'isomap': 'Isometric Mapping',
                       'lle': 'Local Linear Embedding', 'laplacian': 'Laplacian Eigenmaps',
                       'mds': 'Multidimensional Scaling', 'scae': 'Sparse Centroid Autoencoder',
                       'none': 'Generic Projection', 'mds-subspace': str(k) + '-Subspace MDS'}

        if method == 'pca':
            data_trans = calcom.visualizers.PCAVisualizer().project(data, dim=dimension)
        elif method == 'umap':
            import umap
            fit = umap.UMAP(min_dist=.1, n_neighbors=15, n_components=dimension)
            data_trans = fit.fit_transform(data)
        elif method == 'isomap':
            from sklearn.manifold import Isomap
            embedding = Isomap(n_components=dimension)
            data_trans = embedding.fit_transform(data)
        elif method == 'lle':
            from sklearn.manifold import LocallyLinearEmbedding
            # option ‘standard’, ‘hessian’, ‘modified’ or ‘ltsa’
            option = kwargs.get('type', 'standard')
            embedding = LocallyLinearEmbedding(n_components=dimension, method=option)
            data_trans = embedding.fit_transform(data)
        elif method == 'laplacian':
            from sklearn.manifold import SpectralEmbedding
            embedding = SpectralEmbedding(n_components=dimension)
            data_trans = embedding.fit_transform(data)
        elif method == 'mds':
            from sklearn.manifold import MDS
            embedding = MDS(n_components=dimension)
            data_trans = embedding.fit_transform(data)
            stress = embedding.stress_
        elif method == 'mds-subspace':
            from sklearn.manifold import MDS
            embedding = MDS(n_components=dimension, dissimilarity='precomputed')
            distance_mat = kwargs.get('distance_mat', None)
            data_trans = embedding.fit_transform(distance_mat)
            stress = embedding.stress_
        elif method == 'none':
            if dimension == 3:
                data_trans = data
            else:
                if data.shape[1] > 2:
                    A = np.random.rand(3, 2)
                    q, r = np.linalg.qr(A)
                    data_trans = np.matmul(data, q)
                else:
                    data_trans = data
        elif method == 'scae':

            # Grab train/test split
            n_samples = data.shape[0]
            labels = self.generate_labels(attrname)
            labels = labels[idx_list]
            trainTestSplit = kwargs.get('trainTestSplit', .8)
            shuffle_index = np.arange(n_samples)
            np.random.shuffle(shuffle_index)
            n_train = np.round(trainTestSplit * n_samples).astype(int)
            # n_test = n_samples - n_train
            dataTrain = data[shuffle_index[:n_train], :]
            labelsTrain = labels[shuffle_index[:n_train]]
            # dataTest = data[shuffle_index[n_train:], :]
            labelsTest = labels[shuffle_index[n_train:]]

            from calcom.classifiers import SparseCentroidencoderClassifier
            ce = SparseCentroidencoderClassifier()

            # store kwargs
            if kwargs.get('auto_layer_structure', False):
                ce.params['auto_layer_structure'] = True
            else:
                ce.params['hLayer'] = kwargs.get('hLayer', [25, 3, 25])
                ce.params['actFunc'] = kwargs.get('actFunc', ['tanh', 'tanh', 'tanh'])

            ce.params['errorFunc'] = kwargs.get('errorFunc', 'MSE')
            ce.params['optimizationFuncName'] = kwargs.get('optimizationFuncName', 'scg')
            ce.params['bottleneckArch'] = kwargs.get('bottleneckArch', False)
            ce.params['l1Penalty'] = kwargs.get('l1Penalty', 0.001)
            ce.params['noItrPre'] = kwargs.get('noItrPre', 10)
            ce.params['noItrPost'] = kwargs.get('noItrPost', 40)
            ce.params['noItrSoftmax'] = kwargs.get('noItrSoftmax', 10)
            ce.params['noItrFinetune'] = kwargs.get('noItrFinetune', 10)
            ce.params['batchFlag'] = kwargs.get('batchFlag', False)

            # Get bottleneck layer dimension and index
            bDim = np.min(ce.params['hLayer'])
            bIdx = np.argmin(ce.params['hLayer'])

            # Create embedding
            ce.fit(dataTrain, labelsTrain)
            from calcom.classifiers._centroidencoder.utilityDBN import standardizeData

            lOut = [standardizeData(data, ce._mu, ce._std)]
            tmpTrData = lOut[-1] * np.tile(ce.testClassifier.splW, (np.shape(data)[0], 1))
            lOut.append(tmpTrData)
            lLength = len(ce.testClassifier.netW)
            for j in range(bIdx + 1):
                d = np.dot(lOut[-1], ce.testClassifier.netW[j][1:, :]) + ce.testClassifier.netW[j][
                    0]  # first row in the weight is the bias
                # Take the activation function from the dictionary and apply it
                lOut.append(ce.feval('self.' + ce.testClassifier.actFunc[j], d) if j < bIdx else d)

            data_trans = lOut[-1]

            if bDim > 3:
                print("Using PCA to project bottleneck layer for visualization")
                data_trans = calcom.visualizers.PCAVisualizer().project(data_trans, dim=dimension)

        if dimension in [2, 3]:
            # generate data frame
            train_test_labels = kwargs.get('train_test_labels', np.array(['']*n_samples, dtype=object))
            if method == 'scae':
                train_test_labels[shuffle_index[:n_train]] = 0
                train_test_labels[shuffle_index[n_train:]] = 1
            elif method == 'mds-subspace':
                labels = kwargs.get('labels', [])

            if dimension == 2:
                df = pd.DataFrame(columns=['x', 'y'], data=data_trans, dtype=np.float)
            else:
                df = pd.DataFrame(columns=['x', 'y', 'z'], data=data_trans, dtype=np.float)
            df['label'] = labels
            df[attrname] = [label.replace(' Test', '').replace(' Train', '') for label in labels]
            df['mrkr'] = train_test_labels

            # determine plotting backend
            using_plotly = kwargs.get('using_plotly', False)
            marker_size = kwargs.get('marker_size', 100)

            # Choose dimension
            save_ext = kwargs.get('save_ext', feature_label)
            if not using_plotly:
                save_name = self.path + self.name + '_' + method + '_' + str(self.imputation_method) + '_' + str(self.normalization_method) + '_' + str(attrname) + '_' + save_ext
                x_label = 'Imputation: ' + str(self.imputation_method) + '\nNormalization: ' + str(self.normalization_method)
                title = 'Visualization of data set ' + self.name + ' using\n' + method_dict[method] + ' with labels given by ' + attrname

                plot_pandas(df=df, figsize=(14, 10), grp_colors=attrname, grp_mrkrs='mrkr', mrkr_list=mrkr_list,
                            title=title, dim=dimension, x_label=x_label, y_label='', mrkr_size=marker_size, grp_label='label', save_name=save_name)

            else:
                if dimension == 2:
                    if method in ['mds-subspace', 'scae']:
                        title = 'Visualization of data set ' + self.name + ' using<br>' + method_dict[
                            method] + ' with labels given by ' + attrname + '<br>Normalization: ' + str(self.normalization_method) + '; Imputation: ' + str(self.imputation_method)
                        fig = px.scatter(df, x='x', y='y', symbol='mrkr', color=attrname, title=title)
                        fig.update_traces(marker=dict(size=marker_size))
                        fig.for_each_trace(lambda trace: trace.update(marker_symbol="cross") if 'Test' in trace.name else (),
                                           )
                        fig.for_each_trace(
                            lambda trace: trace.update(marker_symbol="circle") if 'Train' in trace.name else (),
                            )
                        fig.update_layout(title={'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
                        if save:
                            #fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
                            plotly.offline.plot(fig, filename=(
                                        self.path + self.name + '_' + method + '_' + str(self.imputation_method) + '_' + str(
                                    self.normalization_method) + '_' + str(attrname) + '_' + save_ext + ".html"))
                    else:
                        title = 'Visualization of data set ' + self.name + ' using<br>' + method_dict[
                            method] + ' with labels given by ' + attrname + '<br>Normalization: ' + str(self.normalization_method) + '; Imputation: ' + str(self.imputation_method)
                        fig = px.scatter(df, x='x', y='y', color=attrname, title=title)
                        fig.update_traces(marker=dict(size=marker_size))
                        fig.for_each_trace(lambda trace: trace.update(marker_symbol="cross") if 'Test' in trace.name else (),
                                           )
                        fig.for_each_trace(
                            lambda trace: trace.update(marker_symbol="circle") if 'Train' in trace.name else (),
                            )
                        fig.update_layout(title={'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
                        if save:
                            #fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
                            plotly.offline.plot(fig, filename=(
                                        self.path + self.name + '_' + method + '_' + str(self.imputation_method) + '_' + str(
                                    self.normalization_method) + '_' + str(attrname) + '_' + feature_label + ".html"))

                elif dimension == 3:
                    if method in ['mds-subspace', 'scae']:
                        title = 'Visualization of data set ' + self.name + ' using<br>' + method_dict[
                            method] + ' with labels given by ' + attrname + '<br>Normalization: ' + str(self.normalization_method) + '; Imputation: ' + str(self.imputation_method)
                        fig = px.scatter_3d(df, x='x', y='y', z='z', symbol='mrkr', color=attrname, title=title)
                        fig.update_traces(marker=dict(size=marker_size))
                        fig.for_each_trace(lambda trace: trace.update(marker_symbol="cross") if 'Test' in trace.name else (),
                                           )
                        fig.for_each_trace(
                            lambda trace: trace.update(marker_symbol="circle") if 'Train' in trace.name else (),
                            )
                        fig.update_layout(title ={'y': 0.9,'x': 0.5,'xanchor': 'center', 'yanchor': 'top'})

                        if save:
                            #fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
                            plotly.offline.plot(fig, filename=(
                                        self.path + self.name + '_' + method + '_' + str(self.imputation_method) + '_' + str(
                                    self.normalization_method) + '_' + str(attrname) + '_' + save_ext + ".html"))
                    else:
                        title = 'Visualization of data set ' + self.name + ' using<br>' + method_dict[
                            method] + ' with labels given by ' + attrname + '<br>Normalization: ' + str(self.normalization_method)+ '; Imputation: ' + str(self.imputation_method)
                        fig = px.scatter_3d(df, x='x', y='y', z='z', color=attrname, title=title)
                        fig.update_traces(marker=dict(size=marker_size))
                        fig.for_each_trace(lambda trace: trace.update(marker_symbol="cross") if 'Test' in trace.name else (),
                                           )
                        fig.for_each_trace(
                            lambda trace: trace.update(marker_symbol="circle") if 'Train' in trace.name else (),
                            )
                        fig.update_layout(title={'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
                        if save:
                            #fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
                            plotly.offline.plot(fig, filename=(
                                        self.path + self.name + '_' + method + '_' + str(self.imputation_method) + '_' + str(
                                    self.normalization_method) + '_' + str(attrname) + '_' + feature_label + ".html"))

        if method == 'mds':
            return stress
        elif method == 'mds-subspace':
            return stress, data_trans

    def create_annotation(self, attrname='Health State'):
        # Imports
        import seaborn as sns

        # Create Classes
        attr_classes = self.partition(attrname)
        num_classes = len(attr_classes)

        # Initialize annotation data
        num_points = len(self.data)
        annot_data = [[] for i in range(num_points)]

        # Create colors for classes
        palette = sns.color_palette("Accent", num_classes)

        for i in range(num_classes):
            label = str(list(attr_classes.keys())[i]).upper()
            index = list(attr_classes.values())[i]

            for j in index:
                annot_data[j] = [i, palette[i], 'o', 50, label]

        return np.vstack(annot_data)

    def classify(self, method='ssvm', save=False, save_scores=False, **kwargs):
        '''This class method returns a classifer fit to a training set'''
        import calcom.classifiers as cc_classifiers

        # Load in general arguments
        feature_set = kwargs.get('feature_set', None)
        save_all_classifiers = kwargs.get('save_all_classifiers', False)
        score = kwargs.get('score', 'acc')
        bounds = kwargs.get('bounds', 'std')
        idx_list = kwargs.get('idx_list', np.arange(self.new_data.shape[0]))
        synth_generator = kwargs.get('synth_generator', None)
        attrname = kwargs.get('attrname', 'Health State')
        cv = kwargs.get('cv', 5)
        viz = kwargs.get('viz', False)

        # If using all features
        if feature_set is None:
            feature_label = ''
            feature_set = np.arange(self.new_data.shape[1])
        else:
            feature_label = 'PFS'

        if method == 'ssvm':
            # Load in specific args
            C = kwargs.get('C', [1])

            # Import CCexperiment and metrics
            import calcom.ccexperiment as cce
            import calcom.classifiers as ccc
            import calcom.metrics as metrics

            # Define classifier
            classifier = ccc.SSVMClassifier()
            classifier.params['use_cuda'] = True

            # Define metric
            evaluation_metric = metrics.ConfusionMatrix(return_measure=score)

            # Initialize accuracy scores vectors
            mean = []
            std = []
            min_vals = []
            max_vals = []

            # Chance reg parameter to ndarray if int
            if (type(C) == float) or (type(C) == int):
                C = [C]

            for c in C:
                # Run experiment
                classifier.params['C'] = c
                experiment = cce.CCExperiment(ccd=self, classifier_list=[classifier], idx_list=idx_list,
                                              classification_attr=attrname, cross_validation='k-fold',
                                              folds=cv, evaluation_metric=evaluation_metric, feature_set=feature_set,
                                              save_all_classifiers=save_all_classifiers,
                                              synth_generator=synth_generator)
                experiment.run()

                # Store accuracy scores
                mean.append(experiment.classifier_results['SSVMClassifier_0']['mean'])
                std.append(experiment.classifier_results['SSVMClassifier_0']['std'])
                min_vals.append(experiment.classifier_results['SSVMClassifier_0']['min'])
                max_vals.append(experiment.classifier_results['SSVMClassifier_0']['max'])

            mean = np.array(mean)
            std = np.array(std)
            min_vals = np.array(min_vals)
            max_vals = np.array(max_vals)

            if viz:
                # Plot accuracies
                import matplotlib.pyplot as plt

                # Basic Plot
                plt.figure(figsize=(20, 10))
                plt.plot(C, mean, 'k', color='#1B2ACC')
                if bounds == 'std':
                    plt.fill_between(C, mean - std, mean + std, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
                                     linewidth=4, linestyle='solid', antialiased=True)
                elif bounds == 'minmax':
                    plt.fill_between(C, min_vals, max_vals, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
                                     linewidth=4, linestyle='solid', antialiased=True)

                plt.yticks(np.arange(0, 1.05, .05))
                plt.xticks(C[0::10])

                if score == 'acc':
                    score_name = 'Accuracy'
                elif score == 'bsr':
                    score_name = 'BSR'

                plt.ylabel('k-fold ' + score_name + ' ( k = ' + str(cv) + ' )')
                plt.xlabel('Regularization Parameter (Low Reg --> High Reg)')
                plt.title('SSVM Classification on' + self.name + '\nby ' + attrname, fontsize=15)
                plt.show()

                if save:
                    plt.savefig(
                        fname=self.path + self.name + '_' + method + '_' + str(attrname) + '_' + str(self.imputation_method) + '_' + str(
                            self.normalization_method) + '_hypertuning.png', format='png')

            return experiment

        elif method == 'enet':
            import sklearn.linear_model.logistic as log_model
            Cs = kwargs.get('Cs', [10])
            cv = kwargs.get('cv', 5)
            penalty = kwargs.get('penalty', 'elasticnet')
            scoring = kwargs.get('scoring', 'accuracy')
            solver = 'saga'
            max_iter = kwargs.get('max_iter', 100)
            refit = kwargs.get('refit', True)
            multi_class = 'multinomial'
            l1_ratios = kwargs.get('l1_ratios', [.5])
            scale = kwargs.get('scale', True)
            n_jobs = kwargs.get('n_jobs', None)

            data = deepcopy(self.new_data[:, feature_set])
            data = data[idx_list, :]
            labels = self.generate_labels(attrname=attrname, idx_list=idx_list)

            if scale:
                import sklearn.preprocessing as sklp
                data = sklp.scale(data)

            classifier = log_model.LogisticRegressionCV(Cs=Cs, cv=cv, penalty=penalty, scoring=scoring, solver=solver,
                                                        max_iter=max_iter, refit=refit, multi_class=multi_class,
                                                        l1_ratios=l1_ratios, n_jobs=n_jobs)

            classifier.fit(data, labels)

            if viz:
                import matplotlib.pyplot as plt

                # Create Meshgrid
                if type(Cs) == int:
                    Cs = classifier.Cs_

                C, mu = np.meshgrid(Cs, l1_ratios)

                # Calculate mean accuracies across folds and classes
                acc = np.zeros_like(C.transpose())
                for key in classifier.scores_.keys():
                    acc = acc + classifier.scores_[key].mean(axis=0)

                acc = acc / len(classifier.scores_.keys())

                # Flip for meshgrids
                acc = acc.transpose()

                # Elastic Net plot
                plt.figure(figsize=(20, 10))
                plt.contourf(C, mu, acc, 100, cmap='jet')
                plt.xscale('log')
                plt.colorbar()
                plt.ylabel('lambda (Lasso --> Ridge Regression)')
                plt.xlabel('Inverse Regularization Parameter (High Reg --> Low Reg)')
                plt.title('Elastic Net Classification on' + self.name + '\nby ' + attrname, fontsize=15)

                # Label grid points with feature numbers
                for j in range(len(Cs)):
                    for k in range(len(l1_ratios)):
                        num_features = 0
                        for key in classifier.scores_.keys():
                            num_features_class = 0
                            for i in np.arange(cv):
                                weight = classifier.coefs_paths_[key][i, j, k, :]
                                num_features_class = num_features_class + np.count_nonzero(weight)
                            num_features_class = num_features_class / cv
                            num_features = num_features + num_features_class
                        num_features = np.ceil(num_features)
                        plt.annotate(s=str(int(num_features)), xy=(Cs[j], l1_ratios[k]), c='w')

                if save:
                    plt.savefig(
                        fname=self.path + self.name + '_' + method + '_' + str(self.imputation_method) + '_' + str(
                            self.normalization_method) + '_' + str(attrname), format='png')

            return classifier

        elif method == 'svm':
            from sklearn.svm import SVC

            # Load in specific args
            C = kwargs.get('C', [1])
            cv = kwargs.get('cv', 5)
            kernel = kwargs.get('kernel', 'linear')
            degree = kwargs.get('degree', 3)
            gamma = kwargs.get('gamma', 'scale')

            # Define classifier
            classifier = SVC(kernel=kernel, degree=degree, gamma=gamma)

            # Import CCexperiment and metrics
            import calcom.ccexperiment as cce
            import calcom.classifiers as ccc
            import calcom.metrics as metrics

            # Define metric
            evaluation_metric = metrics.ConfusionMatrix(return_measure='acc')

            # Initialize accuracy scores vectors
            mean = []
            std = []

            # Chance reg parameter to ndarray if int
            if (type(C) == float) or (type(C) == int):
                C = [C]

            for c in C:
                # Run experiment
                classifier.C = c
                experiment = cce.CCExperiment(ccd=self, classifier_list=[classifier], idx_list=idx_list,
                                              classification_attr=attrname, cross_validation='k-fold',
                                              folds=cv, evaluation_metric=evaluation_metric, feature_set=feature_set,
                                              save_all_classifiers=save_all_classifiers)
                experiment.run()

                # Store accuracy scores
                mean.append(experiment.classifier_results['SVC_0']['mean'])
                std.append(experiment.classifier_results['SVC_0']['std'])

            mean = np.array(mean)
            std = np.array(std)

            if viz:
                # Plot accuracies
                import matplotlib.pyplot as plt

                # Basic Plot
                plt.figure(figsize=(20, 10))
                plt.plot(C, mean, 'k', color='#1B2ACC')
                plt.fill_between(C, mean - std, mean + std,
                                 alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
                                 linewidth=4, linestyle='solid', antialiased=True)

                plt.yticks(np.arange(0, 1.05, .05))
                plt.xticks(C[0::10])
                plt.ylabel('k-fold Accuracy ( k = ' + str(cv) + ' )')
                plt.xlabel('Regularization Parameter (Low Reg --> High Reg)')
                plt.title('SVM Classification on' + self.name + '\nby ' + attrname, fontsize=15)
                plt.show()

                if save:
                    plt.savefig(
                        fname=self.path + self.name + '_' + method + '_' + str(self.imputation_method) + '_' + str(
                            self.normalization_method) + '_' + str(attrname), format='png')

            return experiment

        elif method == 'cae':

            # Import CCexperiment and metrics
            import calcom.ccexperiment as cce
            import calcom.metrics as metrics

            # Define metric
            evaluation_metric = metrics.ConfusionMatrix(return_measure=score)

            from calcom.classifiers import CentroidencoderClassifier

            ce = CentroidencoderClassifier()
            ce = calcom.classifiers.CentroidencoderClassifier()

            # store kwargs
            if kwargs.get('auto_layer_structure', False):
                ce.params['auto_layer_structure'] = True
            else:
                ce.params['hLayer'] = kwargs.get('hlayer', [25, 3, 25])
                ce.params['actFunc'] = kwargs.get('actFunc', ['tanh', 'tanh', 'tanh'])

            ce.params['errorFunc'] = kwargs.get('errorFunc', 'MSE')
            ce.params['optimizationFuncName'] = kwargs.get('optimizationFuncName', 'scg')
            ce.params['noItrPre'] = kwargs.get('noItrPre', 10)
            ce.params['noItrPost'] = kwargs.get('noItrPost', 40)
            ce.params['noItrSoftmax'] = kwargs.get('noItrSoftmax', 10)
            ce.params['noItrFinetune'] = kwargs.get('noItrFinetune', 10)
            ce.params['batchFlag'] = kwargs.get('batchFlag', False)

            experiment = cce.CCExperiment(ccd=self, classifier_list=[ce], idx_list=idx_list,
                                          classification_attr=attrname, cross_validation='k-fold',
                                          folds=cv, evaluation_metric=evaluation_metric, feature_set=feature_set,
                                          save_all_classifiers=save_all_classifiers,
                                          synth_generator=synth_generator)
            experiment.run()

            return experiment

        elif method == 'scae':

            # Import CCexperiment and metrics
            import calcom.ccexperiment as cce
            import calcom.metrics as metrics

            # Define metric
            evaluation_metric = metrics.ConfusionMatrix(return_measure=score)

            from calcom.classifiers import SparseCentroidencoderClassifier

            ce = SparseCentroidencoderClassifier()

            # store kwargs
            if kwargs.get('auto_layer_structure', False):
                ce.params['auto_layer_structure'] = True
            else:
                ce.params['hLayer'] = kwargs.get('hlayer', [25, 3, 25])
                ce.params['actFunc'] = kwargs.get('actFunc', ['tanh', 'tanh', 'tanh'])

            ce.params['errorFunc'] = kwargs.get('errorFunc', 'MSE')
            ce.params['optimizationFuncName'] = kwargs.get('optimizationFuncName', 'scg')
            ce.params['bottleneckArch'] = kwargs.get('bottleneckArch', False)
            ce.params['l1Penalty'] = kwargs.get('l1Penalty', 0.001)
            ce.params['noItrPre'] = kwargs.get('noItrPre', 10)
            ce.params['noItrPost'] = kwargs.get('noItrPost', 40)
            ce.params['noItrSoftmax'] = kwargs.get('noItrSoftmax', 10)
            ce.params['noItrFinetune'] = kwargs.get('noItrFinetune', 10)
            ce.params['batchFlag'] = kwargs.get('batchFlag', False)

            experiment = cce.CCExperiment(ccd=self, classifier_list=[ce], idx_list=idx_list,
                                          classification_attr=attrname, cross_validation='k-fold',
                                          folds=cv, evaluation_metric=evaluation_metric, feature_set=feature_set,
                                          save_all_classifiers=save_all_classifiers,
                                          synth_generator=synth_generator)
            experiment.run()

            if save_scores:
                scores = experiment.classifier_results['SparseCentroidencoderClassifier_0']['scores']

                df_scores = pd.DataFrame(index=['mean', 'min', 'max'] + list(range(cv)), columns=[score])
                score_min = np.min(scores)
                score_max = np.max(scores)
                score_mean = np.mean(scores)
                df_scores.at['mean', 'bsr'] = score_mean
                df_scores.at['min', 'bsr'] = score_min
                df_scores.at['max', 'bsr'] = score_max
                for j in range(cv):
                    df_scores.at[j, 'bsr'] = scores[j]

                df_scores.to_csv(
                    path_or_buf=x.path + x.name + '_' + method + '_' + score + '_' + 'Scores' + '_' + feature_label + '.csv')

            return experiment

    def feature_select(self, method='ssvm', save=False, **kwargs):
        '''This class method returns a feature set table given a method of feature selection'''

        if method == 'ssvm':

            # Intial Parameters
            cv = kwargs.get('cv', 5)
            save_all_classifiers = kwargs.get('save_all_classifiers', False)
            num_top_features = np.zeros(cv)
            df = pd.DataFrame(index=np.arange(self.new_data.shape[1]), columns=['_r'])

            # Run CCExperiment
            exp = self.classify(method=method, save=False, **kwargs)
            classifiers = exp.best_classifiers['SSVMClassifier']

            # Grab top features from each classifier
            if not save_all_classifiers:
                classifiers = [classifiers]

            for i, classifier in enumerate(classifiers):
                # Import KMeans
                from sklearn.cluster import KMeans

                # Compute number of top features via KMeans
                feature_weights = classifier.results['weight'].reshape(1, -1)

                # Split into 2 groups by scale and count smallest group
                scaled_features = np.log(np.abs(feature_weights))
                scaled_features = scaled_features.reshape(-1, 1)
                kmeans = KMeans(n_clusters=2, random_state=0).fit(scaled_features)
                num_group_0 = np.sum(kmeans.labels_ == 0)
                num_group_1 = np.sum(kmeans.labels_ == 1)
                num_top_features[i] = np.min([num_group_0, num_group_1])

                # Order features
                order = (-np.abs(feature_weights)).argsort()[0, :].tolist()
                df.loc[:, 'order_' + str(i)] = order
                df.loc[:, 'id_' + str(i)] = self.varattr.loc[order, 'id'].to_list()
                df.loc[:, 'rt_' + str(i)] = self.varattr.loc[order, 'rt'].to_list()
                df.loc[:, 'mz_' + str(i)] = self.varattr.loc[order, 'mz'].to_list()
                df.loc[:, 'Proportion Missing_' + str(i)] = self.varattr.loc[order, 'Proportion Missing'].to_list()
                df.loc[:, 'Feature Weight_' + str(i)] = feature_weights[0, :][order].tolist()

                # Retype
                df.astype({'order_' + str(i): 'int32', 'id_' + str(i): 'int32'}, copy=False)

            scores = [score for score in exp.classifier_results['SSVMClassifier_0']['scores']]
            df = df.drop('_r', 1)

        if method == 'scae':

            # Intial Parameters
            cv = kwargs.get('cv', 5)
            save_all_classifiers = kwargs.get('save_all_classifiers', False)
            num_top_features = np.zeros(cv)
            df = pd.DataFrame(index=np.arange(self.new_data.shape[1]), columns=['_r'])

            # Run CCExperiment
            exp = self.classify(method=method, save=False, **kwargs)
            classifiers = exp.best_classifiers['SparseCentroidencoderClassifier']

            # Grab top features from each classifier
            if not save_all_classifiers:
                classifiers = [classifiers]

            for i, classifier in enumerate(classifiers):
                # Import KMeans
                from sklearn.cluster import KMeans

                # Compute number of top features via KMeans
                feature_weights = classifier.testClassifier.splW

                # Split into 2 groups by scale and count smallest group
                scaled_features = np.log(np.abs(feature_weights))
                scaled_features = scaled_features.reshape(-1, 1)
                kmeans = KMeans(n_clusters=2, random_state=0).fit(scaled_features)
                num_group_0 = np.sum(kmeans.labels_ == 0)
                num_group_1 = np.sum(kmeans.labels_ == 1)
                num_top_features[i] = np.min([num_group_0, num_group_1])

                # Order features
                order = np.argsort(-np.abs(feature_weights)).tolist()
                df.loc[:, 'order_' + str(i)] = order
                df.loc[:, 'id_' + str(i)] = self.varattr.loc[order, 'id'].to_list()
                df.loc[:, 'rt_' + str(i)] = self.varattr.loc[order, 'rt'].to_list()
                df.loc[:, 'mz_' + str(i)] = self.varattr.loc[order, 'mz'].to_list()
                df.loc[:, 'Proportion Missing_' + str(i)] = self.varattr.loc[order, 'Proportion Missing'].to_list()
                df.loc[:, 'Feature Weight_' + str(i)] = feature_weights[order].tolist()

                # Retype
                df.astype({'order_' + str(i): 'int32', 'id_' + str(i): 'int32'}, copy=False)

            scores = [score for score in exp.classifier_results['SparseCentroidencoderClassifier_0']['scores']]
            df = df.drop('_r', 1)

        if method == 'enet':
            import sklearn.linear_model.logistic as log_model
            Cs = kwargs.get('Cs', [10])
            cv = kwargs.get('cv', 5)
            penalty = kwargs.get('penalty', 'elasticnet')
            scoring = kwargs.get('scoring', 'accuracy')
            solver = 'saga'
            max_iter = kwargs.get('max_iter', 100)
            refit = kwargs.get('refit', True)
            multi_class = 'multinomial'
            l1_ratios = kwargs.get('l1_ratios', [.5])
            scale = kwargs.get('scale', True)
            n_jobs = kwargs.get('n_jobs', None)

            if scale:
                import sklearn.preprocessing as sklp
                data = sklp.scale(data)

            classifier = log_model.LogisticRegressionCV(Cs=[Cs], cv=cv, penalty=penalty, scoring=scoring,
                                                        solver=solver,
                                                        max_iter=max_iter, refit=refit, multi_class=multi_class,
                                                        l1_ratios=[l1_ratios], n_jobs=n_jobs)

            # Fit and order features
            classifier.fit(data, labels)
            feature_weights = classifier.coef_
            order = (-np.abs(feature_weights)).argsort()[0, :].tolist()
            df = self.varattr.loc[order, ['rt', 'mz', 'Proportion Missing']]
            df.loc[order, 'Feature Weight'] = feature_weights[0, :][order]

        if save:
            attrname = kwargs.get('attrname', 'Health State')
            df.to_csv(path_or_buf=self.path + self.name + '_' + method + '_' + str(attrname) + '_' + self.imputation_method + '_' + self.normalization_method + '_features.csv')
        return df, num_top_features, scores

    def append_kfold_attr(self, k=5):
        """Creates a random attribute with k different classes"""
        num_samples = len(self.data)
        rand_attr = np.random.randint(low=1, high=k + 1, size=num_samples)
        self.append_attr('k-fold', rand_attr)

    def iter_feature_removal(self, threshold=.6, method='ssvm', save=False, num_features=None, **kwargs):
        """Iteratively remove features until classification rate is below a certain threshold"""

        # For testing
        acc_list = []

        # Get general args
        attrname = kwargs.get('attrname', 'Health State')
        idx_list = kwargs.get('idx_list', np.arange(self.new_data.shape[0]))

        # Get specific args
        if method == 'ssvm':
            C = kwargs.get('C', 1)
            cv = kwargs.get('cv', 5)

        # Check for self-defined number of features
        if not (num_features is None):
            num_features = int(num_features) * np.ones(cv, dtype=int)
            num_features_self_defined = True
        else:
            num_features_self_defined = False

        # Remove features while above threshold via do-while loop
        i = 0
        while True:
            # Feature Select
            feature_model = self.feature_select(method=method, **kwargs)
            features = feature_model[0]
            acc = np.mean(feature_model[2])
            acc_list.append(acc)

            # Get number of features
            if not (num_features_self_defined):
                num_features = feature_model[1]

            # Collect top features from different runs
            feature_list = np.array([])  # Initialize
            for i in range(cv):
                feature_list = np.append(feature_list, features['order_' + str(i)].to_list()[0:num_features[i]])

            # Convert to index
            feature_list = np.unique([int(item) for item in feature_list])

            # Remove Feature or break
            if acc < threshold:
                break
            else:
                self.remove_feature(identifier=feature_list)
                # self.removed_features.loc[i, 'Classification Rate'] = str(acc)+'/'+attrname
                # i += 1
        if save:
            self.removed_features.to_csv(
                path_or_buf=self.path + self.name + '_' + method + '_' + str(attrname) + '_' + 'removed_features.csv')

        return acc_list

    def __repr__(self):
        return '<CCLymeMetaboliteSet(%s) id=%s>' % (self.name, id(self))

    def __deepcopy__(self, memodict={}):
        df = self.generate_metadata()
        df.sort_index(inplace=True)
        labels = np.array(df.columns).reshape(1, len(df.columns))
        metadata = deepcopy(np.concatenate([labels, np.array(df)], axis=0), memodict)
        data = deepcopy(np.array(self.data), memodict)
        new_instance = CCLymeMetaboliteSet(data=data, metadata=metadata)
        new_instance.varattr = deepcopy(self.varattr, memodict)
        new_instance.name = deepcopy(self.name, memodict)
        new_instance.imputation_method = deepcopy(self.imputation_method, memodict)
        new_instance.normalization_method = deepcopy(self.normalization_method, memodict)
        new_instance.new_data = deepcopy(self.new_data, memodict)
        new_instance.missing_features = deepcopy(self.missing_features, memodict)
        new_instance.removed_features = deepcopy(self.removed_features, memodict)
        new_instance.path = deepcopy(self.path, memodict)
        return new_instance

    def generate_subspaces(self, attrname='Case Type', feature_set=None, idx_list=None, n_neighbors=2, train_test_split=.8, smote=False, **kwargs):
        from sklearn.neighbors import NearestNeighbors

        # Load data
        if feature_set is None:
            feature_set = np.arange(self.new_data.shape[1])

        if idx_list is None:
            idx_list = np.arange(self.new_data.shape[0])

        data = self.new_data[:, feature_set]
        data = data[idx_list, :]
        m, n = data.shape

        # split into training/test
        labels = self.generate_labels(attrname)
        labels = labels[idx_list]
        shuffle_index = np.arange(m)
        np.random.shuffle(shuffle_index)
        n_train = np.round(train_test_split * m).astype(int)
        data_train = data[shuffle_index[:n_train], :]
        labels_train = labels[shuffle_index[:n_train]]
        labels_train = labels_train.astype(object) + ' Train'
        data_test = data[shuffle_index[n_train:], :]
        labels_test = labels[shuffle_index[n_train:]]
        labels_test = labels_test.astype(object) + ' Test'

        if smote:
            from imblearn.over_sampling import SMOTE
            k = kwargs.get('k', 3)
            smote = SMOTE(random_state=42, k_neighbors=k, sampling_strategy='minority')
            data_train, labels_train = smote.fit_resample(data_train, labels_train)
            n_train = len(labels_train)

        data = np.concatenate((data_train, data_test))
        labels = np.concatenate((labels_train, labels_test))


        # global nearest neighbors
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        neigh.fit(data)
        index = np.arange(data.shape[0]).reshape(-1, 1)
        indices = neigh.kneighbors()[1]
        indices = np.concatenate((index, np.array(indices)), axis=1)

        # local nearest neighbors
        for label in np.unique(labels_train):
            label_index = np.argwhere(labels_train == label).reshape(-1,)
            label_dict = dict(enumerate(label_index))
            neigh.fit(data_train[label_index, :])
            label_indices = neigh.kneighbors()[1]
            label_indices = np.vectorize(label_dict.get)(np.array(label_indices))
            label_indices = np.concatenate((label_index.reshape(-1, 1), label_indices), axis=1)
            indices[label_index, :] = label_indices

        indices = np.reshape(indices.transpose(), (-1,), order='F')
        S = data[indices, :]
        S = S.reshape(-1)
        params = np.array([len(labels), len(feature_set), (n_neighbors + 1)])
        S = np.concatenate((params, S), axis=0)

        return S, labels


def Update_scores(modelMin, modelMax, modelMean, modelStd):
    def fill_array_scores(results):
        # Grab results
        iPre = results[0]
        iPost = results[1]
        iSoft = results[2]
        iFine = results[3]
        iAct = results[4]
        iLayer = results[5]
        score_min = results[6]
        score_max = results[7]
        score_mean = results[8]
        score_std = results[9]

        modelMin[iPre, iPost, iSoft, iFine, iAct, iLayer] = score_min
        modelMax[iPre, iPost, iSoft, iFine, iAct, iLayer] = score_max
        modelMean[iPre, iPost, iSoft, iFine, iAct, iLayer] = score_mean
        modelStd[iPre, iPost, iSoft, iFine, iAct, iLayer] = score_std

        return

    return fill_array_scores


if __name__ == "none":
    # Internal Script for Slipstick
    import calcom
    import numpy as np
    import datetime
    import os
    from sklearn.metrics import confusion_matrix
    import pandas as pd

    # Set date
    dt = datetime.datetime.now()
    d_truncated = datetime.date(dt.year, dt.month, dt.day)

    # Set parameters
    cv = 5
    use_all_features = True
    feature_select = True
    C_range = np.arange(1, 10, .2)
    classifier = 'scae'
    normalization_method = 'log'
    imputation_method = 'knn'
    save_loc = '/s/a/home/ekehoe/Metabolomics_Data/urine_feature_selection/' + d_truncated.__str__()
    viz_method = 'mds'
    ni = True

    # Create new directory
    try:
        os.mkdir(save_loc)
    except Exception:
        pass
    save_loc = save_loc + '/'

    # Load in dataset
    x = CCLymeMetaboliteSet(
        data_file_path="/s/a/home/ekehoe/Metabolomics_Data/DoD-Urine-Children_skyline-list-for-Eric_all-features_20200617_adjusted.csv",
        data_format="Skyline",
        metadata_file_path="/s/a/home/ekehoe/Metabolomics_Data/Updated_DoD_LiseNigrovic_Urine_102919_sentTo-BF-NI_02182020_adjusted.csv",
        metadata_format=1,
        osmolality_file_path="/s/a/home/ekehoe/Metabolomics_Data/DOD Urine Osmolarity_Lise_ChildrenSamples_102919.csv")

    # Normalize and Impute
    if ni:
        # x.normalize(algorithm='osmolality')
        x.normalize(algorithm=normalization_method)
        # x.impute(algorithm=imputation_method)
    else:
        # x.impute(algorithm=imputation_method)
        # x.normalize(algorithm='osmolality')
        x.normalize(algorithm=normalization_method)

    # Grab Confirmed Lyme and Control samples
    case_partition = x.partition('Case Type')
    healthy = case_partition['Control Dod']
    lyme = case_partition['Lyme Case']
    unknown = case_partition['Lyme Control']

    # samples = healthy + lyme + unknown
    samples = healthy + lyme

    # Visualize pre-selection
    x.path = save_loc
    x.visualize(attrname='Case Type', dimension=2, method=viz_method, save=True)

    # Define synth generator for balancing classes
    synth_generator = calcom.synthdata.SmoteGenerator()
    synth_generator.params['k'] = 3  # 3 nearest neighbors

    ##### SSVM ######
    if classifier == 'ssvm':
        # Run classifier and save
        x.classify(method=classifier, attrname='Final Result', synth_generator=synth_generator, cv=cv,
                   C=np.arange(0, 3, .03), score='bsr', bounds='minmax', viz=True, save=True)

        # Choose C parameter, feature select, and save
        x.classify(method=classifier, idx_list=samples, attrname='Case Type', synth_generator=synth_generator, cv=cv,
                   C=[1.17], score='bsr', bounds='minmax')
        [df, num_top_features, acc] = x.feature_select(method=classifier, attrname='Case Type', idx_list=samples,
                                                       C=1.18,
                                                       synth_generator=synth_generator, score='bsr', save=True,
                                                       save_all_classifiers=True)
    elif classifier == 'scae':
        # Visualize w/ SCAE
        x.visualize(idx_list=samples, attrname='Case Type', dimension=2, method='scae', noItrPre=20,
                    noItrPost=50, noItrSoftmax=10, noItrFinetune=50, hLayer=[128, 16, 128],
                    actFunc=['tanh', 'tanh', 'tanh'], trainTestSplit=.8, save=True)

        # Run classifier
        experiment = x.classify(method='scae', idx_list=samples, attrname='Case Type', synth_generator=synth_generator,
                                cv=cv, score='bsr', hLayer=[128, 16, 128], actFunc=['tanh', 'tanh', 'tanh'],
                                noItrPre=20, noItrPost=50, noItrSoftmax=10, noItrFinetune=50, save_scores=True,
                                save_all_classifiers=True)

        [df, num_top_features, acc] = x.feature_select(method='scae', attrname='Case Type', idx_list=samples,
                                                       synth_generator=synth_generator, score='bsr', cv=cv,
                                                       hLayer=[128, 16, 128], actFunc=['tanh', 'tanh', 'tanh'],
                                                       noItrPre=20, noItrPost=50, noItrSoftmax=10, noItrFinetune=50,
                                                       save=True, save_all_classifiers=True)

    # Grab Top 5 Features from each and collaspe
    feature_list = np.array([])  # Initialize
    for index in range(5):
        feature_list = np.append(feature_list, df['order_' + str(index)].to_list()[0:5])

    # Count occurences
    feature_list = [int(item) for item in feature_list]
    occurences = [[j, feature_list.count(j)] for j in set(feature_list)]
    feature_list = np.unique([int(item) for item in feature_list])

    # Grab Cumulative Top 5 Features
    df_top = x.varattr.loc[feature_list, ['id', 'rt', 'mz', 'Proportion Missing']]

    # Add occurence column
    df_top.loc[:, 'Occurence'] = 0  # Initialize
    for pair in occurences:
        df_top.at[pair[0], 'Occurence'] = pair[1]

    # Sort Cumulative Top 5
    df_top.sort_values(by=['Occurence'], inplace=True, ascending=False)

    # Save Cumulative Top 5
    df_top.to_csv(path_or_buf=x.path + x.name + '_' + classifier + '_' + 'Case_Type_Top_5_features.csv')

    # Get accuracy off Top 5
    x.classify(method='scae', idx_list=samples, feature_set=feature_list, attrname='Case Type',
               synth_generator=synth_generator, cv=cv, score='bsr', hLayer=[128, 16, 128],
               actFunc=['tanh', 'tanh', 'tanh'], noItrPre=20, noItrPost=50, noItrSoftmax=10, noItrFinetune=50,
               save_scores=True, save_all_classifiers=True)

    # Visualize post-selection and save
    x.visualize(features=feature_list, idx_list=samples, attrname='Case Type', dimension=2, method=viz_method,
                save=True)

    # Visualize w/ SCAE
    x.visualize(idx_list=samples, features=feature_list, attrname='Case Type', dimension=2, method='scae', noItrPre=20,
                noItrPost=50, noItrSoftmax=10, noItrFinetune=50, hLayer=[25, 2, 25],
                actFunc=['tanh', 'tanh', 'tanh'], trainTestSplit=.8, save=True)
