import numpy as np
import pickle
from sklearn.model_selection import KFold
import torch
import pandas as pd
import numpy as np
from sklearn import preprocessing


def csv_load(filename, cost_from_file):
    # csv format
    # 1st row : cost from 2nd col (align with column name in 2nd row) if
    # cost_from_file if True. Else below description of 2nd row is for 1st row
    # 2nd row : columns name starting with 'label' followed by features' name
    # 3rd ~ rows : label and feature values
    header = 1 if cost_from_file else 0
    df = pd.read_csv(filename, header=header)
    labels = df['label'].values.astype(np.int)
    df = df.iloc[:, 1:] # assume 1st col is label
    exist = np.where(pd.isnull(df), 1, 0).astype(np.uint8)

    def norm_to_zero_one(df):
        return (df - df.min(axis=1)) * 1.0 / (df.max(axis=1) - df.min(axis=1))
    def std_norm(df):
        return (df - df.mean(axis=1)) / df.std(axis=1)

    df = norm_to_zero_one(df)
    df = df.fillna(0)#method='backfill')

    if cost_from_file:
        cost = pd.read_csv(filename, nrows=1)
        cost = cost.values.reshape(-1)[1:]
        assert len(cost) == df.shape[1]
    else:
        cost = None

    return df.values.astype(np.float32), exist, labels, cost

def gen_cube(n_features=20, data_points=20000, sigma=0.1, seed=123):
    assert n_features >= 10, 'cube data have >= 10 num of features'
    np.random.seed(seed)
    clean_points = np.random.binomial(1, 0.5, (data_points, 3))
    labels = np.dot(clean_points, np.array([1,2,4]))
    points = clean_points + np.random.normal(0, sigma, (data_points, 3))
    features = np.random.rand(data_points, n_features)
    for i in range(data_points):
        offset = labels[i];
        for j in range(3):
            features[i, offset + j] = points[i, j]
    return features, labels

def data_split_n_ready(features, exist, labels,
        mode='cv', random_seed=123,
        val_test_split=np.array([0.25, 0.25]), action2features=None,
        shuffle=True):
    dataset_size = len(features)
    indices = list(range(dataset_size))
    assert np.sum(val_test_split) < 1
    split = np.floor(val_test_split * dataset_size).astype(np.int)
    split = np.cumsum(split)
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices = indices[split[1]:]
    val_indices = indices[:split[0]]
    test_indices = indices[split[0]:split[1]]

    def pick(indices, iter):
        exist_ = exist[indices] if exist is not None else None
        return DataTemp(features[indices], labels[indices], exist_,
                n_classes=(np.amax(labels) + 1), # label from 0 to n_classes-1
                action2features=action2features, iter=iter)

    return pick(train_indices, True), \
            pick(val_indices, False), pick(test_indices, False)

def data_load(data_type='cube_20_0.3', # "cube_Nfeatures_sigma" or csv
              random_seed=123,
              n_datapoints=20000, # ignored when data_type is csv
              csv_filename=None,
              action2features=None,
              val_test_split=np.array([0.25, 0.25]),
              cost_from_file=False):
    if data_type == 'csv':
        features, exist, labels, cost = csv_load(csv_filename, cost_from_file)
    else:
        data_type = data_type.split(sep='_')
        assert len(data_type) == 3 and data_type[0] == 'cube', "Undefined datatype"
        n_features = int(data_type[1])
        sigma = float(data_type[2])
        features, labels = gen_cube(n_features, data_points=n_datapoints,
                sigma=sigma, seed=random_seed)
        exist = None
        action2features = None
        cost = None

    return data_split_n_ready(features, exist, labels,
                            val_test_split=val_test_split,
                            action2features=action2features), cost


class DataTemp:
    def __init__(self, features, labels, exist, n_classes, shuffle=True, iter=True,
            action2features=None):
        self.features = features
        self.labels = labels
        self.exist = exist
        self.shuffle = shuffle
        self.n_data, self.n_features = features.shape
        self.n_classes = n_classes
        self.index = 0
        self.iter = iter
        self.action2features = action2features
        self.n_actions = len(action2features) + 1 if action2features is not None \
                else features.shape[1] + 1

    def next_batch(self, batch_size):
        if iter:
            new_index = (self.index + batch_size) % self.n_data
        else:
            if self.index == self.n_data:
                return None # Done
            new_index = min(self.index + batch_size, self.n_data)

        if self.index + batch_size <= self.n_data:
            features = self.features[self.index: self.index + batch_size]
            labels = self.labels[self.index: self.index + batch_size]
            exist = self.exist[self.index: self.index + batch_size] \
                if self.exist is not None else None
        else:
            features = self.features[self.index:]
            labels = self.labels[self.index:]
            exist = self.exist[self.index:] if self.exist is not None else None
            if self.iter:
                if self.shuffle:
                    p = np.random.permutation(self.n_data)
                    self.features = self.features[p]
                    self.labels = self.labels[p]
                    self.exist = self.exist[p] if self.exist is not None else None
                features = np.concatenate((features,
                    self.features[:new_index]), axis=0)
                labels = np.concatenate((labels,
                    self.labels[:new_index]), axis=0)
                exist = np.concatenate((exist, self.exist[:new_index]), axis=0) \
                        if self.exist is not None else None
        self.index = new_index
        return features, labels, exist



fatty_liver_3_cost = {
    (0, 'Sex'): 1,
    (1, 'Age'): 1,
    (2, 'HTN'): 100,
    (3, 'DM'): 100,
    (4, 'Cirrhosis'): 100,
    (5, 'ALCOHOL_HX'): 100,
    (6, 'Smoking Hx.1'): 100,
    (7, 'Height'): 100,
    (8, 'Weight'): 100,
    (9, 'BMI'): 100,
    (10, 'WHR'): 100,
    (11, 'BP_HIGH'): 100,
    (12, 'BP_LWST'): 100,
    (13, 'MUSCLE') : 100,
    (14, 'FAT') : 100,
    (15, '근육량 + 체지방량'): 100,
    (16, 'SKELETAL_MUSCLE'): 100,
    (17, 'FAT_RATE'): 100,
    (18, 'ABDOMINAL_FAT_RATE'): 100,
    (19, 'VISCERAL_FAT'): 100,
    (20, 'Testosterone'): 12100,
    (21, 'T3') : 10800,
    (22, 'TPO-ab') : 1e5,
    (23, 'Vit D') : 1e5,
    (24, 'FT4') : 12100,
    (25, 'TSH') : 15800,
    (26, 'RF') : 7500,
    (27, 'HBsAg') : 9800,
    (28, 'Anti_HBcAb') : 15000,
    (29, 'Anti_HBsAb') : 10600,
    (30, 'Anti_HCV') : 14700,
    (31, 'FTA_ABS') : 12100,
    (32, 'VDRL') : 1800,
    (33, 'AFP') : 12000,
    (34, 'CEA') : 29325,
    (35, 'CA15-3') : 1e5,
    (36, 'CA19-9') : 18500,
    (37, 'CA125'): 16300,
    (38, 'PSA'): 15500,
    ((39, 'Proteinuria'), (40, 'Hematuria')): 2940,
    (41, 'WBC'): 1900,
    (42, 'Hb'): 4900,
    (43, 'PLT'): 900,
    (44, 'Homocystein'): 47000,
    ((45, 'HbA1c'), (46, 'est_Ave')): 6600, # 당화혈색소 구할때 나옴
    (47, 'Ca'): 1300,
    (48, 'P'): 1300,
    (49, 'Glu'): 1200,
    (50, 'BUN'): 1500,
    ((51, 'Cr'), (70, 'GFR')): 1300,
    (52, 'Uric_Acid'): 1500,
    (53, 'Chole'): 1500,
    (54, 'Protein'): 1e5,
    (55, 'Alb'): 1600,
    (56, 'T_Bil'): 1200,
    (57, 'ALP'): 1400,
    (58, 'OT'): 1700,
    (59, 'PT'): 1600,
    (60, 'GGT'): 3000,
    (61, 'LD'): 2500,
    (62, 'TG'): 3200,
    (63, 'HDL'): 5600,
    (64, 'LDL'): 5800,
    (65, 'Na'): 1300,
    (66, 'K'): 1300,
    (67, 'CRP_HS'): 6700,
    (68, 'TIBC'): 2300,
    (69, 'Ferritin'): 10200,
    ((71, 'SpineBMC'), (72, 'SpineBMD'), (73, 'SpineT_score'), (74,
        'Femur_Neck_BMC'), (75, 'Femur_Neck_BMD'), (76, 'Femur_Neck_T'), (77,
            'Total_Hip_BMC'), (78, 'Total_Hip_BMD'), (79, 'Total_Hip_T')):
            96050,
    ((80, 'IMT_R_Max'), (81, 'IMT_L_Max'), (82, 'IMT_Carotid Plaque'), (83,
        'IMT_R_Plaque'), (84, 'IMT_L_Plaque'), (85, 'IMT_R_CCA_Size'), (86,
            'IMT_R_ICA_Size'), (87, 'IMT_R_bulb_Size'), (88, 'IMT_L_CCA_Size'),
        (89, 'IMT_L_ICA_Size'), (90, 'IMT_L_bulb_Size')): 156400,
    ((91, 'Echo_EF'), (92, 'Echo_E/E')): 253300,
    ((93, 'Coronary_CT'), (94, 'Coronary_Calciumscore')): 296650,
    (95, 'Brain_MRA, MRI'): 1112650
}

fl_keys = list(sorted(fatty_liver_3_cost.keys(), key=lambda x: x[0] if
    isinstance(x[0], int)
    else x[0][0]))

fl_action2features = {i: x[0] if isinstance(x[0], int) else \
        (lambda x: [y[0] for y in x])(x) for i, x in enumerate(fl_keys)}

fl_action2cost = [fatty_liver_3_cost[fl_keys[i]] for i in range(len(fl_keys))]


imt_cost = {((0, 'HCC_1'), (1, 'HCC_2'), (2, 'HCC_3'), (3, 'HCC_4'), (4, 'HCC_5')): 100,
 (5, 'Sex'): 1,
 (6, 'Age'): 1,
 ((7, 'Stiffness (kPa)'), (8, 'IQR (kPa)'), (9, ' SR (%)'), (10, 'IQR/med.(%)'), (11, 'steatosis (dB/m)'), (12, 'IQR (dB/m)'), (13, 'IQR/med.(%).1')): 80000,
 (14, 'HTN'): 100,
 (15, 'DM'): 100,
 (16, 'Cirrhosis'): 100,
 (17, 'ALCOHOL_HX'): 100,
 (18, 'Smoking Hx.1'): 100,
 (19, 'Height'): 100,
 (20, 'Weight'): 100,
 (21, 'BMI'): 100,
 (22, 'WHR'): 100,
 ((23, 'BP_HIGH'), (24, 'BP_LWST')): 100,
 ((25, 'MUSCLE'), (26, 'FAT'), (27, '근육량 + 체지방량'), (28, 'SKELETAL_MUSCLE'), (29, 'FAT_RATE'), (30, 'ABDOMINAL_FAT_RATE'), (31, 'VISCERAL_FAT')): 100,
 (32, 'Testosterone'): 12100, (33, 'T3'): 10800,
 (34, 'TPO-ab'): 1e5,
 (35, 'Vit D'): 1e5,
 (36, 'FT4'): 12100,
 (37, 'TSH'): 15800,
 (38, 'RF'): 7500,
 (39, 'HBsAg'): 9800,
 (40, 'Anti_HBcAb'): 15000,
 (41, 'Anti_HBsAb'): 10600,
 (42, 'Anti_HCV'): 14700,
 (43, 'FTA_ABS'): 12100,
 (44, 'VDRL'): 1800,
 (45, 'AFP'): 12000,
 (46, 'CEA'): 29325,
 (47, 'CA15-3'): 1e5,
 (48, 'CA19-9'): 18500,
 (49, 'CA125'): 16300,
 (50, 'PSA'): 15500,
 ((51, 'Proteinuria'), (52, 'Hematuria')): 2940,
 (53, 'WBC'): 1900,
 (54, 'Hb'): 4900,
 (55, 'PLT'):900,
 (56, 'Homocystein'): 47000,
 ((57, 'HbA1c'), (58, 'est_Ave')): 6600,
 (59, 'Ca'): 1300,
 (60, 'P'): 1300,
 (61, 'Glu'): 1200,
 (62, 'BUN'): 1500,
 ((63, 'Cr'), (82, 'GFR')): 1300,
 (64, 'Uric_Acid') : 1500,
 (65, 'Chole'): 1500,
 (66, 'Protein'): 1e5,
 (67, 'Alb'): 1600,
 (68, 'T_Bil'): 1200,
 (69, 'ALP'): 1400,
 (70, 'OT'): 1700,
 (71, 'PT'): 1600,
 (72, 'GGT'): 3000,
 (73, 'LD'): 2500,
 (74, 'TG'): 3200,
 (75, 'HDL'): 5600,
 (76, 'LDL'): 5800,
 (77, 'Na'): 1300,
 (78, 'K'): 1300,
 (79, 'CRP_HS'): 6700,
 (80, 'TIBC'): 2300,
 (81, 'Ferritin'):10200,
 ((83, 'SpineBMC'), (84, 'SpineBMD'), (85, 'SpineT_score'), (86, 'Femur_Neck_BMC'), (87, 'Femur_Neck_BMD'), (88, 'Femur_Neck_T'), (89, 'Total_Hip_BMC'), (90, 'Total_Hip_BMD'), (91, 'Total_Hip_T')): 96050,
 ((92, 'Echo_EF'), (93, "Echo_E/E'")): 253300,
 (94, 'Coronary_Calciumscore'): 296650,
 (95, 'US_Fatty_Liver'): 1112650}


imt_keys = list(sorted(imt_cost.keys(), key=lambda x: x[0] if
    isinstance(x[0], int)
    else x[0][0]))

imt_action2features = {i: x[0] if isinstance(x[0], int) else \
        (lambda x: [y[0] for y in x])(x) for i, x in enumerate(imt_keys)}

imt_action2cost = [imt_cost[imt_keys[i]]  for i in range(len(imt_keys))]
