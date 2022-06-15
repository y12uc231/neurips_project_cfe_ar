import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    
    task = "classification"
    filePath = './Data_Sets/Folktables/'

    if task == "regression":
        datafile = 'folktables_regression.csv'
    else:
        datafile = 'folktables_classification.csv'


    column_names = [
        'AGEP',
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
        'RAC1P',
        'Income'
    ]

    categorical_features = [
        'COW',
        'MAR',
        'RELP',
        'SEX',
        'RAC1P',
    ]

    # Read Data from csv
    df = pd.read_csv(filePath + datafile, index_col=False, skipinitialspace=True, header='infer')
    df = df.drop(columns='Unnamed: 0')
    
    # simplify categories from workclass
    workclass_map = {1: 'Private',
                     2: 'Private',
                     3: 'Non-Private',
                     4: 'Non-Private',
                     5: 'Non-Private',
                     6: 'Non-Private',
                     7: 'Non-Private',
                     8: 'Non-Private',
                     9: 'Non-Private'
                     }
    df['COW'] = df['COW'].map(workclass_map)

    # change marital status column
    df['MAR'] = df['MAR'].replace([2, 3, 4, 5], 'Non-Married')
    df['MAR'] = df['MAR'].replace([1], 'Married')

    # drop POB column -> too many categories
    df = df.drop(columns=['OCCP'])
    df = df.drop(columns=['POBP'])

    # change race column
    race_map = {1: 'White', 2: 'Non-White', 3: 'Non-White',
                4: 'Non-White', 5: 'Non-White', 6: 'Non-White',
                7: 'Non-White', 8: 'Non-White', 9: 'Non-White'}
    df['RAC1P'] = df['RAC1P'].map(race_map)

    # change relationship column
    rel_map = {0: 'non-Husband/Wife', 1: 'Husband/Wife',
               3: 'non-Husband/Wife', 10: 'non-Husband/Wife',
               4: 'non-Husband/Wife', 11: 'non-Husband/Wife',
               5: 'non-Husband/Wife', 12: 'non-Husband/Wife',
               6: 'non-Husband/Wife', 13: 'non-Husband/Wife',
               7: 'non-Husband/Wife', 14: 'non-Husband/Wife',
               8: 'non-Husband/Wife', 15: 'non-Husband/Wife',
               9: 'non-Husband/Wife', 16: 'non-Husband/Wife', 17: 'non-Husband/Wife',
               }
    
    df['RELP'] = df['RELP'].map(rel_map)

    # One hot encode all categorical features
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # save cleansed data in edited folder
    df.to_csv(filePath + 'folktables_processed_full.csv', index=False)
    
    
    if task == "classification":
        X_train, X_test = train_test_split(df, stratify=df['>50K'], test_size=0.20)
    else:
        X_train, X_test = train_test_split(df, test_size=0.20)
    
    if task == "regression":
        X_train.to_csv('Data_Sets/Folktables/folktables_regression-train.csv', index=False)
        X_test.to_csv('Data_Sets/Folktables/folktables_regression-test.csv', index=False)
    else:
        X_train.to_csv('Data_Sets/Folktables/folktables_classification-train.csv', index=False)
        X_test.to_csv('Data_Sets/Folktables/folktables_classification-test.csv', index=False)

if __name__ == "__main__":
    main()
