from folktables.folktables import ACSDataSource, ACSIncome, ACSIncome_no_transform
import pandas as pd

def main():
    
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    
    features_classification, label_classification, group_classification = ACSIncome.df_to_numpy(acs_data)
    features_regression, label_regression, group_regression = ACSIncome_no_transform.df_to_numpy(acs_data)
    
    feature_names = [
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
    ]

    features_classification = pd.DataFrame(features_classification)
    features_regression = pd.DataFrame(features_regression)
    features_classification.columns = feature_names
    features_regression.columns = feature_names
    features_classification['>50K'] = label_classification * 1
    features_regression['Income'] = label_regression
    
    features_classification.to_csv(path_or_buf="folktables_classification.csv")
    features_regression.to_csv(path_or_buf="folktables_regression.csv")
    
if __name__ == "__main__":
    main()