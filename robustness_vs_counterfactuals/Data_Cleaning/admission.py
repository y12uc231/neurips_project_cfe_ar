import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    
    filePath = 'Data_Sets/Admission/'
    data_name = 'law_admission_processed.csv'
    
    # Read Data from csv
    all = pd.read_csv(filePath + data_name)
    # if data is already cleansed, don't do anything
    
    all = all.drop(columns='lsat')
    
    X_train, X_test = train_test_split(all, test_size=0.20)
    X_train.to_csv('Data_Sets/Admission/admission-train.csv', index=False)
    X_test.to_csv('Data_Sets/Admission/admission-test.csv', index=False)

    
    
if __name__ == "__main__":

    main()
