import pandas as pd
import numpy as np

def cleandf(df):

    # minimum fare is zero, replacing them with mean value for their class
    df['Fare'] = df['Fare'].map(lambda x: np.nan if x==0 else x)
    df['Fare'] = df.groupby(['Pclass'])['Fare'].transform(lambda x: x.fillna(x.mean()))

    # age
    meanAge = np.mean(df['Age'])
    df['Age'] = df['Age'].fillna(meanAge)

    # Cabin (this column is dropped in initial fitting).
    df['Cabin'] = df['Cabin'].fillna('Unknown')

    # Embarked
    from scipy.stats import mode
    modeEmbarked = mode(df['Embarked'])[0][0]
    df['Embarked'] = df['Embarked'].fillna(modeEmbarked)

    # create dummies for categorical features
    dummies=[]
    cols = ['Pclass', 'Sex', 'Embarked']
    for col in cols:
        dummies.append(pd.get_dummies(df[col]))

    df_dummies = pd.concat(dummies, axis=1)

    df = pd.concat((df, df_dummies), axis=1)

    #drop columns in training dataset
    df = df.drop(['Pclass', 'Sex', 'Embarked', 'Cabin', 'Name', 'Ticket'], axis=1)

    return df

def cleaneddf():

    train_df = cleandf(pd.read_csv('train.csv', header=0))
    train_df = train_df.drop(['PassengerId'], axis=1)
    test_df = cleandf(pd.read_csv('test.csv', header=0))

    return [train_df, test_df]

def write_to_csv(passengerid, results):
    output = np.column_stack((passengerid, results))
    df_results = pd.DataFrame(output.astype('int'), columns=['PassengerId', 'Survived'])
    df_results.to_csv('titanic_results.csv', index=False)