import pandas as pd


def resample(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Resample the original dataframe to have a balanced distribution of classes.
    :param data: original dataframe
    :return: resampled dataframe
    """

    lst = []
    min_size = data[target].value_counts().min()
    for class_index, group in data.groupby(target):
        lst.append(group.sample(n=min_size if len(group) < min_size*2 else min_size*2, replace=False))
    df = pd.concat(lst)
    return df

def fix_give_me_credit(df: pd.DataFrame, y: pd.DataFrame):
    
    # Complete
    #df["target"] = y
    #df = resample(df, "target")
    #df.drop(columns=["target"], inplace=True)

    # https://www.kaggle.com/code/simonpfish/comp-stats-group-data-project-final
    df.dropna(inplace=True)
    df.loc[df['DebtRatio'] > 1, 'DebtRatio'] = 1
    df.loc[df['MonthlyIncome'] > 17000, 'MonthlyIncome'] = 17000
    df.loc[df['RevolvingUtilizationOfUnsecuredLines'] > 1, 'RevolvingUtilizationOfUnsecuredLines'] = 1
    dfn98 = df.copy()
    dfn98.loc[dfn98['NumberOfTime30-59DaysPastDueNotWorse'] > 90, 'NumberOfTime30-59DaysPastDueNotWorse'] = 18
    dfn98.loc[dfn98['NumberOfTime60-89DaysPastDueNotWorse'] > 90, 'NumberOfTime60-89DaysPastDueNotWorse'] = 18
    dfn98.loc[dfn98['NumberOfTimes90DaysLate'] > 90, 'NumberOfTimes90DaysLate'] = 18
    return dfn98