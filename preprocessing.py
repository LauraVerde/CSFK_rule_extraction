import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import category_encoders as ce  # Target Encoding converts categorical variables to numeric ones based on the relationship with the target
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def detect_and_convert_to_boolean(df):
    # Automatically detect columns that can be converted to boolean and handle various formats
    for col in df.columns:
        unique_values = df[col].dropna().unique()
        possible_boolean_values = {'si', 'no', True, False, 1, 0, 'True', 'False', '1', '0'}
        if set(unique_values).issubset(possible_boolean_values):
            df[col] = df[col].map({
                'si': True, 'no': False,
                'True': True, 'False': False,
                '1': True, '0': False,
                True: True, False: False,
                1: True, 0: False
            })
            df[col] = df[col].astype('bool')
        return df
    return None

def fill_missing_values(df):
    for col in df.columns:
        if df[col].dtype == 'object' or 'bool':
            mode_value = df[col].mode()[0]
            df[col] = df[col].replace(np.nan, mode_value)
        else:
            mean_value = df[col].mean()
            df[col] = df[col].replace(np.nan, mean_value)
    return df

def plot_barplot(df_top, score_column, title, xlabel, ylabel, percentile, filename=None, save_plot=False):
    sns.barplot(x=score_column, y='Feature', data=df_top)
    for index, value in enumerate(df_top[score_column]):
        plt.text(value, index, f'{value:.2f}', va='center', ha='left', color='black')
    plt.title(f'{title} ({percentile}° Percentile)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_plot and filename:
        plt.savefig(f'{filename}_{percentile}.png')
        print(f'Chart saved as: {filename}_{percentile}.png')
        plt.close()
    else:
        plt.show()

def grafico_caratteristiche_selezionate(df, score_column, percentiles, top_n, title, xlabel, ylabel, save_plot=True,
                                        filename=None, csv_filename='selected_features.csv'):
    df_features = pd.DataFrame()
    df_bd = pd.DataFrame()
    for i, percentile in enumerate(percentiles):
        threshold = np.percentile(df[score_column], percentile)
        df_filtered = df[df[score_column] > threshold].copy()
        df_top = df_filtered.head(top_n)
        plot_barplot(df_top, score_column, title, xlabel, ylabel, percentile, filename, save_plot)
        df_features[f'Percentile_{percentiles[i]}'] = pd.Series(df_top['Feature'].values)
        df_bd[f'Percentile_{percentiles[i]}_Features'] = pd.Series(df_top['Feature'].values)
        df_bd[f'Percentile_{percentiles[i]}_Scores'] = pd.Series(df_top[score_column].values)
    df_features.to_csv(csv_filename, index=False)
    print(f"CSV file saved as '{csv_filename}'")
    return df_bd, df_features

def funzione_boxplot(df, score_column, title, xlabel, ylabel, save_plot=True, filename=None):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[score_column])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_plot and filename:
        plt.savefig(filename)
        print(f'Chart saved as: {filename}')
    plt.show()
    plt.close()

def merge_dataframes(mi, fi):
    combined_df = pd.merge(mi, fi, on='Percentile_25_Features', how='inner').fillna(0)
    return combined_df

def compare_feature_scores_chart(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(df))
    ax.bar(index, df['Percentile_25_Scores_x'], bar_width, label='Mutual Information')
    ax.bar(index + bar_width, df['Percentile_25_Scores_y'], bar_width, label='Feature Importance')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Score')
    ax.set_title('Comparison between MI and FI for Features')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(df['Percentile_25_Features'], rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.show()

def smote_augmentation(x, y):
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X, Y = smote.fit_resample(x, y)
    return X, Y

def data_econder(x, y):
    encoder = ce.TargetEncoder(cols=x.select_dtypes(include=['object']).columns)
    X = encoder.fit_transform(x, y)
    return X

def MI_feature(x, y):
    mi_scores = mutual_info_classif(x, y, random_state=0)
    return mi_scores

def FI_feature(x, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, Y_train)
    importances = model.feature_importances_
    return importances
