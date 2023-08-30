import pandas as pd
import seaborn as sns
import streamlit as st
# Load data function (replace with your actual data loading)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    ames_train = pd.read_csv("C:/Users/Shetakshi/Downloads/ames_train.csv")
    ames_test = pd.read_csv("C:/Users/Shetakshi/Downloads/ames_test.csv")
    return ames_train, ames_test

ames_train, ames_test=load_data()

st.set_option('deprecation.showPyplotGlobalUse', False)

# Display data properties
def display_data_properties(data):
    st.subheader("Data Properties")
    st.write("Shape:", data.shape)
    st.write("Columns:", data.columns.tolist())
    st.write("Data Types:")
    st.write(data.dtypes)


# Display compact features
def display_compact_features(data, num_rows):
    st.subheader(f'Compact Table: First {num_rows} rows')
    st.write(data.head(num_rows))


# Display column features
def display_column_features(data):
    selected_column = st.selectbox('Select a Column:', data.columns)
    column_type = "Categorical" if data[selected_column].dtype == 'object' else "Numerical"
    unique_values = data[selected_column].nunique()

    st.subheader('Column Details')
    st.write(f'Column Name: {selected_column}')
    st.write(f'Column Type: {column_type}')
    st.write(f'Unique Values: {unique_values}')


# Display visualizations
def display_visualizations(data):
    st.subheader("Visualizations")
    visualization_option = st.sidebar.selectbox("Select Visualization Type:", ["Univariate", "Bivariate"])

    if visualization_option == "Univariate":
        selected_feature = st.sidebar.selectbox('Select a Feature:', data.columns)
        if data[selected_feature].dtype == 'object':
            display_univariate_categorical(data, selected_feature)
        else:
            display_univariate_numeric(data, selected_feature)
    elif visualization_option == "Bivariate":
        feature1 = st.sidebar.selectbox('Select the First Feature:', data.columns)
        feature2 = st.sidebar.selectbox('Select the Second Feature:', data.columns)
        if data[feature1].dtype == 'object' and data[feature2].dtype == 'object':
            display_bivariate_categorical(data, feature1, feature2)
        elif data[feature1].dtype != 'object' and data[feature2].dtype != 'object':
            display_bivariate_numeric(data, feature1, feature2)
        else:
            display_default_plot()


def display_univariate_categorical(data, selected_feature):
    st.subheader(f'Bar Chart: {selected_feature}')
    sns.countplot(data=data, x=selected_feature)
    plt.xticks(rotation=45)
    st.pyplot()


def display_univariate_numeric(data, selected_feature):
    st.subheader(f'Histogram: {selected_feature}')
    sns.histplot(data=data, x=selected_feature, bins=20, kde=True)
    st.pyplot()


def display_bivariate_categorical(data, feature1, feature2):
    st.subheader(f'Stacked Bar Chart: {feature1} vs {feature2}')
    crosstab = pd.crosstab(data[feature1], data[feature2])
    crosstab.plot(kind='bar', stacked=True)
    plt.xticks(rotation=45)
    st.pyplot()


def display_bivariate_numeric(data, feature1, feature2):
    st.subheader(f'Scatter Plot: {feature1} vs {feature2}')
    sns.scatterplot(data=data, x=feature1, y=feature2)
    st.pyplot()


def display_default_plot():
    st.subheader('Default Plot')
    st.write('Please select one categorical and one numeric feature for this option.')


# Display missing columns
def display_missing_columns(dataframe):
    missing_values_count = dataframe.isnull().sum()
    missing_columns = missing_values_count[missing_values_count > 0]
    if not missing_columns.empty:
        st.write("Columns with missing values:")
        st.write(missing_columns)
    else:
        st.write("No missing values found.")

    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')




def display_missing_values_table(data):
    # Calculate the missing value count and percentage for each column
    missing_values_count = data.isnull().sum()
    missing_values_percentage = (missing_values_count / len(data)) * 100

    # Create a DataFrame with the columns that have missing values and their percentages
    missing_values_table = pd.DataFrame({
        'Column Name': missing_values_count[missing_values_count > 0].index,
        'Missing Value Percentage': missing_values_percentage[missing_values_count > 0].values,
        'Missing Value Count': missing_values_count[missing_values_count > 0].values
    })

    # Sorting the table by missing value percentage in descending order
    #missing_values_table = missing_values_table.sort_values(by='Missing Value Percentage', ascending=False)

    # Display the table
    #print(missing_values_table)


def drop_columns_with_missing_values(data, threshold):
    # Calculate the missing value percentage for each column
    missing_values_percentage = (data.isnull().sum() / len(data)) * 100

    # Sort the columns in descending order based on missing values percentage
    missing_values_sorted = missing_values_percentage.sort_values(ascending=False)

    # Identify columns with more than the specified threshold of missing values
    columns_to_drop = missing_values_sorted[missing_values_sorted > threshold].index

    # Drop the identified columns from the DataFrame
    data_dropped = data.drop(columns=columns_to_drop)

    # Display the remaining columns
    print("Remaining columns:")
    print(data_dropped.columns)

    # Display the updated number of columns
    print("Number of remaining columns:", len(data_dropped.columns))




def impute_missing_values(train_df, test_df):
    # Select columns with missing values in train dataset
    train_columns_with_missing = train_df.columns[train_df.isnull().any()]

    # Select columns with missing values in test dataset
    test_columns_with_missing = test_df.columns[test_df.isnull().any()]

    # Identify columns with missing values in both train and test datasets
    columns_with_missing = set(train_columns_with_missing).union(test_columns_with_missing)

    # Separate categorical and numerical columns
    categorical_columns = train_df.select_dtypes(include='object').columns.tolist()
    numerical_columns = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Exclude target variable column (SalePrice) from imputation
    numerical_columns = [col for col in numerical_columns if col != 'SalePrice']

    # Impute missing values in categorical columns
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    train_df[categorical_columns] = categorical_imputer.fit_transform(train_df[categorical_columns])
    test_df[categorical_columns] = categorical_imputer.transform(test_df[categorical_columns])

    # Impute missing values in numerical columns
    numerical_imputer = SimpleImputer(strategy='median')
    train_df[numerical_columns] = numerical_imputer.fit_transform(train_df[numerical_columns])
    test_df[numerical_columns] = numerical_imputer.transform(test_df[numerical_columns])

    return train_df, test_df


# Impute missing values in train and test datasets
#train_data_imputed, test_data_imputed = impute_missing_values(ames_train, ames_test)




numerical_features = ['OverallQual', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
                      'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
                      'GarageArea', 'WoodDeckSF', 'OpenPorchSF']


def scale_numerical_features(train_data, test_data, numerical_features):
    scaler = StandardScaler()
    train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])
    test_data[numerical_features] = scaler.transform(test_data[numerical_features])
    return train_data, test_data


#scale_numerical_features(ames_train, ames_test, numerical_features)


def calculate_scaled_stats(data, numerical_features):
    numerical_mean = data[numerical_features].mean()
    numerical_std = data[numerical_features].std()
    st.subheader("Mean and Standard Deviation (Before Scaling)")
    st.write(numerical_mean)
    st.write(numerical_std)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numerical_features])
    scaled_mean = scaled_data.mean(axis=0)
    scaled_std = scaled_data.std(axis=0)
    st.subheader("Mean and Standard Deviation (After Scaling)")
    st.write(scaled_mean)
    st.write(scaled_std)


#calculate_scaled_stats(ames_train, numerical_features)


# Define the function to perform PCA
def update_dataframe_with_pca(data):
    categorical_columns = data.select_dtypes(include=['object'])
    encoded_data = pd.get_dummies(categorical_columns)
    numerical_columns = data.select_dtypes(include=['float64', 'int64'])
    combined_data = pd.concat([numerical_columns, encoded_data], axis=1)

    pca = PCA(n_components=10)
    reduced_data = pca.fit_transform(combined_data)
    reduced_columns = [f"PC{i}" for i in range(1, reduced_data.shape[1] + 1)]
    combined_reduced_data = pd.DataFrame(reduced_data, columns=reduced_columns)
    updated_data = pd.concat([numerical_columns, combined_reduced_data], axis=1)

    return updated_data


def main():
    st.title("Data Analysis App")

    ames_train,ames_test= load_data()

    st.sidebar.title("Options")
    selected_section = st.sidebar.radio("Select Section:", ["Data Analysis", "Visualization"])

    if selected_section == "Data Analysis":
        st.subheader("Data Analysis")

        show_data_properties = st.sidebar.checkbox("Data Properties")
        show_column_features = st.sidebar.checkbox("Column")
        show_details = st.sidebar.checkbox("Details")
        num_rows = st.sidebar.number_input("Number of Rows to Display", value=5)

        if show_data_properties:
            display_data_properties(ames_train)

        if show_column_features:
            display_column_features(ames_train)

        if show_details:
            st.write("Selected Columns:")
            selected_columns = st.multiselect("Select columns", ames_train.columns.tolist())
            if selected_columns:
                st.write([selected_columns].describe())

        if num_rows:
            display_compact_features(ames_train, num_rows)

        if st.sidebar.checkbox("Data Preprocessing"):
            st.subheader("Data Preprocessing")

            if st.checkbox("Display Missing Values"):
                display_missing_values_table()

            if st.checkbox("Impute Missing Values"):
                # Perform imputation
                train_data_imputed, test_data_imputed = impute_missing_values(ames_train,ames_test)
                print(train_data_imputed.isnull().sum())
                print(test_data_imputed.isnull().sum())

                # Verify if there are no more missing values
                st.write("Missing Value Counts After Imputation:")
                st.write(train_data_imputed.isnull().sum())
                st.write(test_data_imputed.isnull().sum())

            if st.checkbox("Scale Numerical Features"):
                scale_numerical_features(ames_train,ames_test , numerical_features)
                st.write("Numerical features scaled.")

            if st.checkbox("calculate scaled stats"):
                calculate_scaled_stats(ames_train, ames_test,numerical_features)
                st.write("calculate scaled stats")


    elif selected_section == "Visualization":
        st.subheader("Visualization")
        # Add your visualization code here
        display_visualizations(ames_train,ames_test)

if __name__ == "__main__":
    main()