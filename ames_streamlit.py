from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, ylabel, legend, xlabel, title
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os


# Load the Ames Housing datasets
#@st._cache(allow_output_mutation=True)
#def load_data():
 #   ames_train = pd.read_csv("C:/Users/Shetakshi/Downloads/ames_train.csv")
  #  ames_test = pd.read_csv("C:/Users/Shetakshi/Downloads/ames_test.csv")
   # return ames_train, ames_test


def load_data():
    ames_train = pd.read_csv("ames_train.csv")
    ames_test = pd.read_csv("ames_test.csv")
    return ames_train, ames_test


st.set_option('deprecation.showPyplotGlobalUse', False)


def main():
    st.title("Ames Housing Data Analysis")

    ames_train, ames_test = load_data()

    # Sidebar for user inputs
    st.sidebar.header('User Inputs')
    num_rows = st.sidebar.number_input('Number of Rows to Display', min_value=1,
                                       max_value=max(len(ames_train), len(ames_test)), value=5)

    # Data Preview Section
    st.header('Data Preview')
    st.subheader(f'Train Data: First {num_rows} rows')
    st.write(ames_train.head(num_rows))
    st.subheader(f'Test Data: First {num_rows} rows')
    st.write(ames_test.head(num_rows))

    # Check Duplicated Values Section
    st.header('Check Duplicated Values')
    duplicated_rows_train = ames_train[ames_train.duplicated()]
    st.subheader("Duplicated Rows in Train Data")
    st.write(duplicated_rows_train)

    duplicated_rows_test = ames_test[ames_test.duplicated()]
    st.subheader("Duplicated Rows in Test Data")
    st.write(duplicated_rows_test)

    # Dataset Statistics Section
    st.header('Dataset Statistics')
    train_stats = ames_train.describe()
    st.subheader("Train Data Statistics")
    st.write(train_stats)

    test_stats = ames_test.describe()
    st.subheader("Test Data Statistics")
    st.write(test_stats)

    def display_missing_columns(dataframe):
        pd.set_option('display.max_rows', None)  # display all rows
        pd.set_option('display.max_columns', None)  # display all columns
        missing_values_count = dataframe.isnull().sum()  # displays only the columns which have missing values
        missing_columns = missing_values_count[
            missing_values_count > 0]  # Display the columns with missing values only.
        if not missing_columns.empty:
            print("Columns with missing values:")
            print(missing_columns)
        else:
            print("No missing values found.")
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')

    display_missing_columns(ames_train)
    display_missing_columns(ames_test)

    missing_values_count = ames_train.isnull().sum()
    missing_values_count_test = ames_test.isnull().sum()

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
        missing_values_table = missing_values_table.sort_values(by='Missing Value Percentage', ascending=False)

        # Display the table
        print(missing_values_table)

    display_missing_values_table(ames_train)
    display_missing_values_table(ames_test)

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

        # Example usage

    drop_columns_with_missing_values(ames_train, 15)
    drop_columns_with_missing_values(ames_test, 15)

    # Sidebar to select variables and scatter plot parameters
    st.sidebar.title("Scatter Plot Parameters")
    x_var = st.sidebar.selectbox("Select X Variable", ames_train.columns)
    y_var = st.sidebar.selectbox("Select Y Variable", ames_train.columns, index=1)
    hue_var = st.sidebar.selectbox("Select Hue Variable (Optional)", ames_train.columns, index=0)
    size_var = st.sidebar.selectbox("Select Size Variable (Optional)", ames_train.columns, index=0)
    alpha = st.sidebar.slider("Select Transparency (Alpha)", 0.1, 1.0, 0.5)
    marker = st.sidebar.selectbox("Select Marker Style", ['o', 's', 'D', '^', 'v'])

    # Scatter plot
    st.title("Scatter Plot")
    st.write(f"Scatter Plot of {x_var} vs {y_var}")

    # Create scatter plot using Seaborn
    figure(figsize=(10, 6))
    sns.scatterplot(data=ames_train, x=x_var, y=y_var, hue=hue_var, size=size_var, alpha=alpha, marker=marker)
    xlabel(x_var)
    ylabel(y_var)
    title(f"{x_var} vs {y_var}")
    legend()
    st.pyplot()

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

    train_data = ames_train
    test_data = ames_test

    # Impute missing values in train and test datasets
    train_data_imputed, test_data_imputed = impute_missing_values(train_data, test_data)

    # Verify if there are no more missing values
    print(train_data_imputed.isnull().sum())
    print(test_data_imputed.isnull().sum())
    # Streamlit app
    st.title("Missing Value Imputation")

    # Display some info about the app
    st.write("This app allows you to perform missing value imputation on your data.")

    # Button to start imputation
    if st.button("Impute Missing Values"):
        st.subheader("Imputation Results")

        # Perform imputation
        train_data_imputed, test_data_imputed = impute_missing_values(train_data, test_data)

        # Verify if there are no more missing values
        st.write("Missing Value Counts After Imputation:")
        st.write("Train Data:")
        st.write(train_data_imputed.isnull().sum())
        st.write("Test Data:")
        st.write(test_data_imputed.isnull().sum())

    numerical_features = ['OverallQual', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
                          'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
                          'GarageArea', 'WoodDeckSF', 'OpenPorchSF']

    def scale_numerical_features(train_data, test_data, numerical_features):
        scaler = StandardScaler()
        train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])
        test_data[numerical_features] = scaler.transform(test_data[numerical_features])
        return train_data, test_data

    scale_numerical_features(ames_train, ames_test, numerical_features)

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

    calculate_scaled_stats(ames_train, numerical_features)

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

    # Streamlit app
    st.title("Data Preprocessing and Analysis")

    # Display scaled statistics
    numerical_features = ['OverallQual', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
                          'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
                          'GarageArea', 'WoodDeckSF', 'OpenPorchSF']
    st.header("Scaled Statistics")
    st.subheader("Train Data")
    st.write("Numerical Features: ", numerical_features)
    calculate_scaled_stats(ames_train, numerical_features)
    st.subheader("Test Data")
    st.write("Numerical Features: ", numerical_features)
    calculate_scaled_stats(ames_test, numerical_features)

    # Perform PCA
    st.header("Perform PCA")
    st.subheader("Original Shape of ames_train")
    st.write(ames_train.shape)
    ames_train_pca = update_dataframe_with_pca(ames_train)
    st.subheader("Shape of ames_train after PCA")
    st.write(ames_train_pca.shape)

    def display_correlation_heatmap(data_frame):
        correlation_matrix = data_frame.corr()

        # Create a heatmap using Seaborn
        figure(figsize=(30, 30))
        sns.heatmap(correlation_matrix, cmap="Greens", annot=True, linewidth=.10, square=True)
        title("Correlation Heatmap")

        # Display the heatmap using Streamlit's st.pyplot()
        st.pyplot()

    # Assuming you have 'ames_train' DataFrame
    if st.button('Display Correlation Heatmap'):
        display_correlation_heatmap(ames_train)

    def filter_features_by_correlation(data_train, data_test, target_column, correlation_threshold):
        # Calculating the correlation matrix
        correlation_matrix = data_train.corr()

        # Getting the correlation values with the target variable
        correlation_with_target = correlation_matrix[target_column].abs()

        # Identifying the features with correlation less than the threshold
        less_correlated_features = correlation_with_target[correlation_with_target < correlation_threshold].index

        # Dropping the less correlated features from the train dataset
        data_train_filtered = data_train.drop(less_correlated_features, axis=1)

        # Dropping the same features from the test dataset
        data_test_filtered = data_test.drop(less_correlated_features, axis=1)

        # Getting the count of remaining columns
        column_count_train = data_train_filtered.shape[1]
        column_count_test = data_test_filtered.shape[1]

        # Displaying the count and remaining columns using Streamlit
        st.write("Number of remaining columns in train dataset:", column_count_train)
        st.write("Number of remaining columns in test dataset:", column_count_test)

        st.write("Remaining columns in train dataset:")
        st.write(data_train_filtered.columns.tolist())
        st.write("Remaining columns in test dataset:")
        st.write(data_test_filtered.columns.tolist())

        # Updating the original train and test datasets with the filtered columns
        return data_train_filtered, data_test_filtered

    # Assuming 'ames_train' and 'ames_test' are DataFrames with mixed data types
    # Drop non-numeric columns and encode categorical columns
    numeric_train = ames_train.select_dtypes(include=['number'])
    numeric_test = ames_test.select_dtypes(include=['number'])

    # Filter features by correlation on the numeric datasets
    ames_train_filtered, ames_test_filtered = filter_features_by_correlation(numeric_train, numeric_test, 'SalePrice',
                                                                             0.3)

    def detect_outliers_boxplot(df):
        # Select numerical columns
        numerical_cols = df.select_dtypes(include='number').columns

        # Create a boxplot for each numerical column
        for col in numerical_cols:
            st.subheader(col)
            fig, ax = plt.subplots()  # Create a new figure and axis for each plot
            ax.boxplot(df[col].dropna())
            ax.set_title(col)

            # Convert the Matplotlib figure to an image
            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            st.image(buffer, use_column_width=True)

            # Calculate the upper and lower bounds for outliers
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Detect upper bound and lower bound outliers
            upper_outliers = df[df[col] > upper_bound][col]
            lower_outliers = df[df[col] < lower_bound][col]

            st.write(f"Column: {col}")
            st.write("Upper Bound Outliers:")
            st.write(upper_outliers)
            st.write("\n")

            st.write("Lower Bound Outliers:")
            st.write(lower_outliers)
            st.write("\n")

    detect_outliers_boxplot(ames_train)

    detect_outliers_boxplot(ames_test)

    # Histogram Analysis Section
    st.header('Histogram Analysis')

    # Sidebar options for histogram analysis
    st.sidebar.subheader('Histogram Options')
    selected_column = st.sidebar.selectbox('Select a Column for Histogram Analysis', ames_train.columns)
    num_bins = st.sidebar.slider('Number of Bins', min_value=5, max_value=50, value=10)
    kde = st.sidebar.checkbox('Kernel Density Estimation', value=True)

    # Display histogram using Seaborn
    plt.figure(figsize=(10, 6))
    sns.histplot(data=ames_train[selected_column], bins=num_bins, kde=kde)
    plt.title(f'Histogram of {selected_column}')
    plt.xlabel(selected_column)
    plt.ylabel('Frequency')
    st.pyplot()

    # Plot histograms, check normality, and detect outliers
    if st.button("Plot Histograms and Check Data"):
        plot_histograms(ames_train, selected_column, num_bins, kde)

    # ... (Same as before)


def plot_histograms(df, selected_column, num_bins, kde):
    data = df[selected_column].dropna()

    # Plot histogram
    plt.figure()
    sns.histplot(data=data, bins=num_bins, kde=kde)
    plt.title(selected_column)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    st.pyplot()  # Display plot in Streamlit

    # Check if normally distributed
    is_normal = np.abs(data.skew()) < 0.5
    st.write("Is normally distributed:", is_normal)

    # Check for outliers
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    has_outliers = (data < lower_bound) | (data > upper_bound)
    st.write("Has outliers:", has_outliers.any())
    st.write()


# ... (Same as before)

if __name__ == "__main__":
    main()
