import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

# Define global variables for user-uploaded data and trained model
uploaded_data = None
trained_model = None

# Define a function to load data from user-uploaded CSV files
@st.cache(allow_output_mutation=True)
def load_data(uploaded_files):
    data_frames = [pd.read_csv(file) for file in uploaded_files]
    return data_frames

# Streamlit UI
st.title("Custom Data Analysis and Regression App")

# File Upload
st.sidebar.header("Upload CSV Files")
uploaded_files = st.sidebar.file_uploader("Upload your CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    # Load the uploaded data
    uploaded_data = load_data(uploaded_files)

    # Display data properties
    st.sidebar.subheader("Data Properties")
    st.sidebar.write("Number of Uploaded Files:", len(uploaded_data))
    st.sidebar.write("File Names:", [file.name for file in uploaded_files])

    # Data Analysis Options
    data_analysis_option = st.sidebar.checkbox("Data Analysis")

    if data_analysis_option:
        st.sidebar.subheader("Data Analysis Options")
        show_data_properties = st.sidebar.checkbox("Data Properties")
        show_column_features = st.sidebar.checkbox("Columns")
        show_details = st.sidebar.checkbox("Details")
        num_rows = st.sidebar.number_input("Number of Rows to Display", value=5)

        # Display data properties
        if show_data_properties:
            for i, data_frame in enumerate(uploaded_data):
                st.write(f"**File {i + 1} Data Properties**")
                st.write("Shape:", data_frame.shape)
                st.write("Columns:", data_frame.columns.tolist())
                st.write("Data Types:")
                st.write(data_frame.dtypes)

        # Display column features
        if show_column_features:
            selected_column = st.sidebar.selectbox('Select a Column:', data_frame.columns.tolist())
            if selected_column:
                st.sidebar.write(f"**{selected_column} Features**")
                missing_percentage = data_frame[selected_column].isnull().mean() * 100
                valid_percentage = data_frame[selected_column].notnull().mean() * 100
                mismatched_percentage = (data_frame[selected_column] == 'invalid').mean() * 100

                st.sidebar.write(
                    f"Missing: {missing_percentage:.2f}% | "
                    f"Valid: {valid_percentage:.2f}% | "
                    f"Mismatched: {mismatched_percentage:.2f}%"
                )

        # Display data details
        if show_details:
            st.write("Selected Columns:")
            selected_columns = st.sidebar.multiselect("Select columns", data_frame.columns.tolist())
            if selected_columns:
                for column in selected_columns:
                    st.write(f"**Descriptive Statistics for {column}**")
                    st.write(data_frame[column].describe())

        # Display compact features
        if num_rows:
            compact_checkbox = st.sidebar.checkbox("Compact")
            selected_column = None

            if compact_checkbox:
                selected_column = st.sidebar.selectbox("Select a Column:", data_frame.columns.tolist())
                if selected_column:
                    st.sidebar.write(f"**{selected_column}**")
                    missing_percentage = data_frame[selected_column].isnull().mean() * 100
                    valid_percentage = data_frame[selected_column].notnull().mean() * 100
                    mismatched_percentage = (data_frame[selected_column] == 'invalid').mean() * 100

                    st.sidebar.write(
                        f"Missing: {missing_percentage:.2f}% | "
                        f"Valid: {valid_percentage:.2f}% | "
                        f"Mismatched: {mismatched_percentage:.2f}%"
                    )

    # Regression Model Options
    regression_option = st.sidebar.checkbox("Regression Model")

    if regression_option:
        st.sidebar.subheader("Regression Model Options")

        # Data Splitting
        test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.sidebar.number_input("Random State", min_value=0, value=42)

        # Select Target Variable
        target_variable = st.sidebar.selectbox("Select Target Variable:", data_frame.columns.tolist())

        # Select Regression Algorithm
        regression_algorithm = st.sidebar.selectbox("Select Regression Algorithm", ["Linear Regression", "Ridge", "Lasso"])

        # Model Training
        train_button = st.sidebar.button("Train Model")

        if train_button:
            st.sidebar.write("Training the model...")

            # Prepare the data
            X = data_frame.drop(columns=[target_variable])
            y = data_frame[target_variable]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Preprocessing
            numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()

            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            # Model Selection and Training
            if regression_algorithm == "Linear Regression":
                model = LinearRegression()
            elif regression_algorithm == "Ridge":
                model = Ridge()
            elif regression_algorithm == "Lasso":
                model = Lasso()

            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('model', model)])

            # Train the model
            pipeline.fit(X_train, y_train)

            # Make predictions
            y_pred = pipeline.predict(X_test)

            # Evaluate the model
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            st.sidebar.write(f"RMSE: {rmse:.2f}")

            # Store the trained model
            trained_model = pipeline

    # Visualization
    visualization_option = st.sidebar.checkbox("Visualization")

    if visualization_option:
        st.sidebar.subheader("Visualization Options")

        # Select Visualization Type
        visualization_type = st.sidebar.selectbox("Select Visualization Type", ["Histogram", "Scatter Plot"])

        if visualization_type == "Histogram":
            selected_column = st.sidebar.selectbox("Select a Column for Histogram", data_frame.columns.tolist())
            plt.figure()
            plt.hist(data_frame[selected_column], bins=10)
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            st.sidebar.pyplot(plt)

        elif visualization_type == "Scatter Plot":
            x_column = st.sidebar.selectbox("Select X-Axis Column for Scatter Plot", data_frame.columns.tolist())
            y_column = st.sidebar.selectbox("Select Y-Axis Column for Scatter Plot", data_frame.columns.tolist())
            plt.figure()
            plt.scatter(data_frame[x_column], data_frame[y_column])
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            st.sidebar.pyplot(plt)

    # Data Preprocessing Options
    data_preprocessing_option = st.sidebar.checkbox("Data Preprocessing")

    if data_preprocessing_option:
        st.sidebar.subheader("Data Preprocessing Options")
        display_missing_values = st.sidebar.checkbox("Display Missing Values")
        display_missing_percentage = st.sidebar.checkbox("Display Missing Values Percentage")
        display_missing_features = st.sidebar.checkbox("Display Features with Missing Values")
        impute_missing_values_option = st.sidebar.checkbox("Impute Missing Values")
        convert_categorical_to_numeric = st.sidebar.checkbox("Convert Categorical to Numeric")

        if display_missing_values:
            st.subheader("Missing Values")
            for i, data_frame in enumerate(uploaded_data):
                st.write(f"**File {i + 1} Missing Values**")
                missing_values_count = data_frame.isnull().sum()
                missing_columns = missing_values_count[missing_values_count > 0]
                if not missing_columns.empty:
                    st.write("Columns with missing values:")
                    st.write(missing_columns)
                else:
                    st.write("No missing values found.")

        if display_missing_percentage:
            st.subheader("Missing Values Percentage")
            for i, data_frame in enumerate(uploaded_data):
                st.write(f"**File {i + 1} Missing Values Percentage**")
                missing_values_count = data_frame.isnull().sum()
                missing_values_percentage = (missing_values_count / len(data_frame)) * 100
                missing_values_table = pd.DataFrame({
                    'Column Name': missing_values_count[missing_values_count > 0].index,
                    'Missing Value Percentage': missing_values_percentage[missing_values_count > 0].values,
                    'Missing Value Count': missing_values_count[missing_values_count > 0].values
                })
                missing_values_table = missing_values_table.sort_values(by='Missing Value Percentage', ascending=False)
                st.write(missing_values_table)

        if display_missing_features:
            st.subheader("Features with Missing Values")
            threshold = st.sidebar.slider("Missing Value Threshold (%)", min_value=0, max_value=100, value=30)
            for i, data_frame in enumerate(uploaded_data):
                st.write(f"**File {i + 1} Features with Missing Values (Threshold: {threshold}%)**")
                missing_values_percentage = (data_frame.isnull().sum() / len(data_frame)) * 100
                missing_features = missing_values_percentage[missing_values_percentage > threshold].index
                st.write(missing_features)

        if impute_missing_values_option:
            st.subheader("Impute Missing Values")
            for i, data_frame in enumerate(uploaded_data):
                st.write(f"**File {i + 1} Imputed Data**")
                for column in data_frame.columns:
                    if data_frame[column].dtype == 'object':
                        data_frame[column] = data_frame[column].fillna(data_frame[column].mode()[0])
                    else:
                        data_frame[column] = data_frame[column].fillna(data_frame[column].median())
                st.write(data_frame)

        if convert_categorical_to_numeric:
            st.subheader("Convert Categorical to Numeric")
            for i, data_frame in enumerate(uploaded_data):
                st.write(f"**File {i + 1} Converted Data**")
                label_encoder = LabelEncoder()
                for column in data_frame.columns:
                    if data_frame[column].dtype == 'object':
                        data_frame[column] = label_encoder.fit_transform(data_frame[column])
                st.write(data_frame)

# Display the main content
if uploaded_data is not None:
    st.subheader("User-Uploaded Data Preview")
    for i, data_frame in enumerate(uploaded_data):
        st.write(f"**File {i + 1} Data Preview**")
        if st.button(f"Show Data for File {i + 1}"):
            st.write(data_frame.head())

if trained_model is not None:
    st.subheader("Trained Model")
    st.write("The model has been trained.")

st.sidebar.write("Please upload your CSV files and select options.")
