# import necessary modules
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Step 1: Load data stored in csv files and cache them
@st.cache
def load_data():
    # load data to dataframes
    df_goalkeeping_value = pd.read_csv('df_goalkeeper_value.csv')
    df_defense_value = pd.read_csv('df_defender_value.csv')
    df_midfield_value = pd.read_csv('df_midfielder_value.csv')
    df_attack_value = pd.read_csv('df_attack_value2.csv')
    return df_attack_value, df_goalkeeping_value, df_defense_value, df_midfield_value


# add page title
st.title('What affects the market value of a football player')
# add subtitle and author name
st.markdown('**La Liga players as an example (seasons 2010-11 to 2021-22)**')
st.markdown('by Zakaria Chbani')
# insert a horizontal line
st.markdown('---')

# Steps 2 and 3: Choose a player's position group and relevant explanatory variables
position = st.selectbox("Choose the player's position", ["Attackers", "Goalkeepers", "Defenders", "Midfielders"])
if position == "Attackers":
    df = load_data()[0]
elif position == "Goalkeepers":
    df = load_data()[1]
elif position == "Defenders":
    df = load_data()[2]
else:
    df = load_data()[3]
# relevant explanatory variables
df_explanatory = df.drop(['market_value_next_season', 'player_id', 'season_label'], axis=1)
columns = st.multiselect("Choose the explanatory variables", df_explanatory.columns)

# Step 4: Choose the regression algorithm
algorithm = st.selectbox("Choose the regression algorithm", ["Linear Regression", "Gradient Boosting", "Random Forest"])
if algorithm == "Linear Regression":
    regressor = LinearRegression()
elif algorithm == "Gradient Boosting":
    regressor = GradientBoostingRegressor()
else:
    regressor = RandomForestRegressor()

# Step 5: Create a pipeline and run the model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", regressor)])
# add a submit button that will run the model
if st.button('Submit'):
    # drop rows that have NaN values in market_value_next_season
    df = df.dropna(subset=['market_value_next_season'])
    # define training and test sets
    X = df[columns]
    y = df["market_value_next_season"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # fit the model
    pipeline.fit(X_train, y_train)

    # Step 6: Evaluate the model
    y_pred = pipeline.predict(X_test)
    st.write("")
    # show evaluation metrics
    st.markdown('**Evaluation metrics:**')
    st.write(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    st.write("")
    # print out the coefficients with their variable names
    st.markdown('**Coefficients:**')
    for i in range(len(columns)):
        # if the chosen algorithm is linear regression, print out the coefficients
        if algorithm == "Linear Regression":
            st.write(f"{columns[i]}: {pipeline.named_steps['regressor'].coef_[i]:.2f}")
        # if the chosen algorithm is gradient boosting, or random forest, print out the feature importances
        else:
            st.write(f"{columns[i]}: {pipeline.named_steps['regressor'].feature_importances_[i]:.2f}")

    # plot the predicted values against the actual values
    st.write("")
    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    st.pyplot(fig)
    # plot subplots of the dependent variable against each independent variable
    st.write("")
    st.markdown('**Dependent variable against each independent variable:**')
    fig2, ax2 = plt.subplots(len(columns), 1, figsize=(10, 5*len(columns)))
    for i in range(len(columns)):
        ax2[i].scatter(X[columns[i]], y, alpha=0.5)
        ax2[i].set_xlabel(columns[i])
        ax2[i].set_ylabel('market value in the following season')
    st.pyplot(fig2)

# hide page footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Step 6: Make the predictions
# st.subheader("Make prediction")
# show input fields for each explanatory variable
# input_fields = []
# for column in columns:
#     input_field = st.number_input(label=column)
#     input_fields.append(input_field)
# convert input fields to numpy array
# input_fields = np.array(input_fields).reshape(1, -1)
# add predict button that will calculate prediction
# if st.button('Predict'):
#     df2 = df.dropna(subset=['market_value_next_season'])
#     X = df2[columns]
#     y = df2["market_value_next_season"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     pipeline.fit(X_train, y_train)
#     prediction = pipeline.predict(input_fields)
#     st.write(f"Market value next season: {prediction[0]:.2f}")
