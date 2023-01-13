# Player's Market Value Prediction with a Streamlit Web App
In this simple project, we created and deployed a simple web app that allows playing with variables representing players' statistics to determine the most relevant ones in predicting the market value of a player in the upcoming season.

Data is specific to Spanish La Liga for seasons 2010-11 to 2021-22.

The current repository includes four files containing relevant statistics for goalkeepers, defenders, midfielders, and attackers. We load these files to data frames that serve as inputs to the app.
The app works as follows:
1. The user chooses a player's position. Following the position the user chooses, one of the four data frames is considered.
2. The user chooses from the corresponding explanatory variables the ones he wants to include in the model.
3. The user chooses a regression algorithm to apply (Linear regression, Gradient Boosting, or Random Forest).
4. The algorithm runs with the chosen variables.
5. Evaluation metrics are then calculated and shown, and plots showing the relationship between the dependent and independent variables are shown below.

Please [click here](https://breathingdata-regression-player-market-va-regression-app-m2uucj.streamlit.app/) to open the app.
