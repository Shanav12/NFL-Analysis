import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv("2021NFL.csv")
#Removing noise
df = df[(df["pass_att"] >= 15) | (df["rush_att"] >= 10)]
#Computing the completion percentage for each quarterback in each game
df["pass_cmp_percentage"] = df["pass_cmp"] / df["pass_att"]
#Normalzing the data
df["home_score"] = df["home_score"]  / df["home_score"].abs().max()
df["vis_score"] = df["vis_score"]  / df["vis_score"].abs().max()
df["pass_yds"] = df["pass_yds"]  / df["pass_yds"].abs().max()
df["pass_cmp_percentage"] = df["pass_cmp_percentage"]  / df["pass_cmp_percentage"].abs().max()
df["pass_td"] = df["pass_td"]  / df["pass_td"].abs().max()
df["pass_sacked"] = df["pass_sacked"]  / df["pass_sacked"].abs().max()
df["pass_rating"] = df["pass_rating"]  / df["pass_rating"].abs().max()
df["rush_att"] = df["rush_att"]  / df["rush_att"].abs().max()
df["rush_yds"] = df["rush_yds"]  / df["rush_yds"].abs().max()
df["rush_td"] = df["rush_td"]  / df["rush_td"].abs().max()
df["rush_yac"] = df["rush_yac"]  / df["rush_yac"].abs().max()
df["rush_broken_tackles"] = df["rush_broken_tackles"]  / df["rush_broken_tackles"].abs().max()
df["rush_yds_before_contact"] = df["rush_yds_before_contact"] / df["rush_yds_before_contact"].abs().max()
#Splitting into datasets that contain quarterback and running back data for home and away games
df_qb_home = df[(df.pos == "QB") & (df.team == df["home_team"])]
df_qb_away = df[(df.pos == "QB") & (df.team == df["vis_team"])]
df_rb_home = df[(df.pos == "RB") & (df.team == df["home_team"])]
df_rb_away = df[(df.pos == "RB") & (df.team == df["vis_team"])]
#Using passing metrics to predict the points a team scores at home
X_qb_home = df_qb_home[["pass_yds", "pass_cmp_percentage","pass_td","pass_sacked", "pass_rating"]]
Y_qb_home = df_qb_home[["home_score"]]
# Setting a constant random state to ensure the testing and training data is consistently split in the same manner
X_train_qb_home, X_test_qb_home, Y_train_qb_home, Y_test_qb_home = train_test_split(X_qb_home, Y_qb_home, test_size= 0.5, random_state = 42)
qb_home_model = LinearRegression()
qb_home_score_train = qb_home_model.fit(X_train_qb_home, Y_train_qb_home)
print("The coefficient of determination for the home scoring quarterback model is: " + str(round(qb_home_score_train.score(X_test_qb_home, Y_test_qb_home), 2)))
#Using passing metrics to predict the points a team scores on the road
X_qb_away = df_qb_away[["pass_yds", "pass_cmp_percentage", "pass_td", "pass_sacked", "pass_rating"]]
Y_qb_away = df_qb_away[["vis_score"]]
# Setting a constant random state to ensure the testing and training data is consistently split in the same manner
X_train_qb_away, X_test_qb_away, Y_train_qb_away, Y_test_qb_away = train_test_split(X_qb_away, Y_qb_away, test_size= 0.5, random_state = 42)
qb_away_model = LinearRegression()
qb_away_score_train = qb_away_model.fit(X_train_qb_away, Y_train_qb_away)
print("The coefficient of determination for the away scoring quarterback model is: " + str(round(qb_away_score_train.score(X_test_qb_away, Y_test_qb_away), 2)))
#Using rushing metrics to predict the points a team scores at home
X_home_rb = df_rb_home[["rush_yds","rush_td", "rush_yac", "rush_yds_before_contact", "rush_att"]]
Y_home_rb = df_rb_home[["home_score"]]
# Setting a constant random state to ensure the testing and training data is consistently split in the same manner
X_train_rb_home, X_test_rb_home, Y_train_rb_home, Y_test_rb_home = train_test_split(X_home_rb, Y_home_rb, test_size= 0.5, random_state = 42)
rb_home_model = LinearRegression()
rb_home_score_train = rb_home_model.fit(X_train_rb_home, Y_train_rb_home)
print("The coefficient of determination for the home scoring running back model is: " + str(round(rb_home_score_train.score(X_test_rb_home, Y_test_rb_home), 2)))
#Using rushing metrics to predict the points a team scores on the road
X_away_rb = df_rb_away[["rush_yds", "rush_td", "rush_yac", "rush_yds_before_contact", "rush_att"]]
Y_away_rb = df_rb_away[["vis_score"]]
# Setting a constant random state to ensure the testing and training data is consistently split in the same manner
X_train_rb_away, X_test_rb_away, Y_train_rb_away, Y_test_rb_away = train_test_split(X_away_rb, Y_away_rb, test_size= 0.5, random_state = 42)
rb_away_model = LinearRegression()
rb_away_score_train = rb_away_model.fit(X_train_rb_away, Y_train_rb_away)
print("The coefficient of determination for the away scoring running back model is: " + str(round(rb_away_score_train.score(X_test_rb_away, Y_test_rb_away), 2)))