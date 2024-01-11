#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 17:07:27 2021

@author: apoorvupadhye
"""

import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns


class MyLogisticModel:
    def fit(
        self, home_team_name, away_team_name, home_score, away_score, team_names=None
    ):
        """
        Fitting a logistic regression model with three classes.

        Parameters
        ----------
        home_team_name: list
            A list of home team name.

        away_team_name: list
            A list of away team name.

        home_score: list
            A list of home team score

        away_score: list
            A list of away team score

        team_names: list, optional
            A list of team names that contains all teams. Useful when new teams are not
            in home_team_name and away_team_name history. When None it default to all teams
            included in home_team_name and away_team_name.

        Attributes
        ----------

        team_encoding_: OneHotEncoder
            A sklearn OneHotEncoder class that contains the team encoding.

        model_: LogisticRegression
            A sklearn LogisticRegression model fitted to the data.

        """

        X,y = self.encodeData(home_team_name, away_team_name, home_score, away_score)
        model = LogisticRegression(
            penalty="l2", fit_intercept=False, multi_class="ovr", C=1
        )
        model.fit(X, y)
        self.model_ = model
    
    def encodeData(self, home_team_name, away_team_name, home_score, away_score, team_names=None):
        home_team_name, away_team_name, home_score, away_score = [
            np.array(x)
            for x in [home_team_name, away_team_name, home_score, away_score]
        ]
        if team_names is None:
            team_names = np.array(list(home_team_name) + list(away_team_name)).reshape(
                -1, 1
            )
        else:
            team_names = np.array(team_names).reshape(-1, 1)

        self.team_encoding_ = OneHotEncoder(sparse=False).fit(team_names)

        home_dummies = self.team_encoding_.transform(home_team_name.reshape(-1, 1))
        away_dummies = self.team_encoding_.transform(away_team_name.reshape(-1, 1))

        X = np.concatenate([home_dummies, away_dummies], 1)
        y = np.sign(home_score - away_score)
        return X,y
    
    def validate_inputs(self, probabilities,true_results):
        '''Check metrics inputs'''
        assert probabilities.shape[0]==true_results.shape[0],'The number of match prediction and result should be the same'
        assert np.alltrue(probabilities>0), 'probabilities should be larger than 0.'
        assert np.alltrue(probabilities<=1), 'Probabilities should be lower or equal to 1.'
        assert np.alltrue( abs(1-probabilities.sum(1))<1e-3), 'Probabilities should sum to 1.'
        assert np.alltrue(~probabilities.isin([np.inf, -np.inf,np.nan]))
        assert probabilities.shape[1]==3, 'You must provide probabilities for 1, 2 and X.'
        assert np.alltrue([x in ['1','2','X'] for x in probabilities.columns]),'Probabilities columns name have to be "1","2" and "X" '
        assert np.alltrue([x in ['1', '2', 'X'] for x in np.unique(true_results)])
        assert isinstance(probabilities,pd.DataFrame)
        assert isinstance(true_results,pd.Series)
    
    def plotTeamDiagrams(self, data):
        
        df1 = data[:4]
        df4 = data[16:20]

        df1.plot.scatter(x = 'Club', y = 'Points', s = 120
        ,title=' Points Graph For Teams going ahead in Champions League')
        plt.xticks(rotation = 45)
        
        df4.plot.scatter(x = 'Club', y = 'Points', s = 120
        ,title=' Points Graph For Teams to relegation zone')
        plt.xticks(rotation = 45)
        plt.show()
    def compute_1_into_2_log_loss(self,probabilities, true_results):
        '''
        Compute the log-loss for 1x2 football results.
        Parameters
        ----------
        probabilities:  dataframe that contains probabilities for 1, X ,and 2 results.
    
        true_results: Series that contains the true results, 1 X or 2.
        Returns
        -------
        The log-loss'''
        
        self.validate_inputs(probabilities,true_results)
        true_results_dum = pd.get_dummies(true_results)
        return (np.log(probabilities)*true_results_dum).sum(1).mean()
    

    def compute_1_into_2_hit_ratio(self, probabilities, true_results):
        '''
        Compute the hit-ratio for 1x2 football results.
        '''
        self.validate_inputs(probabilities, true_results)
        predicted_results = probabilities.idxmax(1)
        return (predicted_results==true_results).mean()
    
    def get_coef(self,model):

        #Setting the coefficients of three logistic models for each team.
        
        home_feature_names = [
            x.replace("x0", "home") for x in self.team_encoding_.get_feature_names()
        ]
        away_feature_names = [
            x.replace("x0", "away") for x in self.team_encoding_.get_feature_names()
        ]
        coeffs = pd.DataFrame(
            model.coef_,
            index=model.classes_,
            columns=[home_feature_names + away_feature_names],
        ).T
        return coeffs.rename(columns={-1: "away wins", 0: "draw", 1: "home wins"})

    #checking if team is encoded
    def check_teams(self, home_team_name, away_team_name):
        
        assert (
            home_team_name in self.team_encoding_.categories_[0]
        ), f"{home_team_name} is recognized. It was not in the training data."
        assert (
            away_team_name in self.team_encoding_.categories_[0]
        ), f"{away_team_name} is recognized. It was not in the training data."
        
    def encode_to_predict_winner(self, home_team_name, away_team_name,model):
        check_is_fitted(model)
        self.check_teams(home_team_name, away_team_name)
        home_dummies = self.team_encoding_.transform(
            np.array(home_team_name).reshape(-1, 1)
        )
        away_dummies = self.team_encoding_.transform(
            np.array(away_team_name).reshape(-1, 1)
        )
        X = np.concatenate([home_dummies, away_dummies], 1)
        return X
    # predicting the winner
    # name of the winning team or if not exists then draw
    def predict_winner(self, home_team_name, away_team_name):
        '''        pred = 0
        try:
        except:'''

        X = self.encode_to_predict_winner(home_team_name, away_team_name,self.model_)
        pred = self.model_.predict(X)
            
        
        if pred == 0:
            return "draw"
        if pred > 0:
            return str(home_team_name)
        else:
            return str(away_team_name)

    #predicting the probabilitie of draw and win
    def predict_proba(self, home_team_name, away_team_name):
    
        check_is_fitted(self.model_)
        self.check_teams(home_team_name, away_team_name)
        home_team_name = np.array(home_team_name)
        away_team_name = np.array(away_team_name)
        home_dummies = self.team_encoding_.transform(home_team_name.reshape(-1, 1))
        away_dummies = self.team_encoding_.transform(away_team_name.reshape(-1, 1))
        X = np.concatenate([home_dummies, away_dummies], 1)
        return pd.DataFrame(
            self.model_.predict_proba(X),
            index=["probability"],
            columns=self.model_.classes_,
        ).rename(columns={-1: f"{away_team_name}", 0: "draw", 1: f"{home_team_name}"})

    def visualizationHistoGram(self, df,title):
        data = df['Points']
        mean = data.mean()
        min_value = min(data)
        max_value = max(data)
        
        plt.title(title)
        plt.ylim(min_value - 10, max_value + 10)
        plt.scatter(x=df.index, y=data)
        plt.hlines(y=mean, xmin=0, xmax=len(data))
        plt.show()
        df.plot.hist(alpha=0.5, title='Histogram for ' + title);
        df.plot.hist(stacked=True, bins=20,title='Histogram for ' + title);

    def fitSVM(
        self, home_team_name, away_team_name, home_score, away_score, team_names=None
    ):

        X,y = self.encodeData(home_team_name, away_team_name, home_score, away_score)
        svm_clf = svm.SVC()
        svm_clf.fit(X, y)
        return svm_clf
    
    def predict_winner_svm(self, home_team_name, away_team_name,model):
        '''        pred = 0
        try:
        except:'''

        X = self.encode_to_predict_winner(home_team_name, away_team_name,model)
        pred = model.predict(X)
            
        return pred   
    
    #processing the output by using all the dataset 
    #that exists to get an outcome of all the matches in the season       
    def processOutput(self):
        currentTotalPoints = {'Arsenal':0,'Aston Villa':0,'Brighton and Hove Albion':0,'Burnley':0,'Chelsea': 0, 'Crystal Palace': 0, 'Everton':0,'Fulham':0, 'Leeds United':0,
                'Leicester City':0,'Liverpool':0,'Manchester City':0,'Manchester United':0,
                'Newcastle':0,'Sheffield United':0,'Southampton':0,
                'Tottenham Hotspur':0,'West Bromwich Albion':0,'West Ham United':0
                ,'Wolverhampton':0}

        weekWiseData = {'Arsenal':{},'Aston Villa':{},'Brighton and Hove Albion':{},'Burnley':{},'Chelsea': {}, 'Crystal Palace': {}, 'Everton':{},'Fulham':{}, 'Leeds United':{},
                'Leicester City':{},'Liverpool':{},'Manchester City':{},'Manchester United':{},
                'Newcastle':{},'Sheffield United':{},'Southampton':{},
                'Tottenham Hotspur':{},'West Bromwich Albion':{},'West Ham United':{}
                ,'Wolverhampton':{}}

        for home in currentTotalPoints:
            for away in currentTotalPoints:
                if home != away:
                   output = model.predict_winner(home,away)
                   if output == home:
                       weekWiseData[home][home+"_"+away] = 3
                       weekWiseData[away][home+"_"+away] = 0
                       currentTotalPoints[home] = currentTotalPoints[home] +3
                   elif output == 'draw':
                       weekWiseData[home][home+"_"+away] = 1
                       weekWiseData[away][home+"_"+away] = 1
                       currentTotalPoints[home] = currentTotalPoints[home] +1
                       currentTotalPoints[away] = currentTotalPoints[away] +1
                   else:
                        weekWiseData[home][home+"_"+away] = 0
                        weekWiseData[away][home+"_"+away] = 3
                        currentTotalPoints[away] = currentTotalPoints[away] +3
                else:   
                        print()
        return currentTotalPoints, weekWiseData
    
input_dir = "/Users/apoorvupadhye/Documents/BU Assignment/Fall 21/Artificial Intelligence/Project/MainProject/ProjectSubmission/EPL_data.csv"
data = pd.read_csv(input_dir)


#compute the winner
data['1'] = (data['home_goals']>data['away_goals'])*1
data['X'] = (data['home_goals']==data['away_goals'])*1
data['2'] = (data['home_goals']<data['away_goals'])*1

probabilities = pd.DataFrame(np.random.randint(1,100,size=data[['1','X','2']].shape),columns=['1','X','2'])
probabilities = (probabilities.T/probabilities.sum(1)).T
true_result = data[['1','X','2']].idxmax(1)
model = MyLogisticModel()
#no need to add the index but index have to be the same.
probabilities.index = [x+' - '+y for x,y in (data[['home','away']].values)]
true_result.index = [x+' - '+y for x,y in (data[['home','away']].values)]

probabilities.tail().round(2)
true_result.tail()

#compute the losses!
log_loss = model.compute_1_into_2_log_loss(probabilities.tail(50),true_result.tail(50))
hit_ratio = model.compute_1_into_2_hit_ratio(probabilities.tail(50),true_result.tail(50))
number_of_matches = probabilities.shape[0]
print(f'The log-loss for the {number_of_matches} matches is {np.round(log_loss,3)} and the hit-ratio is {hit_ratio*100}%')

#get the data in the train date range
data.date = pd.DatetimeIndex(data.date)  
data_train = data.loc[(data.date < datetime.datetime(2021,3,1)) & (data.date >= (datetime.datetime(2018,3,4)))]

#train the model

model.fit(data_train.home,data_train.away,data_train.home_goals,data_train.away_goals)

teams_coef = model.get_coef(model.model_)[['home wins']].sort_values('home wins',ascending=False) .round(2)
teams_coef.index = [x[0] for x in teams_coef.index.values.ravel()]
print(teams_coef)
new_teams_coef = teams_coef.head(5).append(pd.DataFrame('...',index=['...'],columns=['home wins'])).append(teams_coef.tail(5))

print(teams_coef)

currentTotalPoints, weekWiseData = model.processOutput();

outputdf = pd.DataFrame(currentTotalPoints.items(), columns=['Club', 'Points'])
print(outputdf)
plt.xticks(rotation = 90)
outputdf.plot.scatter(x = 'Club', y = 'Points', s = 114
        ,title=' Points Graph', cmap='viridis')

plt.xticks(rotation = 90)

sorted_df = outputdf.sort_values(by='Points', ascending=False)
model.plotTeamDiagrams(sorted_df)



sns.set(font_scale=1.4)
outputdf.set_index('Club')['Points'].plot(figsize=(12, 10), linewidth=2.5, color='orange')
plt.xlabel("Club", labelpad=15)
plt.ylabel("Points", labelpad=15)
plt.title("Total Points of All the clubs", y=1.02, fontsize=22)
plt.legend()
plt.xticks(rotation = 90)

plt.yticks([x for x in range(0,120,10)])
plt.clf()
clubWeekWiseList = []
for keys in weekWiseData.keys():
        df = pd.DataFrame(weekWiseData[keys].items(), columns=['Club', 'Points'])
        df['total'] = 0
        x = 0
        for i in df.index:
            x = x + df.loc[i,'Points']
            df.at[i, 'total'] = x
        plt.plot(df['total'], label = keys)
        clubWeekWiseList.append(df)

plt.xticks(rotation = 90)
plt.yticks([x for x in range(0,120,10)])
plt.xlabel('Week Numbers', fontsize=12)
plt.ylabel('Points Accumulated', fontsize=20)
plt.legend()
plt.show()
svm = model.fitSVM(data_train.home,data_train.away,data_train.home_goals,data_train.away_goals)
z = model.predict_winner_svm('Arsenal','Manchester City',svm)
a = model.predict_winner_svm('Liverpool','Manchester City',svm)
b = model.predict_winner_svm('Liverpool','Everton',svm)
c = model.predict_winner_svm('Manchester United','Manchester City',svm)