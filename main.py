# Run this app with `python main.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

app = Dash(__name__)
server = app.server

colors = {
    'background': '#00A619',
    'text': '#000000',
    'background2': '#FFFFFF',
}

#'background': '#111111',
# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

url = 'https://github.com/Isaacgv/data_visual_project/blob/main/data/train.csv?raw=true'
df = pd.read_csv(url)

df['date'] = pd.to_datetime(df['match_date'], errors='coerce')
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

df_ = df.groupby(["year", "month"]).count().reset_index()[["year", "month", "id"]]

df_["period"] = df_["year"].astype(str) +" - "+ df_["month"].astype(str) 

fig1 = px.bar(df_, x='period', y='id', labels={'id':'Matchs'})


league_team =pd.concat([df[['home_team_name', 'league_name']].rename(columns={"home_team_name":'away_team_name'}), 
           df[['away_team_name', 'league_name']]]).drop_duplicates(ignore_index=True)

#"Top 10 Most Represented Leagues"
league_team.set_index(['league_name'])

leagues_count = league_team.set_index(['league_name']).value_counts(['league_name', "away_team_name"]).reset_index()
result = leagues_count['league_name'].value_counts().reset_index().sort_values(by="league_name", ascending=False)[:20]
leagues = leagues_count[leagues_count['league_name'].isin(result["index"])]
fig2 = px.treemap(leagues, path=['league_name', 'away_team_name'])
fig2.update_traces(root_color="lightgrey")
fig2.update_layout(margin = dict(t=50, l=25, r=25, b=25))


#Rate for Different Competitions Types
df2 = df.copy(deep=True)
df2.loc[df2['is_cup'] == False,'game_type'] = 'League'
df2.loc[df2['is_cup'] == True,'game_type'] = 'Cup'
df2.loc[df2['league_name'] == 'Club Friendlies','game_type'] = 'Friendly'

df1 = df2.groupby('game_type')['target'].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

features = ['target','game_type']

fig3 = px.histogram(df1, x="game_type", y="percent",
             color='target', barmode='group',
             histfunc='avg',
             height=400, labels={'game_type':'Game Type'})


#"Match result if it's a cup match or not"
# Result per home and away team
impact_cup = df.groupby(['is_cup'])['target'].value_counts(normalize=True).rename('percentage').mul(100).reset_index().sort_values('target')
is_cup = impact_cup[impact_cup['is_cup']==True]
not_cup = impact_cup[impact_cup['is_cup']==False]
fig4 = go.Figure()

fig4 = px.histogram(impact_cup, x="target", y="percentage",
             color='is_cup', barmode='group',
             histfunc='avg',
             height=400, labels={'target':'Streak'})



s = 'away_team_history_rating_'
s2 = 'home_team_history_rating_'
s3 = 'away_team_history_opponent_rating_'
s4 = 'home_team_history_opponent_rating_'
s_id = 'away_team_history_league_id'
s_id = 'home_team_history_league_id'

p = "away_team_history_is_play_home_"
p2 = "home_team_history_is_play_home_"
p3 = "away_team_opponent_history_is_play_home_"
p4 = "home_team_opponent_history_is_play_home_"

column_names = ["date"]
new_df = pd.DataFrame(columns = column_names)

history_df = df.loc[:, df.columns.str.contains(p)|df.columns.str.contains(p2)\
                    |df.columns.str.contains(s)|df.columns.str.contains(s2)|df.columns.str.contains(s3)|df.columns.str.contains(s4)|(df.columns == 'id')|(df.columns=='match_date')]
history_df = pd.wide_to_long(history_df, stubnames=[s,s2,s3,s4,p,p2,p3,p4], i="id",j='i')


history_df.loc[history_df[p2]==0.0,p4] = 'home'
history_df.loc[history_df[p2]==1.0,p4] = 'away'
history_df.loc[history_df[p]==0.0,p3] = 'home'
history_df.loc[history_df[p]==1.0,p3] = 'away'


history_df.loc[history_df[p2]==0.0,p2] = 'away'
history_df.loc[history_df[p2]==1.0,p2] = 'home'
history_df.loc[history_df[p]==0.0,p] = 'away'
history_df.loc[history_df[p]==1.0,p] = 'home'

order = ['home','away']


res = pd.DataFrame(columns=['ratings','home_or_away'])
ps = [p,p2]
for i, v in enumerate([s,s2]):
    temp = history_df[[v,ps[i]]]
    temp.rename(columns={v: 'ratings', ps[i]: 'home_or_away'}, inplace=True)
    res = pd.concat([res,temp], axis=0)
res.reset_index(inplace=True)

fig5 = go.Figure()

fig5.add_trace(go.Histogram(x=res[res["home_or_away"]=="home"]["ratings"], name='Home', histnorm='probability density'))
fig5.add_trace(go.Histogram(x=res[res["home_or_away"]=="away"]["ratings"], name='Away', histnorm='probability density'))

# Overlay both histograms
fig5.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig5.update_traces(opacity=0.75)

fig5.update_layout(
    xaxis_title="Team Rating",
)


def add_streak(df):
    h_t_goal = "home_team_history_goal_"
    h_t_opp_goal = "home_team_history_opponent_goal_"
    a_t_goal = "away_team_history_goal_"
    a_t_opp_goal = "away_team_history_opponent_goal_"
    df["away_streak"] = np.nan
    df["home_streak"] = np.nan
    df['streak_type'] = np.nan
    
    def apply_streaks(x):
        home_streak = 0
        away_streak = 0
        home_streak_over = False
        away_streak_over = False
        home_streak_type = None
        away_streak_type = None
        
        if  x[h_t_goal+str(1)] > x[h_t_opp_goal+str(1)]:
            home_win_streak = True
            home_streak_type = 'win'
        elif x[h_t_goal+str(1)] < x[h_t_opp_goal+str(1)]:
            home_win_streak = False
            home_streak_type = 'loss'
        else:
            home_streak_type = 'draw'
            
        if  x[a_t_goal+str(1)] > x[a_t_opp_goal+str(1)]:
            away_win_streak = True
            away_streak_type = 'win'
        elif x[a_t_goal+str(1)] < x[a_t_opp_goal+str(1)]:
            away_win_streak = False
            away_streak_type = 'loss'
        else:
            away_streak_type = 'draw'
            
        def check_streak(streak,streak_over,streak_type,team_goal,opp_goal):
            if not streak_over and streak_type=='win' and team_goal > opp_goal:
                streak+=1
            elif not streak_over and streak_type=='loss' and team_goal < opp_goal:
                streak+=1
            elif not streak_over and streak_type =='draw' and team_goal == opp_goal:
                streak+=1
            else:
                streak_over = True
            return streak, streak_over
                
        for i in range(1,11):
            if not home_streak_over:
                home_streak, home_streak_over = check_streak(home_streak,home_streak_over, home_streak_type, x[h_t_goal+str(i)],x[h_t_opp_goal+str(i)])
            if not away_streak_over:
                away_streak, away_streak_over = check_streak(away_streak,away_streak_over, away_streak_type, x[a_t_goal+str(i)],x[a_t_opp_goal+str(i)])
            
        return pd.Series([home_streak,home_streak_type,away_streak,away_streak_type],index=['home_streak','home_streak_type','away_streak','away_streak_type'])

    
    df[['home_streak','home_streak_type','away_streak','away_streak_type']] = df.apply(lambda x: apply_streaks(x), axis=1)
    return df


temp_df = add_streak(df)

hue_order = ['win','loss','draw']
temp_df["home_target"] = 'draw'

temp_df.loc[temp_df['target']=='home','home_target'] = 'win'
temp_df.loc[temp_df['target']=='away','home_target'] = 'loss'

t1 = temp_df.loc[temp_df['home_streak_type']=='win']
grouped = t1.groupby(['home_streak','home_streak_type'])['home_target'].value_counts(normalize=True)
grouped = grouped.mul(100)
grouped = grouped.rename('percent').reset_index()

fig6 = px.line(grouped, x="home_streak", y='percent', color='home_target', markers=True)


t1 = temp_df.loc[temp_df['home_streak_type']=='loss']
grouped = t1.groupby(['home_streak','home_streak_type'])['home_target'].value_counts(normalize=True)
grouped = grouped.mul(100)
grouped = grouped.rename('percent').reset_index()

fig7 = px.line(grouped, x="home_streak", y='percent', color='home_target', markers=True)



def number_of_previous_games(row, home):
    for i in range(1,11):
        col = f"{home}_team_history_match_date_{i}"
        if pd.isna(row[col]):
            return i-1  
    return i
  
df["number_home_previous_games"] = df.apply(lambda row: number_of_previous_games(row, "home"), axis=1)
df["number_away_previous_games"] = df.apply(lambda row: number_of_previous_games(row, "away"), axis=1)

def change_coach_last_10(row, home):
    number_previous_games = row[f"number_{home}_previous_games"]

    list_coaches = [f"{home}_team_history_coach_{x}" for x in range(1,number_previous_games+1)]
    current_coach = row[f"{home}_team_coach_id"]
    change_coach=0
    for coach_col in list_coaches:
        prev_coach = row[coach_col]
        if pd.notna(prev_coach):
            if prev_coach!=current_coach:
                change_coach=1
                break
            else:
                current_coach = prev_coach
                
                
    return change_coach

df["home_team_change_coach"] = df.apply(lambda row: change_coach_last_10(row, "home"), axis=1)
df["away_team_change_coach"] = df.apply(lambda row: change_coach_last_10(row, "away"), axis=1)

data_coach = df[["home_team_change_coach", "away_team_change_coach", "target"]]

fig8 = px.parallel_categories(data_coach, 
                             dimensions=['home_team_change_coach', 'away_team_change_coach', 'target'],
                            )

app.layout = html.Div(style={'backgroundColor': colors['background2']}, children=[
    html.H1(
        children='Football Match Probability Prediction',
        style={
            'textAlign': 'center'
            
        }
    ),
    html.H3(
        children='The data set contains more than 150000 historical world football matches from 2019 to 2021, with more than 860 leagues and 9500 teams.',
        
    ),

    html.H3(
        children='Hipotesis : ',  
    ),

    html.H3(
        children='- Home team has advantage on winning the game.',  
    ),

    html.H3(
        children='- The Match in a cup influence the streak.',  
    ),

    html.H3(
        children='- Streaks results from past games will influence the result.',  
    ),

    html.H3(
        children='- New coach influence in the perform.',  
    ),


    html.H2(children='Match Date Distribution Over Year and Month', style={
        'textAlign': 'center',
        'color': colors['text'],
        'backgroundColor': colors['background']
    }),

    dcc.Graph(
        id='example-graph-1',
        figure=fig1
    ),
    html.H2(children='Top 20 Most Represented Leagues', style={
        'textAlign': 'center',
        'color': colors['text'],
        'backgroundColor': colors['background']
    }),

    dcc.Graph(
        id='example-graph-2',
        figure=fig2
    ),
    
    html.H2(children="Match Result - Cup Match or Not", style={
        'textAlign': 'center',
        'color': colors['text'],
        'backgroundColor': colors['background']
    }),

    dcc.Graph(
        id='example-graph-3',
        figure=fig3
    ),
     
    html.H2(children='Different Competitions Types', style={
        'textAlign': 'center',
        'color': colors['text'],
        'backgroundColor': colors['background']
    }),

    dcc.Graph(
        id='example-graph-4',
        figure=fig4
    ),
   
    html.H2(children='Rating History - Home vs Away', style={
        'textAlign': 'center',
        'color': colors['text'],
        'backgroundColor': colors['background']
    }),

    dcc.Graph(
        id='example-graph-5',
        figure=fig5
    ),

    html.H2(children='Rate Depending on Win Streak', style={
        'textAlign': 'center',
        'color': colors['text'],
        'backgroundColor': colors['background']
    }),
    dcc.Graph(
        id='example-graph-6',
        figure=fig6
    ),

    html.H2(children='Rate Depending on Loss Streak', style={
        'textAlign': 'center',
        'color': colors['text'],
        'backgroundColor': colors['background']
    }),
    dcc.Graph(
        id='example-graph-7',
        figure=fig7
    ),
    html.H2(children='Change of Coaches in Last 10 Matches', style={
        'textAlign': 'center',
        'color': colors['text'],
        'backgroundColor': colors['background']
    }),
    dcc.Graph(
        id='example-graph-8',
        figure=fig8
    ),

    html.H2(children='Conclusions', style={
        'textAlign': 'center',
        'color': colors['text'],
        'backgroundColor': colors['background']
    }),
    html.H3(
        children='- There is a bigger chance of winning when playing in home.',  
    ),
    html.H3(
        children='- When a match is a cup, there are fewer draws, teams tend to take more risks.',  
    ),
    html.H3(
        children='- Exist relationship between match result and points on the last games.',  
    ),
    html.H3(
        children='- Higher chance of the streak to continue.',  
    ),
    html.H3(
        children='- There is a bigger chance of losing if the is a new coach.',  
    ),
  
])

if __name__ == '__main__':
    app.run_server(host="0.0.0.0",
    debug=True)
