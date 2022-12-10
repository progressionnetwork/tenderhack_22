from sklearn.preprocessing import LabelEncoder
from datetime import date, datetime
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import json

validation_file_name = r"../Chekpoint_test_data.xlsx" 
columns = ['id', 'name', 'okpd2','kpgz','region','nmck','date','inn']

validation_df = pd.read_excel(validation_file_name, header=None, names=columns, skiprows=1)
df['okpd2'] = df['okpd2'].fillna(0)
val_df = validation_df

# Encode inn
le = LabelEncoder()

for cat_var in ['inn']:
    val_df[cat_var] = le.fit_transform(val_df[cat_var])  
    
print(val_df["inn"].unique())
print(val_df["inn"].max())

# Encode region
le = LabelEncoder()

for cat_var in ['region']:
    val_df[cat_var] = le.fit_transform(val_df[cat_var])  
    
print(val_df["region"].unique())
print(val_df["region"].max())


def get_names(kpgzs: str, mapping: dict) -> str:
    try:
        kpgzs = kpgzs.strip()
        result = ""
        for kpgz in kpgzs.split(";"):
            result+=" "+mapping[kpgz]
        return result.strip()
    except Exception as e:
        return ""

def get_embeddings(s: str, ft_model) -> list:
    try:
        return ft_model.get_sentence_vector(s)
    except:
        return []
    
# load fasttext model
ft_model = fasttext.load_model(model)
mapping = {}
for kpgz, name in zip(df_kpgz.str.split().str[0], df_kpgz.str.split().str[1:].str.join(" ")):
    mapping[kpgz] = name

    
# Load OKPD catalog
with open("../data-20160929T0100.json", "r", encoding="utf-8") as f:
    okpd2 = json.load(f)
    
okpd2_dict = {}

for odin_okpd in okpd2:
    try:
        okpd2_dict[odin_okpd["Kod"]] = odin_okpd["Name"]
    except Exception as e:
        pass

kpgz = val_df.kpgz.map(lambda x: get_names(x, mapping=mapping)).to_list()
okpd = val_df.okpd2.map(lambda x: get_names(x, mapping=okpd2_dict)).to_list()
names = val_df.name.to_list()
texts = [" ".join([i,j,z]) for i,j,z in zip(kpgz, okpd, names)]

val_df["kpgz"] = pd.DataFrame({"kpgz":list(map(lambda x: get_embeddings(x, ft_model), texts))})
val_df['date'] = pd.to_datetime(val_df.date, format='%Y-%m-%d %H:%M:%S')

def get_season(now):
    Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
    seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
               ('spring', (date(Y,  3, 21),  date(Y,  6, 20))),
               ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
               ('autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
               ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]
    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= now <= end)


def get_season_delta(stamp):
    season_begin = {"winter":datetime(stamp.year+1,12,1),
                    "spring":datetime(stamp.year+1,3,1),
                    "summer":datetime(stamp.year+1,6,1),
                    "autumn":datetime(stamp.year+1,9,1)}
    current_season = get_season(stamp)
    season_delta = {"winter":None,
                    "spring":None,
                    "summer":None,
                    "autumn":None}
    for season in season_delta.keys():
        if(season == current_season):
            season_delta[season] = (stamp - season_begin[season]).days%365*-1
        else:
            season_delta[season] = (season_begin[season] - stamp).days%365
        season_delta[season]/=365
    return season_delta

timedeltas = val_df.date.map(get_season_delta).to_numpy()

delta_dict = {"winter": [x["winter"] for x in timedeltas],
              "spring": [x["spring"] for x in timedeltas],
              "summer": [x["summer"] for x in timedeltas],
              "autumn": [x["autumn"] for x in timedeltas]}
              
val_df = pd.concat([val_df, pd.DataFrame(delta_dict)], axis=1)

df_dt = pd.DataFrame()
df_dt['date_hour'] = val_df['date'].dt.hour
df_dt['date_minute'] = val_df['date'].dt.minute
val_df = pd.concat([val_df, df_dt], axis=1)
val_df = val_df.drop('date', axis='columns')
val_df

vecs = pd.DataFrame([list(x) for x in val_df.kpgz.to_numpy()])
val_df = pd.concat([val_df, vecs], axis=1)
val_df = val_df.drop('kpgz', axis = 1)
val_df = val_df.drop('name', axis='columns')
val_df = val_df.drop('okpd2', axis = 1)
val_df

X_test = val_df

predict_participans = CatBoostRegressor()
predict_participans.load_model('MAE_loss_16_42_participants.catboost')

predicted_participants = predict_participans.predict(X_test)
print("MAE participants: ", predicted_participants)


price_percent = CatBoostRegressor()
price_percent.load_model('MAE_loss_16_37_price.catboost')

predicted_price = price_percent.predict(X_test)
print("MAE price: ", predicted_price)


#############
#Save results

submit_df = pd.DataFrame(columns=['id','Уровень снижения','Участники'])
submit_df['id'] = val_df['id']
submit_df['Уровень снижения'] = predicted_price * 100
submit_df['Участники'] =  (np.abs(np.round(predicted_participants))).astype(int)
submit_df.to_csv('submit.csv', delimiter=';', index=False)