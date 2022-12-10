
from datetime import date, datetime

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

timedeltas = prod_df.date.map(get_season_delta).to_numpy()

delta_dict = {"winter": [x["winter"] for x in timedeltas],
              "spring": [x["spring"] for x in timedeltas],
              "summer": [x["summer"] for x in timedeltas],
              "autumn": [x["autumn"] for x in timedeltas]}
              
prod_df = pd.concat([df, pd.DataFrame(delta_dict)], axis=1)