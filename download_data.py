# %%
import subprocess
import requests
import re
from datetime import datetime, timedelta
import pathlib
import json
from tqdm import tqdm
import pandas as pd


def request_progress_bar(url):
    r = requests.get(url, stream=True, allow_redirects=True)
    total_size = int(r.headers.get('content-length'))
    initial_pos = 0
    output = b""

    with tqdm(total=total_size, unit='B', unit_scale=True, initial=initial_pos, desc="Downloading Data") as pbar:
        for ch in r.iter_content(chunk_size=2**20):
            if ch:
                output += ch
                pbar.update(len(ch))

    return output

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))

def download_data(forced_download=False):
    p = pathlib.Path("data/LastUpdated.json")

    if p.exists():
        with open(p) as f:
            data = json.load(f)
        
        match_string = "%Y-%m-%dT%H:%M:%S"
        for key, value in data.items():
            data[key] = datetime.strptime(value[:len(match_string)+1], match_string)
        output_dir = pathlib.Path("data", data['last_updated'].strftime('%Y-%m-%d'))

    else:
        data = {
            "last_updated": None,
            "last_checked": None,
        }

    if forced_download or data['last_checked'] is None or data['last_checked'] + timedelta(minutes=5) < datetime.now() or not output_dir.exists():
        data['last_checked'] = datetime.now()
        
        # Make request to check for new data
        # Cases (publish & specimen), virusTests, deaths
        metrics = [
            'newCasesByPublishDate',
            'newCasesBySpecimenDate',
            'newVirusTestsBySpecimenDate',
            'newDeaths28DaysByDeathDate',
        ]
        url = f"https://api.coronavirus.data.gov.uk/v2/data?areaType=ltla&metric={'&metric='.join(metrics)}&format=json"
        
        r = requests.get(url, headers={"Range": "bytes=0-1024"})

        match = re.findall(r"\d{4}-\d{2}-\d{2}", r.text)[0]

        online_date = datetime.strptime(match, '%Y-%m-%d')
        
        if forced_download or data['last_updated'] is None or online_date > data['last_updated'] or not output_dir.exists():
            data['last_updated'] = online_date

            # Download new data
            json_string = request_progress_bar(url)
        
            json_data = json.loads(json_string.decode("utf-8"))

            df = pd.DataFrame(json_data['body'])

            output_dir = pathlib.Path("data", online_date.strftime('%Y-%m-%d'))
            output_dir.mkdir(parents=True, exist_ok=True)

            dfs = {}
            for metric in metrics:
                dfs[metric] = []
                for i, group in tqdm(df.groupby("date"), desc=f"Transforming {metric}"):
                    dfs[metric].append(
                        group
                        .set_index("areaCode")
                        .loc[:, [metric]].T
                        .rename({metric: datetime(*[int(x) for x in i.split("-")])})
                    )
                pd.concat(dfs[metric]).reset_index().to_feather(output_dir.joinpath(f"{metric}.feather"))

            # with open(output_dir.joinpath("raw_data.json"), 'wb') as f:
            #     f.write(json_string)

        with open(p, "w+") as f:
            json.dump(data, f, default=json_serial)
    
    return output_dir

if __name__ == "__main__":
    print(download_data(True))