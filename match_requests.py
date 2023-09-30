import requests
import numpy as np
import pandas as pd
from io import StringIO

def main():
    query = ['100000-query', '100001-query', '101000-query']
    q = ",".join([vec for vec in query])

    r = requests.get("http://localhost:5000/search", params={"items": q})

    if r.status_code == 200:
        if r.json().get('status') == 'Error': 
            print(r.json()['message'])
        else:
            print('Top 5 items:')
            print(pd.read_json(StringIO(r.json().get('data')), orient='index')) 
    else:
        print(r.status_code)

if __name__ == "__main__":
    main()