import requests, os
import pandas as pd

SERVICE_KEY = ""

def main():
    data = None
    for j in range(1, 12):
        url = f"http://openapi.seoul.go.kr:8088/{SERVICE_KEY}/json/tbLnOpendataRtmsV/{1+((j-1)*1000)}/{j*1000}"
        print(url)
        req = requests.get(url)
        content = req.json()
        con = content['tbLnOpendataRtmsV']['row']
        result = pd.DataFrame(con)
        data = pd.concat([data, result])
        
    data = data.reset_index(drop=True)
    data['CTRT_DAY'] = pd.to_datetime(data['CTRT_DAY'], format="%Y%m%d")
    os.makedirs("./data", exist_ok=True)
    data.to_csv('./data/seoul_real_estate.csv', index=False)
    
if __name__ == "__main__":
    main()