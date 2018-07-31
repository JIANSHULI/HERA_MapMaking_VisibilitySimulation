import pandas as pd
import numpy as np
import time
import json
import requests

tomato = pd.DataFrame(columns=['date', 'score', 'city', 'comment', 'nick'])
for i in range(0, 1000):
	j = np.random.randint(1, 1000)

	print(str(i) + ' ' + str(j))
	
	try:
		time.sleep(2)
		url = 'http://m.maoyan.com/mmdb/comments/movie/1212592.json?_v_=yes&offset=' + str(j)
		html = requests.get(url=url).content
		data = json.loads(html.decode('utf-8'))['cmts']
		
		for item in data:
			tomato = tomato.append({
                'date': item['time'].split(' ')[0],
                'city': item['cityName'],
				'score': item['score'],
				'comment': item['content'],
				'nick': item['nick']}, ignore_index=True)
		tomato.to_csv('/Users/JianshuLi/Downloads/' + '西虹市首富4.csv', index=False)
		
	except:
		pass