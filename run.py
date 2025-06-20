import json
import requests

# Flask server endpoint
url = 'http://localhost:8888/detect'

# Path to your JSON lines file
filename = 'access.json'

with open(filename, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue

        try:
            json_data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"❌ Line {idx}: Invalid JSON - {e}")
            continue

        try:
            response = requests.post(url, json=json_data)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Line {idx}: {result}")
            else:
                print(
                    f"❌ Line {idx}: Error {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Line {idx}: Request failed - {e}")
