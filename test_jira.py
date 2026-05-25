import os, requests
from dotenv import load_dotenv
load_dotenv('.env')

url = os.environ.get('JIRA_BASE_URL', '').rstrip('/') + '/rest/api/3/project'
email = os.environ.get('JIRA_EMAIL', '').strip()
token = os.environ.get('JIRA_API_TOKEN', '').strip()

if not url or not token:
    print('Missing credentials in .env')
    exit()

print('Testing connection for:', email)
response = requests.get(url, auth=(email, token), headers={'Accept': 'application/json'})
print('STATUS:', response.status_code)
if response.status_code == 200:
    projects = response.json()
    print('Accessible Projects:')
    for p in projects:
        print(f"- {p.get('name')} (Key: {p.get('key')})")
else:
    print('ERROR:', response.text)
