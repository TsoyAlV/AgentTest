"""Run from backend/. Tokens are read from .env, never passed as command-line arguments."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from uuid import uuid4

import httpx

from app.config import Settings

parser=argparse.ArgumentParser(description='Parallel chat API test; creates a separate chat for each request')
parser.add_argument('--base-url',default='http://127.0.0.1:8000')
parser.add_argument('--users',required=True,help='Existing active user IDs separated by commas')
parser.add_argument('--requests',type=int,default=4)
parser.add_argument('--parallel',type=int,default=4)
parser.add_argument('--message',default='Какие сведения есть в доступных документах?')
parser.add_argument('--deep',action='store_true',help='Enable bounded deep analysis for each request')
args=parser.parse_args()
if args.requests<1 or args.parallel<1:parser.error('requests and parallel must be positive')
users=[int(value.strip()) for value in args.users.split(',')]
token=Settings().dev_admin_token.get_secret_value()
if not token:parser.error('Set DEV_ADMIN_TOKEN in backend/.env')


def run(index):
    user=users[index%len(users)];started=perf_counter()
    try:
        with httpx.Client(base_url=args.base_url,headers={'Authorization':'Bearer '+token},timeout=240,trust_env=False) as c:
            chat=c.post('/dev/conversations',json={'user_id':user,'title':f'API test {index+1}'})
            if not chat.is_success:return {'user_id':user,'http':chat.status_code,'seconds':round(perf_counter()-started,2)}
            cid=chat.json()['id']
            response=c.post(f'/dev/conversations/{cid}/messages',json={
                'user_id':user,'message':args.message,'client_id':'parallel-test',
                'request_id':str(uuid4()),'limit':5,'deep_thinking':args.deep,
            })
            return {'user_id':user,'conversation_id':cid,'http':response.status_code,'seconds':round(perf_counter()-started,2)}
    except httpx.HTTPError as exc:return {'user_id':user,'error':type(exc).__name__,'seconds':round(perf_counter()-started,2)}


with ThreadPoolExecutor(args.parallel) as executor:
    for result in executor.map(run,range(args.requests)):print(result)
