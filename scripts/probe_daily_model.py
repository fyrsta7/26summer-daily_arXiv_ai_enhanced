import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request

base = os.environ['OPENAI_BASE_URL'].rstrip('/')
key = os.environ['OPENAI_API_KEY']
model = os.environ['MODEL_NAME']


def probe(name, path, payload=None):
    started = time.monotonic()
    request = urllib.request.Request(
        base + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            if payload and payload.get('stream'):
                first_content = None
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    fragment = line[6:].strip()
                    if fragment == b'[DONE]':
                        break
                    chunk = json.loads(fragment)
                    for choice in chunk.get('choices', []):
                        content = choice.get('delta', {}).get('content')
                        if content:
                            first_content = content
                            break
                    if first_content:
                        break
                data = {'choices':[{'first_content':first_content}]}
            else:
                data = json.load(response)
            if path == '/models':
                result = {'matching_models': [m['id'] for m in data.get('data', []) if 'deepseek' in m.get('id', '').lower()]}
            else:
                result = {'choices': data.get('choices', []), 'usage': data.get('usage'), 'error':data.get('error')}
            result['status'] = response.status
    except urllib.error.HTTPError as error:
        result = {'status':error.code, 'error':error.read().decode(errors='replace')[:800]}
    except Exception as error:
        result = {'error':str(error)}
    result.update(name=name, model=model, elapsed_seconds=round(time.monotonic()-started, 2))
    print(json.dumps(result, ensure_ascii=False).replace(key, '[REDACTED]').replace(base, '[BASE_URL]'), flush=True)


plain = {'model':model, 'messages':[{'role':'user','content':'Reply with the word OK.'}], 'max_tokens':64}
structured = {'model':model, 'messages':[{'role':'user','content':'Return JSON: {"ok": true}'}], 'max_tokens':64,
              'response_format':{'type':'json_object'}, 'temperature':0}
plain['thinking'] = {'type':'disabled'}
structured['thinking'] = {'type':'disabled'}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(probe, 'plain_no_thinking', '/chat/completions', plain),
               executor.submit(probe, 'json_no_thinking', '/chat/completions', structured),
               executor.submit(probe, 'stream_no_thinking', '/chat/completions', {**plain, 'stream':True})]
    for future in futures:
        future.result()
