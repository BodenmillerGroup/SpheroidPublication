URL_ZENODO = 'https://sandbox.zenodo.org'
import requests
import time
def get_meta(dep_id):
    status = 0
    wait = 0
    while status == 0:
        time.sleep(wait)
        r = requests.get(f'{URL_ZENODO}/api/records/{dep_id}',
                       # Headers are not necessary here since "requests" automatically
                       # adds "Content-Type: application/json", because we're using
                       # the "json=" keyword argument
                       # headers=headers,
                       headers={'Content-Type': 'application/json'})

        code = r.status_code
        if code == 200:
            status = 1
        elif code == 429:
            wait += 1 + (wait**2)
        else:
            raise Exception(f'Unexpected Zenodo response: status code: {code} for deposition {dep_id}.')
    rjson = r.json()
    return rjson


def _get_file_dict(dep_id):
    fn_dict = {f['key']: f['links']['self'] for f in get_meta(dep_id)['files']}
    return fn_dict

def get_file_dicts(dep_ids):
    fn_dict = {}
    for dep_id in dep_ids:
        fn_dict.update(_get_file_dict(dep_id))
    return fn_dict

