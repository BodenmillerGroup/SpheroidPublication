URL_ZENODO = 'https://sandbox.zenodo.org'
import requests
def get_meta(dep_id):
    r = requests.get(f'{URL_ZENODO}/api/records/{dep_id}',
                   # Headers are not necessary here since "requests" automatically
                   # adds "Content-Type: application/json", because we're using
                   # the "json=" keyword argument
                   # headers=headers,
                   headers={'Content-Type': 'application/json'})
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

