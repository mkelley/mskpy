"""
Written by Michael S. P. Kelley for syncing DECam data by propopsal
ID from the NOIRLab archive.  Based on the example
https://github.com/NOAO/nat-nb/blob/master/api-authentication.ipynb

2022-01-16  DRAFT

Requirements
------------
* Python 3.7
* requests module

Requires your NOIRLab username and password in a JSON-formatted file, e.g.,:

    {
        "email": "xxx",
        "password": "yyy"
    }

"""

import os
import hashlib
import json
import argparse
import requests

parser = argparse.ArgumentParser()
parser.add_argument('propid')
parser.add_argument('--auth', default='auth.json',
                    help='JSON formatted file: {"email": "xxx", "password": "yyy"}')
parser.add_argument('--limit', type=int,
                    help='stop after downloading this many files')
args = parser.parse_args()


def login(auth_file):
    token_url = 'https://astroarchive.noirlab.edu/api/get_token/'
    with open(auth_file) as inf:
        auth = json.load(inf)

    r = requests.post(token_url, json=auth)
    token = r.json()
    if r.status_code == 200:
        headers = dict(Authorization=token)
    else:
        raise Exception(f"Could got get authorization: {token['detail']}")

    return headers


# test out queries at: https://astroarchive.noirlab.edu/api/docs/
query = {
    "outfields": [
        "AIRMASS",
        "md5sum",
        "caldat",
        "instrument",
        "proc_type",
        "prod_type",
        "obs_type",
        "proposal",
        "DATE-OBS",
        "archive_filename"
    ],
    "search": [
        [
            "PROPID",
            args.propid
        ],
        [
            "instrument",
            "decam"
        ],
        [
            "proc_type",
            "resampled"
        ],
        [
            "prod_type",
            "image"
        ]
    ]
}


def mkdir(path):
    p = '.'
    for d in path.split('/'):
        p = os.path.join(p, d)
        if not os.path.exists(p):
            os.mkdir(p)
            print('  Created directory', p)


def query_results(query):
    api_url = 'https://astroarchive.noirlab.edu/api/adv_search/find/'

    last = 0
    while True:
        params = {
            'rectype': 'file',
            'count': 'N',
            'sort': 'archive_filename',
            'offset': last
        }

        print('Querying NOIRLab...')
        res = requests.post(api_url, json=query, params=params).json()

        if len(res) == 1:
            # no more results
            print('Query results exhausted.')
            break
        else:
            print(f'{len(res) - 1} files to check.')

        with open('last-query.json', 'w') as outf:
            json.dump(res, outf)

        for row in res[1:]:
            yield row

        last = res[0]["PARAMETERS"]['last']


headers = login(args.auth)

downloaded = 0
for row in query_results(query):
    fn = os.path.join(*row['archive_filename'].split('/')[-4:])
    if os.path.exists(fn):
        continue
    print(fn)

    mkdir(fn[:fn.rindex('/')])

    file_url = f'https://astroarchive.noirlab.edu/api/retrieve/{row["md5sum"]}'
    r = requests.get(file_url, headers=headers)
    if r.status_code == 200:
        print(f'  Read file with size {len(r.content):,} bytes.')
    else:
        msg = (f'  Error getting file ({requests.status_codes._codes[r.status_code][0]}). '
               f'{r.json()["message"]}')
        raise Exception(msg)

    h = hashlib.new('md5')
    h.update(r.content)
    sum = h.hexdigest()
    if sum != row["md5sum"]:
        msg = f'MD5 checksum failed.'
        raise Exception(msg)
    else:
        with open(fn, 'wb') as outf:
            outf.write(r.content)
        print('  Saved.')

    downloaded += 1
    if args.limit is not None and downloaded >= args.limit:
        print(f'User requested download limit ({args.limit}) reached.')
        break
