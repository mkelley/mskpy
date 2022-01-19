"""
Written by Michael S. P. Kelley for syncing DECam data by propopsal ID from the
NOIRLab archive.  Based on the example:
https://github.com/NOAO/nat-nb/blob/master/api-authentication.ipynb

2022-01-16  DRAFT

2022-01-17  Added a failed file log and the ability to count expected files.
            Improved the logging.


Requirements
------------
* Python 3.7
* requests module


Also requires your NOIRLab username and password in a JSON-formatted file,
e.g.,:

    {
        "email": "xxx", "password": "yyy"
    }

"""

import os
import hashlib
import json
import logging
import argparse
import requests


def command_line_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('propid')
    parser.add_argument('--auth', default='auth.json',
                        help='JSON formatted file: {"email": "xxx", "password": "yyy"}')
    parser.add_argument('--limit', type=int,
                        help='stop after downloading this many files')
    parser.add_argument('--on-fail', choices=['halt', 'save', 'ignore'], default='save',
                        help=('fail behavior: [halt] the program, [save] file to '
                              'failed.txt and skip in the future, or [ignore] and '
                              'continue.'))
    parser.add_argument('--count', action='store_true',
                        help='only count matching files in archive')
    return parser.parse_args()


API_URL = 'https://astroarchive.noirlab.edu/api/adv_search/find/'


def get_logger():
    logger = logging.getLogger('decam-dl')
    # reset handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)

    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler('decam-dl.log'))
    for handler in logger.handlers:
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )

    logger.setLevel(logging.INFO)

    return logger


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


def mkdir(path, logger):
    p = '.'
    for d in path.split('/'):
        p = os.path.join(p, d)
        if not os.path.exists(p):
            os.mkdir(p)
            logger.info('  Created directory %s', p)


def query_count(query, logger):
    params = {
        'rectype': 'file',
        'count': 'Y'
    }

    logger.info('Querying NOIRLab for number of matching files...')
    res = requests.post(API_URL, json=query, params=params).json()
    count = res[1]['count']
    logger.info('...%d files.', count)
    return count


def query_results(query, logger):
    last = 0
    while True:
        params = {
            'rectype': 'file',
            'count': 'N',
            'sort': 'archive_filename',
            'offset': last
        }

        logger.info('Querying NOIRLab for matching file metadata...')
        res = requests.post(API_URL, json=query, params=params).json()

        if len(res) == 1:
            break
        else:
            logger.info('...%d files to check.', len(res) - 1)

        with open('last-query.json', 'w') as outf:
            json.dump(res, outf)

        for row in res[1:]:
            yield row

        last = res[0]["PARAMETERS"]['last']


class FailLog:
    FILENAME = 'decam-dl-failed.txt'
    HEADER = '''# decam-dl.py failed download log
# Remove a file from this list to try downloading again.
#
# filename,md5sum,message
'''

    def __init__(self):
        self._failed = {}
        if os.path.exists(self.FILENAME):
            with open(self.FILENAME, 'r') as inf:
                for line in inf:
                    if line[0] == '#':
                        continue
                    fn, checksum, msg = line.strip().split(',')
                    self._failed[fn] = (checksum, msg)
        else:
            with open(self.FILENAME, 'w') as outf:
                outf.write(self.HEADER)

    def __contains__(self, fn):
        return fn in self._failed

    def append(self, fn, checksum, msg):
        self._failed[fn] = (checksum, msg)
        self.save()

    def save(self):
        with open(self.FILENAME, 'w') as outf:
            outf.write(self.HEADER)
            for fn, (checksum, msg) in self._failed.items():
                outf.write(','.join(fn, checksum, msg))
                outf.write('\n')


if __name__ == '__main__':
    args = command_line_arguments()
    logger = get_logger()
    fail_log = FailLog()

    # test out queries at: https://astroarchive.noirlab.edu/api/docs/
    # Use adv_search/find
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

    headers = login(args.auth)

    matching_files = query_count(query, logger)
    if args.count:
        exit(0)

    total_files = 0
    downloaded = 0
    existing_files = 0
    failed_files = 0

    try:
        for row in query_results(query, logger):
            total_files += 1
            fn = os.path.join(*row['archive_filename'].split('/')[-4:])
            if os.path.exists(fn):
                existing_files += 1
                continue
            logger.info('[%d/%d] %s', total_files, matching_files, fn)

            mkdir(fn[:fn.rindex('/')])

            file_url = f'https://astroarchive.noirlab.edu/api/retrieve/{row["md5sum"]}'
            r = requests.get(file_url, headers=headers)
            if r.status_code == 200:
                logger.debug(
                    f'  Read file with size {len(r.content):,} bytes.')
            else:
                failed_files += 1
                msg = (f'  Error getting file ({requests.status_codes._codes[r.status_code][0]}). '
                       f'{r.json()["message"]}')
                logger.error(msg)
                if args.on_fail == 'halt':
                    break
                elif args.on_fail == 'save':
                    fail_log.append(fn, row['md5sum'], msg)

                continue

            h = hashlib.new('md5')
            h.update(r.content)
            sum = h.hexdigest()
            if sum != row["md5sum"]:
                failed_files += 1
                msg = f'  MD5 checksum failed.'
                logger.error(msg)

                if args.on_fail == 'halt':
                    break
                elif args.on_fail == 'save':
                    fail_log.append(fn, row['md5sum'], msg)

                continue
            else:
                with open(fn, 'wb') as outf:
                    outf.write(r.content)

            downloaded += 1
            if args.limit is not None and downloaded >= args.limit:
                logger.info('User requested download limit (%d) reached.',
                            args.limit)
                break
    except Exception as e:
        logger.error(e)

    logger.info('%d files matching query.', matching_files)
    logger.info('%d files checked.', total_files)
    logger.info('  • %d successfully downloaded.', downloaded)
    logger.info('  • %d already downloaded.', existing_files)
    logger.info('%d failed downloads.', failed_files)
