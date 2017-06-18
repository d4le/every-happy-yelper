# -*- coding: utf-8 -*-
"""
Modified version of the Yelp Fusion API for getting a
 little more out of the json, like parent categories.

--------------------------------- orig text:
Yelp Fusion API code sample.

This program demonstrates the capability of the Yelp Fusion API
by using the Search API to query for businesses by a search term and location,
and the Business API to query additional information about the top result
from the search query.

Please refer to http://www.yelp.com/developers/v3/documentation for the API
documentation.

This program requires the Python requests library, which you can install via:
`pip install -r requirements.txt`.

Sample usage of the program:
`python sample.py --term="bars" --location="San Francisco, CA"`
"""

import json
import pprint
import requests
import sys
import urllib
import pickle
import re

import csv              # for the split_csvstring function from Part 3.2.2
try:                    # Python 3 compatibility
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import os


from urllib.error import HTTPError
from urllib.parse import quote
from urllib.parse import urlencode





# https://www.yelp.com/developers/v3/manage_app
CLIENT_ID = os.environ['YELP_CLIENT_ID']
CLIENT_SECRET = os.environ['YELP_CLIENT_SECRET']
DATA_FOLDER = 'data'


# API constants, you shouldn't have to change these.
API_HOST = 'https://api.yelp.com'
SEARCH_PATH = '/v3/businesses/search'
BUSINESS_PATH = '/v3/businesses/'  # Business ID will come after slash.
TOKEN_PATH = '/oauth2/token'
GRANT_TYPE = 'client_credentials'

# I added to cache the business categories
DATA_FOLDER = 'data'

def obtain_bearer_token(host, path):
    """Given a bearer token, send a GET request to the API.

    Args:
        host (str): The domain host of the API.
        path (str): The path of the API after the domain.
        url_params (dict): An optional set of query parameters in the request.

    Returns:
        str: OAuth bearer token, obtained using client_id and client_secret.

    Raises:
        HTTPError: An error occurs from the HTTP request.
    """
    url = '{0}{1}'.format(host, quote(path.encode('utf8')))

    data = urlencode({
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'grant_type': GRANT_TYPE,
    })
    headers = {
        'content-type': 'application/x-www-form-urlencoded',
    }
    response = requests.request('POST', url, data=data, headers=headers)
    bearer_token = response.json()['access_token']
    return bearer_token


def get_business(business_id, url_params = {}):
    """Query the Business API by a business ID.

    Args:
        business_id (str): The ID of the business to query.

    Returns:
        dict: The JSON response from the request.
    """
    bearer_token = obtain_bearer_token(API_HOST, TOKEN_PATH)

    path = BUSINESS_PATH + business_id
    url = '{0}{1}'.format(API_HOST, quote(path.encode('utf8')))
    headers = {
        'Authorization': 'Bearer %s' % bearer_token,
    }
    response = requests.request('GET', url, headers=headers, params=url_params)

    return response.json()

def get_reviews(business_id, url_params = {}):
    """Query the Business API by a business ID.

    Args:
        business_id (str): The ID of the business to query.

    Returns:
        dict: The JSON response from the request.
    """
    bearer_token = obtain_bearer_token(API_HOST, TOKEN_PATH)


    #https://api.yelp.com/v3/businesses/{id}/reviews
    path = BUSINESS_PATH + business_id + '/reviews'
    url_params = url_params or {}
    url = '{0}{1}'.format(API_HOST, quote(path.encode('utf8')))
    headers = {
        'Authorization': 'Bearer %s' % bearer_token,
    }
    response = requests.request('GET', url, headers=headers, params=url_params)

    return response.json()



def get_biz_id_from_url(url):
    #http://www.yelp.com/biz/axis-optical-berkeley?large_photo=1
    #https://www.yelp.com/biz/sawasdee-thai-restaurant-wuppertal-2
    parts = url.split('/')

    #handels m.yelp.com www.yelp.com yelp.com
    domain = '.'.join(parts[2].split('.')[-2:])

    #will fail for https://www.yelp.co.uk, etc
    #but not an issue yet.
    yelps = ['yelp.com','yelp.ca','yelp.ie']

    if (parts[-2] != 'biz') and (domain not in yelps) :
        #wrong kind
        print("Non-yelp domain for {}".format(domain))
        return

    #get rid of query sting (if present)
    return parts[-1].split('?')[0]


def get_json_from_biz_url(url, url_params = {}):
    """Queries the API by the input values from the user.

    Args:
        term (str): The search term to query.
        location (str): The location of the business to query.
    """

    business_id = get_biz_id_from_url(url)

    business_json = get_business(business_id, url_params)
    reviews_json = get_reviews(business_id, url_params)

    return business_json, reviews_json

def load_yelp_toplevel_categories(refetch=False,
                                  pickle_it=True,
                                  pickle_file="{}/yelp_categories.pkl".format(DATA_FOLDER),
                                  verbose=False):

    if (refetch is False) and os.path.exists(pickle_file):
        try:
            with open(pickle_file, "rb") as file:
                if verbose :
                    print("Locally reading pickle '{}'".format(pickle_file))
                unpickler = pickle.Unpickler(file);
                parent = unpickler.load();
                if not isinstance(parent, dict):
                    print("Error not correct instance, reloading from Yelp. '{}'".format(pickle_file))
                    refetch = True
                elif verbose :
                    print("Successfully loaded parent dictionary")
        except EOFError:
            print("Error reading file, reloading from Yelp. '{}'".format(pickle_file))
            refetch = True

    if refetch is True:
        if verbose :
            print("Fetching Yelp categories JSON from web...")
        with urllib.request.urlopen("https://www.yelp.com/developers/documentation/v3/all_category_list/categories.json") as url:
            data = json.loads(url.read().decode())

        toplevel = []
        parent = {}
        for item in data :
            if len(item['parents']) > 0:
                parent[item['alias']] = item['parents'][0]
            else:
                toplevel.append(item['alias'])

        for k in parent.keys():
            while parent[k] not in toplevel:
                #print("k NOT in top ", k, parent[k])
                parent[k] = parent[parent[k]]
        if verbose :
            print("{} top level categories\n{} sub categories".format(len(toplevel),len(parent)))
        parent.update(dict(zip(toplevel, toplevel)))

        if pickle_it :
            try :
                with open(pickle_file, 'wb') as f:
                    if verbose :
                        print("...writing pickle '{}'".format(pickle_file))
                    pickle.dump(parent, f, pickle.HIGHEST_PROTOCOL)
            except Exception as error:
                print("Error writing pickle '{}': {}".format(pickle_file, error))

    return parent


if __name__ == "__main__":
    from pymongo import MongoClient
    import pandas as pd
    sheet_id = ''
    sheet = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv'
    df = pd.read_csv(sheet,header=None)
    urls = df.iloc[:,1].unique()

    #url = 'https://www.yelp.com/biz/jack-n-grill-littleton'
    #url = 'https://www.yelp.com/biz/abes-deli-peoria'
    #error in 59 https://www.yelp.com/biz/la-viblia-barcelona-2
    urls = urls[60:1000]
    error_urls = []
    for i, url in enumerate(urls) :
        print('starting...', i, url)
        biz, reviews = get_json_from_biz_url(url)
        if biz.get('error') :
            print("error ",url)
            error_urls.append(url)
            continue

        #pprint.pprint(biz)
        #print("\n\n\n===========================\n\n\n")
        #pprint.pprint(reviews)

        #pprint.pprint(reviews['reviews'])
        parent = load_yelp_toplevel_categories(refetch=False)

        #biz['toplevel_cat']= parent[biz['categories'][0]['alias']]
        cats = list(set(parent[cat['alias']] for cat in biz['categories']))
        if len(cats) > 1 and ('restaurants' in cats) :
            #priority for restaurant category
                biz['toplevel_cat'] = 'restaurants'
        else :
            biz['toplevel_cat'] = cats[0]

        #add to top leve for mongo efficiency
        for item in biz['location'] :
            biz[item] = biz['location'][item]
        if biz['country'] == 'US':
            biz['city-state'] = biz['city'].lower() + '-' + biz['state'].lower()

        if reviews.get('error') :
            print("error in reviews",url)
        else :
            for item in reviews['reviews']:
                item['review_id'] = re.findall('hrid=([^&]*)', item['url'])[0]
                del item['url']
            biz['reviews'] = reviews['reviews']

        #pprint.pprint(biz)

        client = MongoClient()
        #set database that collection is in
        db = client.yelp

        db.businesses.update(biz, biz, upsert=True)
        print("biz..",biz['id'])


    with open('error_urls.pkl', 'wb') as f:
        pickle.dump(error_urls, f, pickle.HIGHEST_PROTOCOL)
