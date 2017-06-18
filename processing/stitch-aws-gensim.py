import pandas as pd
import numpy as np
import os
import glob

def find_by_ext(files_path, ext='csv') :
    files_path = os.path.join(files_path, '*.' + ext)
    files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)
    return files


if __name__ == '__main__':

    gensim_data_path = 'preprocessed'
    pickled_dfs = find_by_ext(gensim_data_path, ext='pkl')
    df = None

    def make_pkl() :
        #merge the lemmatized reviews from aws into one df
        for pickled_df in pickled_dfs :
            if df is None :
                df = pd.read_pickle(pickled_df)
            else :
                df = df.append(pd.read_pickle(pickled_df))

        df.to_pickle('reviews_proc.pkl')
        print("Finished")


    def make_complete_pkl() :
        #bf = pd.read_csv('csvs/biz_details.csv')
        #lemmatized reviews...
        rf = pd.read_pickle('reviews_proc.pkl')
        #port stemmed reviews
        pf = pd.read_csv('csvs/reviews-stemmed.csv')

        #reorder columns
        columns_proc = ['review_id', 'biz_id',  'stars', 'review_date', 'user_photos', 'user_reviews', 'page_num', 'page_pos', 'check_ins',
       'coupon_mentioned', 'delivery_mentioned' , 'wait_mentioned', 'server_mentioned', 'elite',
       'current_review', 'happyhour_mentioned', 'listed_in', 'num_lists',
       'ordered_online_with_yelp', 'page_date', 'page_start', 'purchased_a_yelp_deal', 'review',  'rotd',  'user_city',
       'user_date_sign_up', 'user_display_name', 'user_friends', 'user_id',
       'user_location', 'user_state',
       'votes_cool', 'votes_funny', 'votes_total', 'votes_useful','elite_years',
        'yelp_deal','all_captions', 'review_proc',
           'biz_owner_reply', 'biz_owner_reply_by']
        rf = rf[columns_proc]

        rf['user_id1'] = rf['user_display_name'] + rf['user_date_sign_up']

        #add port-stemmed reviews from older csv
        rf['review_port'] = pf['review_port']
        rf.to_pickle('reviews.pkl')
