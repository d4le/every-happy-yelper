
# coding: utf-8

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os

from kaggler.metrics.regression import gini as kmgini

if __name__ == '__main__':

    DATA = '../..'
    TMP = '../../tmp'



    def update_biz_model(pickle_it=False) :
        bf = pd.read_pickle('../biz-models.pkl')

        bf100 = bf[bf['reviewCount']>= 100]
        print(bf100.info())

        #these mini-functions are for pandas column-wise updating

        def sorted_list(df,col) :
            dist = df[col]
            return list(np.sort(dist)[::-1])

        def avgs(df):
            tot = df['1stars'] + df['2stars'] + df['3stars'] + df['4stars'] + df['5stars']
            st = df['1stars'] + 2*df['2stars'] + 3*df['3stars'] + 4*df['4stars'] + 5*df['5stars']
            return st/tot


        bf100['rating'] = bf100.apply(lambda x: avgs(x), axis=1)
        bf100['sorted_star_cnts'] = bf100.apply(lambda x: sorted_list(x,'star_cnts'), axis=1)
        bf100['sorted_hdp_probs'] = bf100.apply(lambda x: sorted_list(x,'hdp_probs'), axis=1)


        def prop(df) :
            stars = list(df['sorted_star_cnts'])
            s = sum(stars)
            return [x/s for x in stars]
        def prop2(df) :
            stars = list(df['star_cnts'])
            s = sum(stars)
            return [x/s for x in stars]

        bf100['star_prop'] = bf100.apply(lambda x: prop2(x), axis=1)
        bf100['sorted_star_prop'] = bf100.apply(lambda x: prop(x), axis=1)


        def prop_prob(df) :
            probs = list(df['sorted_hdp_probs'])
            s = sum(probs)
            return [x/s for x in probs[:15]]

        bf100['sorted_hdp_prop'] = bf100.apply(lambda x: prop_prob(x), axis=1)


        from sklearn.metrics import r2_score

        def rscore(df, pcol, ycol):
            y = list(df[ycol])
            p = list(df[pcol])[:5]
            return r2_score(np.array(y),np.array(p))

        bf100['rscore'] = bf100.apply(lambda x: rscore(x,'sorted_hdp_probs','sorted_star_prop'), axis=1)


        def g(df):
            dist = df['sorted_star_prop']
            return gini(np.array(dist))

        def g2(df):
            dist = df['star_prop']
            return gini(np.array(dist))

        def h(df,limit):
            dist = df['sorted_hdp_probs'][:limit]
            return gini(np.array(dist))

        bf100['star_gini'] = bf100.apply(lambda x: g(x), axis=1)
        bf100['star_gini_unsorted'] = bf100.apply(lambda x: g2(x), axis=1)
        bf100['hdp_gini'] = bf100.apply(lambda x: h(x,-1), axis=1)
        bf100['hdp_gini_5'] = bf100.apply(lambda x: h(x,5), axis=1)

        if pickle_it :
            bf100.to_pickle('bf100.pkl')

        bf120 = bf100[bf['reviewCount']>= 120][['biz_id','rating','ratingValue',"reviewCount",'rscore',
                                                'star_gini','star_gini_unsorted','hdp_gini','hdp_gini_5',"star_prop",'sorted_star_prop',
                                                'sorted_hdp_probs']].sort_values('ratingValue')

        return bf120
