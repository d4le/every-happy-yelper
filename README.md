>"All happy families are alike; each unhappy family is unhappy in its own way" - Leo Tolstoy, _Anna Karenina_

<img src="images/header.png">


# Objective
<img src="images/tolstoy.png"><p>
Can we predict a restaurant's Yelp rating from the number of topics in its reviews? Our premise is that when using unsupervised modeling, restaurants that are "happy" (well-reviewed) will have a more focused-distribution of topics among those reviews. And, conversely, the unhappy restaurants will be more diverse in their complaints.
<p><img src="images/restaurants.png">

# The Data
Our data comes from once place - Denver, Colorado. Yelp, by policy, returns up to 1000 businesses per query and that's it. This should be enough if we supplement it with some shuttered restaurants for balance. In early June, 2017, our query of <a href="https://www.yelp.com/search?cflt=restaurants&find_loc=Denver%2C+CO">restaurants in Denver</a> returned 992 active restaurants. After removing those with less than 10 reviews, our total number from Yelp stood at 921. With 71 restaurants having less than 10 reviews, this seems like a pretty good clue we are nearing the end the list for Denver regardless of Yelp's hard limit of 1000 search results.

<img src="images/restaurants-dist.png">

For balance, I supplemented this list with 264 shuttered restaurants, arriving at a total of 1191 Denver restaurants. Survival bias is surely at play, and adding some shuttered restaurants to the mix might balance things out. From these 1191 restaurants, I grabbed all 228,276 their reviews. Not quite n=all for Denver restaurants on Yelp, but I think we are getting close. Surprisingly, at least for me, the overall average for all the reviews is 3.94 stars, well above the middling value of 3 one might naively suppose. Of note, Yelp rounds to the nearest half-star in their listing summary for a restaurant, but I've calculated the true star average for each restaurant based on all its reviews.

<img src="images/reviews.png">

# EDA and problem ideation
As part of our initial EDA we clean and stem the reviews and run <a href="https://radimrehurek.com/gensim/models/ldamodel.html">Gensim's LDA model</a> on the entire corpus. LDA stands for Latent Dirichlet Allocation. This is a soft topic model that allows for each document in the corpus to belong to multiple topics.
<p>
At this point, I was not sure where the project was heading, and I was just looking over the results. One thing I noticed was how often family members are mentioned in reviews, and how highly ranked such terms are in the corpus of Denver Yelp reviews. A casual browsing will reveal many reviewers on Yelp speak for the entire party, mentioning what the other members of their party had or why they happened upon the restaurant.
<p>
This brought up the question, what is common among all the reviews, regardless of the food served? I wondered if I could create an enormous collection of food and family related stop words to get down to the most basic abstract quality of being a good review.
<p>
After a few attempts with LDA and TF-IDF to get at the heart of abstract goodness, I realized the task was impossible, at least for me. Topic models and similarity measures are always going to find a difference between sushi and steak restaurants at almost every pass. It dawned on me I could use they same unsupervised topic analysis on each restaurant separately as its own corpus. I would lose similarity measures in general, but we could compare intra-restaurant topic distributions. This seemed novel in itself - and worth a try. Of course, we have a corpus problem for smaller restaurants, dropping out those with less than 150 or so reviews
<p>
Finally, we arrive at our question: Do well-rated restaurants have more concentrated, cohesive reviews? Or more specifically, are there fewer topics in a happy restaurant's reviews.


# Measuring happiness by topic density
<img src="images/traditionalk.png"><p>
Traditional topic modeling has a parameter problem -- at least for our quest. The hyperparameter of the number of topics is set before any modeling is done.
<img src="images/hdp.png"><p>


# The recommendations

# References
