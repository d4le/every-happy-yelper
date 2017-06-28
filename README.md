![alt text](images/header.png "Every Happy Yelper")
>"All happy families are alike; each unhappy family is unhappy in its own way" - Leo Tolstoy, _Anna Karenina_

# Objective
<img src="images/tolstoy.png"><p>
Can we predict a restaurant's Yelp rating from the number of topics in its reviews? Our premise is that when using unsupervised modeling, restaurants that are "happy" (well-reviewed) will have a more focused-distribution of topics among those reviews. And, conversely, the unhappy restaurants will be more diverse in their complaints.

# The Data
Our data comes from once place - Denver, Colorado. Yelp, by policy, returns up to 1000 businesses per query and that's it. This should be enough if we supplement it with some shuttered restaurants for balance. In early June, 2017, our query of <a href="https://www.yelp.com/search?cflt=restaurants&find_loc=Denver%2C+CO">restaurants in Denver</a> returned 992 active restaurants. After removing those with less than 10 reviews, our total number from Yelp stood at 921, but I was really hoping for more than 1000 to make my capstone a little more "Big Data". With 71 restaurants having less than 10 reviews, this seems like a pretty good clue we are nearing the end of Yelp's list for Denver regardless of their hard limit of 1000. I supplemented this list with 264 shuttered restaurants to get a total of 1191 Denver restaurants. From these, we going to grab all 228,276 their reviews. Not quite n=all for Denver restaurants on Yelp, but I think we are getting close, and certainly representative. Surprisingly, at least for me, the overall average for all the reviews is 3.94 stars, well above the middle value of 3 one might naively suppose. Of note, Yelp rounds to the nearest half-star in their listing summary for a restaurant, but I've calculated the true star average for each restaurant. Survival bias is surely at play, but this is not a concern for the nature of our question.


# The process
We're going to use unsupervised latent topic analysis. We are not giving hints or labels to our topic modeling algorithms.

# The measurement

# The recommendations

# References
