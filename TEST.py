import classify

example_tweets = [
    'Science plays a vital but sometimes limited role in #criminal #investigation. Our reports study the opportunities for and challenges of incorporating science in #crime investigation, and identify best practices for improved application.',
    'Raphael to receive living legend award at 2018 La Musa Awards',
    'Our girls Hima Das, Poovamma Raju, Saritaben Gayakwad & Vismaya Koroth on the victory podium flashing their Gold Medals (4X400 Relay) National Anthem playing in the background',
    'home robbery took place with homicide'
]

for tweet in example_tweets:
    print ("\n",tweet,"\n")
    print (classify.relatibility(tweet),"\n")
