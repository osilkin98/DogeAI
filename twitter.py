# Import from a different file so as not to reveal keys to Git VCS
import keys
import tweepy
from image_file import TempImage


# set the keys in the authorization
auth = tweepy.OAuthHandler(keys.consumer_key, keys.consumer_secret)
auth.set_access_token(keys.access_token, keys.access_token_secret)


# returns image object from tweet
def get_image_from_tweet(target_tweet):
    if 'media' in target_tweet.entities:
        for image in target_tweet.entities['media']: # to retrieve the one element from the array
            # For debugging
            # print(image['media_url'])
            return TempImage(image['media_url'])


def display_image_from_tweet(target_tweet):
    my_image = get_image_from_tweet(target_tweet)
    my_image.image.show()


try:
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    user = api.me()

    # retrieve the first post from timeline
    first_tweet = api.home_timeline()[0]
    display_image_from_tweet(first_tweet)

except tweepy.TweepError as e:
    print(e.message)
finally:
    print("Program Finished Execution")
