# Import from a different file so as not to reveal keys to Git VCS
import keys
import tweepy
import wget
import os
from PIL import Image

# set the keys in the authorization
auth = tweepy.OAuthHandler(keys.consumer_key, keys.consumer_secret)
auth.set_access_token(keys.access_token, keys.access_token_secret)


# returns image object from tweet
def get_image_from_tweet(target_tweet):
    if 'media' in target_tweet.entities:
        for image in target_tweet.entities['media']: # to retrieve the one element from the array
            print(image['media_url'])
            # all the images from media are guaranteed to be .jpg
            image_dir = 'tmp/image{}.jpg'.format(len(os.listdir('tmp/')) + 1)
            wget.download(image['media_url'], image_dir)
            image_object = Image.open(image_dir)
            return image_object


def display_image(target_tweet, delete_image=True):
    if 'media' in target_tweet.entities:
        for image in target_tweet.entities['media']:
            print(image['media_url'])
            # all the images from media are guaranteed to be .jpg
            image_dir = 'tmp/image{}.jpg'.format(len(os.listdir('tmp/')) + 1)
            wget.download(image['media_url'], image_dir)
            viewable = Image.open(image_dir)
            viewable.show()
            if delete_image:
                os.remove(image_dir)


try:
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    user = api.me()

    # retrieve the first post from timeline
    first_tweet = api.home_timeline()[0]
    display_image(first_tweet)

except tweepy.TweepError as e:
    print(e.message)
finally:
    print("Program Finished Execution")
'''
if 'media' in target_tweet.entities:
    for image in target_tweet.entities['media']:
        
# print(target_tweet.entities)
# api.update_status("{} is deployed and ready for action".format(user.name))

'''
'''
    # print(json.dumps(target_tweet.entities, sort_keys=True, indent=2, separators=(',', ': ')))
    if 'media' in target_tweet.entities:
        for image in target_tweet.entities['media']:
            print(image['media_url'])
            # all the images from media are guaranteed to be .jpg
            image_dir = 'tmp/image{}.jpg'.format(len(os.listdir('tmp/')) + 1)
            wget.download(image['media_url'], image_dir)
            viewable = Image.open(image_dir)
            viewable.show()
            os.remove(image_dir)
'''