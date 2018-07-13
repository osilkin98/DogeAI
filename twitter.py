# Import from a different file so as not to reveal keys to Git VCS
import keys
import tweepy
from image_file import TempImage
import tensorflow as tf
import doge_classifier as dc
import json
import datetime

# set the keys in the authorization
auth = tweepy.OAuthHandler(keys.consumer_key, keys.consumer_secret)
auth.set_access_token(keys.access_token, keys.access_token_secret)
try:
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
except Exception as e:
    print("Exception e raised: {}".format(e))


# returns image object from tweet
def get_image_from_tweet(target_tweet):
    for image in target_tweet.entities['media']: # to retrieve the one element from the array
        # For debugging
        # print(image['media_url'])
        return TempImage(image['media_url'])


def identify_doge(image):
    classifier = tf.estimator.Estimator(model_fn=dc.doge_convolution, model_dir='trained_doge/')

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": image},
        shuffle=False,
        num_epochs=1
    )
    return list(classifier.predict(
        input_fn=test_input_fn))[0]



def doge_or_not_doge(target_tweet):
    print("entered doge or not doge")
    image = get_image_from_tweet(target_tweet)
    # Image was Doge
    # print(json.dumps(target_tweet.entities, indent=4, sort_keys=True, separators=(',', ': ')))
    print("tweet id: {}".format(target_tweet.id))
    doger = identify_doge(image.get_numpy())
    print(doger)
    if doger['classes'] == 1:
        print("doge")
        api.update_status("@{} doge\n\n(Probability of doge: {})".format(
            target_tweet.user.screen_name,
            doger['probabilities'][1]), in_reply_to_status_id=target_tweet.id)
    # Image was Not Doge
    else:
        print("not doge")
        api.update_status("@{} not doge\n\n(Probability of doge: {})".format(
            target_tweet.user.screen_name,
            doger['probabilities'][1]
        ), in_reply_to_status_id=target_tweet.id)


def display_image_from_tweet(target_tweet):
    if 'media' in target_tweet.entities:
        doge_or_not_doge(target_tweet)
    else:
        print("Tweet by {} was not an image, tweet body: {}".format(target_tweet.user.screen_name, target_tweet.text))


class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        # print("[MyStreamListener.on_status]: {}".format(status))
        # print("Formatting as json: ")
        # print(json.dumps(status._json, sort_keys=True, indent=4, separators=(',', ': ')))
        print("got tweet: {}".format(status.text))
        display_image_from_tweet(status)

    def on_error(self, status_code):
        if status_code == 420:
            print("Status code 420, returning")
            return False

    def __del__(self):
        try:
            api.update_status("Going Down For Maintenance at {}".format(datetime.datetime.now()))
        except tweepy.TweepError as e:
            print("{}".format(e.response))
    ''' 
    def on_event(self, status):
        print("[MyStreamListener.on_event]: {}".format(status))

    def on_data(self, raw_data):
        print("[MyStreamListener.on_data]: {}".format(raw_data))
    '''


try:

    user = api.me()

    myStreamListener = MyStreamListener()
    myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
    print("listening")
    myStream.filter(track=["@doge_or"])

    '''
    # retrieve the first post from timeline
    first_tweet = api.home_timeline()[0]
    display_image_from_tweet(first_tweet)
    '''
except tweepy.TweepError as e:
    print(e)
except KeyboardInterrupt as ki:
    print("{}".format(ki.args))
except Exception as e:
    print(e)
finally:
    '''try:
        api.update_status("Going Down For Maintenance at {}".format(datetime.datetime.now()))
    except tweepy.TweepError as e:
        print("{}".format(e.response))
    '''
    print("Program Finished Execution")
