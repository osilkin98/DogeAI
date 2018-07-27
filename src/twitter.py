# Import from a different file so as not to reveal keys to Git VCS
from src import keys, doge_classifier as dc
import tweepy
from src.image_file import TempImage
import queue
import threading
print("importing Tensorflow...")
import tensorflow as tf
import datetime
from time import time


#################################################################################################################
# TODO(0): Create an Experimental Branch & Secret Twitter Account for Testing & Deploying Experimental Versions #
# TODO(1): Create an LSTM RNN to generate doge-styled responses for funny captions                              #
# TODO(2): Convert From Using a Binary Classifier to 3-Way Classifier to Filter out unwholesome Doges Submitted #
# TODO(3): Create a WebPage to Display Wholesome User-Submitted Doge Images                                     #
#################################################################################################################


# set the keys in the authorization
auth = tweepy.OAuthHandler(keys.consumer_key, keys.consumer_secret)
auth.set_access_token(keys.access_token, keys.access_token_secret)

# set the tweet queue
tweet_queue = queue.Queue()
processing_thread_name = "main_processing_thread"

try:
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
except Exception as e:
    print("Exception e raised: {}".format(e))

if __name__ == '__main__':
    classifier = tf.estimator.Estimator(model_fn=dc.doge_convolution,
                                        model_dir='{}'.format(keys.training_set_path)
                                        )


# returns image object from tweet
def get_image_from_tweet(target_tweet):
    for image in target_tweet.entities['media']: # to retrieve the one element from the array
        # For debugging
        # print(image['media_url'])
        return TempImage(image['media_url'])


# Take in an image object and return an estimator object that classifies whether or not the image is a doge
def identify_doge(image):
    # creates and return an estimator object with an anonymous numpy input function creation
    return list(classifier.predict(input_fn=tf.estimator.inputs.numpy_input_fn(
        x={"x": image},
        shuffle=False,
        num_epochs=1)
    ))[0]


# Send an error message to the username receiving message logs
def send_error_message(message):
    api.send_direct_message(user="{}".format(keys.log_username),
                            text="Ran Into Error: {}".format(message))


# Send an ordinary log message to the user specified in the keys file
def log_message(message):
    api.send_direct_message(user="{}".format(keys.log_username),
                            text=message)


# Main method to handle whether or not the image submitted was a doge and send out a tweet
def doge_or_not_doge(target_tweet):
    # print("entered doge or not doge")
    image = get_image_from_tweet(target_tweet)
    # Image was Doge
    # print(json.dumps(target_tweet.entities, indent=4, sort_keys=True, separators=(',', ': ')))
    print("tweet id: {}".format(target_tweet.id))
    start_time = time()
    doger = identify_doge(image.get_numpy())
    end_time = time()
    print(doger)
    log_message("Image {} sent from @{} got a doge probability \
of {:0.3f}%\n\nTime taken for image convolution: {:0.3f} seconds".format(
                                target_tweet.entities['media'][0]['media_url'],
                                target_tweet.user.screen_name,
                                doger['probabilities'][1] * 100,
                                (end_time - start_time))
    )

    ''' Varied replies based on doge class probabilities '''
    if doger['probabilities'][1] >= 0.8:
        print("very doge")
        api.update_status("@{} Such doge, much excite!".format(target_tweet.user.screen_name),
                          in_reply_to_status_id=target_tweet.id)
    elif 0.8 > doger['probabilities'][1] >= 0.6:
        api.update_status("@{} wow, doge!".format(target_tweet.user.screen_name), in_reply_to_status_id=target_tweet.id)
    elif 0.5 <= doger['probabilities'][1] < 0.6:
        api.update_status("@{} much doge".format(target_tweet.user.screen_name), in_reply_to_status_id=target_tweet.id)
    elif 0.3 <= doger['probabilities'][1] < 0.5:
        api.update_status("@{} maybe doge".format(target_tweet.user.screen_name), in_reply_to_status_id=target_tweet.id)
    else:
        print("wow not doge")
        api.update_status(
            "@{} wow not doge, much sad".format(target_tweet.user.screen_name), in_reply_to_status_id=target_tweet.id)

    # Regardless of the probability, if the classifer detected a doge
    if doger['classes'] == 1:
        try:
            print("Attempting to upload {}".format(image.absolute_path))
            #################################################################
            # TODO(1): Create an LSTM based RNN to generate funny responses #
            #################################################################
            api.update_with_media(filename="{}".format(image.absolute_path),
                                  status="amazing doge submitted by @{}!".format(target_tweet.user.screen_name))
        except tweepy.TweepError as te:
            msg = "Updating with Media {} failed: {}".format(image.absolute_path, te)
            print(msg)
            send_error_message(msg)
        except FileNotFoundError as fnf:
            print(fnf)
            send_error_message(fnf)
        except Exception as e:
            print(e)
            send_error_message(e)


# this is a deprecated function
def process_tweet(target_tweet):
    if 'media' in target_tweet.entities:
        doge_or_not_doge(target_tweet)
    else:
        print("Tweet by {} was not an image, tweet body: {}".format(target_tweet.user.screen_name, target_tweet.text))


# Main function that will process the queue of tweets
def process_queue():
    while not tweet_queue.empty():
        tweet_to_process = tweet_queue.get()
        print("Processing tweet with ID {} by {}".format(tweet_to_process.id, tweet_to_process.user.screen_name))
        process_tweet(tweet_to_process)
    print("tweet_queue is empty, process_queue is exiting")


class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        # print("[MyStreamListener.on_status]: {}".format(status))
        # print("Formatting as json: ")
        # print(json.dumps(status._json, sort_keys=True, indent=4, separators=(',', ': ')))
        print("got tweet from {}: {}".format(status.user.screen_name, status.text))

        # Add the tweet to the queue of tweets to be processed
        tweet_queue.put(status)

        # Check to see if there's already a thread that's processing tweets in the background
        for thread in threading.enumerate():
            # if there is a thread that's currently processing items
            if thread.name.lower() == processing_thread_name:
                # Return from the current function
                print("Thread {} was found and so we are not creating a new one.".format(thread.name))
                return

        # Since we were unable to find the processing thread, we just start a new one
        main_thread = threading.Thread(name=processing_thread_name, target=process_queue)
        main_thread.daemon = True  # Enable the thread to run in the background as a daemon
        main_thread.run()

    def on_error(self, status_code):
        if status_code == 420:
            print("Status code 420, returning")
            send_error_message(status_code)
            return False

    # def on_data(self, raw_data):
    #     print("got raw data: {}".format(json.dumps(raw_data, indent=4, sort_keys=True)))

    def on_direct_message(self, status):
        print("Got direct message {} from {}".format(status.text, status.user.screen_name))

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            api.send_direct_message("{}".format(keys.log_username),
                                    text="Going Offline {}".format(datetime.datetime.now()))
            api.update_profile(description="[OFFLINE] {}".format(api.me().description))
        except tweepy.TweepError as e:
            print("{}".format(e.response))
            send_error_message(e)


# main function run stack
try:
    print("Starting up...")
    user = api.me()

    print("Changing Description to be online...")
    api.update_profile(description=api.me().description.replace("[OFFLINE] ", ""))

    print("Creating Stream Listener...")
    myStreamListener = MyStreamListener()

    print("Connecting myStreamListener to twitter api...")
    myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)

    log_message("[{}:{}] Doge is online".format(datetime.datetime.now().hour, datetime.datetime.now().minute))

    print("Listening in on '@{}'".format(keys.listen_username))
    myStream.filter(track=["@{}".format(keys.listen_username)])

    api.update_status("Going Offline {}".format(datetime.datetime.now()))
    api.update_profile(description="[OFFLINE] {}".format(api.me().description))

except tweepy.TweepError as e:
    print(e)
except KeyboardInterrupt as ki:
    print("{}".format(ki))
except Exception as e:
    print(e)
finally:
    '''try:
        api.update_status("Going Down For Maintenance at {}".format(datetime.datetime.now()))
    except tweepy.TweepError as e:
        print("{}".format(e.response))
    '''
    print("Program Finished Execution")
