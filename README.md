# DogeAI


### *What is DogeAI?*
DogeAI is a project that I started to develop a better 
understanding of Google's [TensorFlow](https://github.com/TensorFlow/TensorFlow) 
framework, and what could be done to improve its functionality. 

### *How Does it Work?*

##### High Level Details:
DogeAI is *currenly* a simple binary classifer that was written in Python 3.5 using
TensorFlow as the deep learning framework, and was implemented as a Twitter Bot. The 
program connects itself to the Twitter API using the Tweepy Python module, and 
creates a stream where it filters out tweets that have the [@DogeAI](https://twitter.com/DogeAI) 
handle. 


##### The Server:
The bot itself runs on a Raspberry PI 1 version B, running Gentto Linux,
as it is the most optimzed Distro for the device. The service itself starts
at either runlevel 3 or 5. All logging messages are directed to my Twitter inbox, so I 
can be alerted if anything goes haywire.


##### OpenCV2 Image Wrapper:
Once the bot receives a tweet, it checks to see if there are any images attached, 
(currently it only processes one at a time) and then downloads and loads them into a 
custom OpenCV2 Image class wrapper which disposes of the image from the computer's
memory once they are no longer being referenced.

##### Binary Classifier:

###### Convolutional Network
Once the images are pre-processed into the program, they are then passed
to the main binary classifer, which takes input images of size `96x96x3`,
where the height and width are 96 pixels, and there are 3 different color channels.
This is then passed through four convolutional 'slices' for lack of a better word.
Each slice consists of a Convolutional layer which applies 12 * 2^i image filters
of size `5x5x1`,for `i = {0, 1, 2, 3}`, and then after each convolution, the data is passed into 
a max-pooling layer which reduces the dimensionality of the  data by 
`(height/(2^i)) x (width/(2^i)) x m`, where `m` is the number of 
filters from the previous convolution operation.

###### Fully-Connected Neural Network
After this, the data is flattened, and passed into a dense, fully-connected
layer of `1024` units, with a unit dropout rate of `10%`. After this, its loss
gets computed by a sparce softmax cross-entropy function, and then it finally gets
passed into the output layer which is normalized using the sigmoid function.

##### Output
Once the processing has been done, the function returns an estimator object 
which tells us what our predictions are, and then based on its confidence, it
tweets out to the user it's prediction in some funny variation. 

##### Featured Images
If the classifier detected that the image is a doge, it will get featured on the 
home page, along with the username of the person who tweeted the image. This
makes it more engaging and fun for people to use, as each time they have a chance
of being seen by other people.


#### *Why Twitter?*
Since the Twitter platform allows for 
users to tweet out images and be able to aggregate them under one field, while 
leaving the home page reserved for "Special" Tweets, it seemed like the perfect choice.


One of the major downsides to using Twitter, however, is the inability to allow users
to continously train the bot in realtime. This is in part due to the fact that 
twitter doesn't have a way to negatively react to tweets, but also because usually 
only the user who sent the Tweet out will ever be able to see it. This 
creates an issue as it allows anyone to continuously provide the bot with falsified
data and throw its learned parameters off.



### *How Did This Project Start?*
The goal of the project, initially, was to just have a bit of fun with friends 
while learning how to actually implement convolutionary classifiers, so I 
could then implement a user-friendly library for doing so in C++, similar to Keras.

As the project progressed I continued to get ideas for various features to implement,
allowing for the bot to be more interactive and engaging, and received tons of 
positive feedback from other users. This just led me to continue to tweak the code
and improve it more and more.

### *What Are The Future Plans of This Project?*
#### Caption Generation using Natural Language Processing
One of the things that make bots so fun to interact with is how they will respond 
to certain phrases and messages being sent to them. What I would like to do, is 
create a Recurrent Neural Network (RNN) with Long-Short Term Memory (LSTM) units
in order to be able to create funny responses to users interacting with the bot,
and learning over time using the responses which the bot will receive from users.

#### Just in Time Package Design
Ever since I created the server in order for the code to run on a separate machine
which has virtually no downtime, I've realized just how much of a pain it can be 
to transfer trained parameter data, and manage version control between my desktop
and the server itself. I'm planning on implementing a feature which would allow
me to achieve code synchronization by simply messaging the bot commands through
the Twitter direct message system. This way, I'd be able to have them transfer
trained data, rebase with the current version, and update modular pieces of the code
without actually having to open a terminal window and do it all manually. 

This way, services can be updated and fixed without having to restart the server
manually. 


#### Wholesome Featuring
During my time of scraping the internet for hundreds of doge images, I found that they
can be very hard to come by, and so I really had to go to the depths of the internet,
and found that there are many versions of the Doge meme which are not as wholesome and 
friendly as others. Some will make jokes about poor relationship dynamics, while others will 
go as far as to be explicitly edgy and racist out of poor taste. 

In order to keep that content away from the main news feed, I am planning on 
remodeling the classifier to be able to differentiate between wholesome and 
goofy pictures, and those that just make fun of drugs and murder. It presents
a very intriguing challenge that would involve the use of deep object recognition
and maybe some form Convolutional to Bi-Directional Recurrent Neural network 
to do text recognition within an image.


