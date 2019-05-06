# GAN

This repository was mainly a challenge for me to create a tool using machine learning that actually remotely had a reasonable application. There are two small mini-projects that each revolve around the same concept: using a generative adversarial network to teach the computer how to draw a number.

## General number generation

The first task I sought to do was try to generate any number, which proved to be quite difficult — no matter how hard I tried, it would always generate noise for a bit, then eventually converge to a white canvas. It also didn't help that I was trying to train a model of over 20 million weights with the CPUs of my 2010 Mac (picture 5 seconds per training iteration). After a bunch of convolutional neural networks, hours of tweaking with model sizes and learnin rates, and over twenty dollars of cloud computing, I finally realized the issue: I was not freezing the weights of the discriminator network while backpropagating over the adversarial network. This caused the adversarial network to adjust the weights of the discriminator to conveniently allow it to score a 100% accuracy every time. 

If you run the `main.py` file it will start a server at port 8007 that will allow you to train the network and generate images of random numbers. If you don't feel like pressing the "Train" button over and over again (unlike me), the "Fun" button will repeatedly train the neural net and output images as it progresses. You can view a slideshow of the images generated every few training steps at `/static/images` (look for the "Slideshow" button). Around 5000 training iterations is recommended to get some pretty nice numbers.

Looking at how the images progress is mesmerizing. It's really interesting how some numbers will change, going from a 5 to a 2 to a 4 to a 9.

<img src="numbers.gif" />

## Number to handwriting

After completing the part above, I wanted to make something actually useful. One caveat of using all numbers as a training data set is that you cannot know which number the neural net outputs (unless you pass it through another classifcation neural net). This can be solved by training 10 different GANs that are each only fed a single type of digit, and train each of those. I used this to create a tool that will allow you to input a number and make it look (kind of) handwritten.

In order to use this tool, you need to first train 10 generator models. This can be done with the `train.py` script, which will train and save to disk 10 models. Then, running the `number_writer.py` will create a server at port 8007 with a tool that will allow you to generate handwritten digits. Again, 5000 training iterations for each model is recommend, but because this can take several hours (and a lot of cloud computing), there are already 10 models trained for 5000 iterations in the repository. This should allow you to be able to use the tool without having to train any models. Also, it can be run a really bad computer (like mine) since no training occurs.

<img src="https://user-images.githubusercontent.com/10279512/57212924-16392000-6f9a-11e9-89fa-4fcd40ee6768.jpeg">

## Reflection

Overall, this project taught me one important lesson: I can't train a model on my laptop. I learned how to use the google compute engine to host my project and train it, all for free thanks to a free trial. However, I was still running this project using 72 CPUs (probably the least efficient way possible) instead of using their built in TPUs or AI Platform. Eventually I will get around to figuring that out.
