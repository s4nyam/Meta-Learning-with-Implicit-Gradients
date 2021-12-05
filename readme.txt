To run the codes follow the steps below:

Please note that author codes can be run directly using the github repository along with the steps mentioned in the website. (with attached readme.txt in original author code)

Jupyter Notebook is explained here as follows:

1. initialize number of tasks i.e number of tasks we need in each batch of tasks
2. number of samples i.e number of shots  -number of data points (k) we need to have in each task
3. number of epochs i.e training iterations
4. hyperparameter for the inner loop (inner gradient update)
5. hyperparameter for the outer loop (outer gradient update) i.e meta optimization
6. randomly initialize our model parameter theta
7. define our sigmoid activation function
8. now let us get to the interesting part i.e training :P
9. for the number of epochs
10. for task i in batch of tasks
11. sample k data points and prepare our train set
12. since we are performing classification, we use cross entropy loss as our loss function
13. minimize the loss by calculating gradients
14. update the gradients and find the optimal parameter theta' for each of tasks
15. initialize meta gradients
16. sample k data points and prepare our test set for meta training
17. predict the value of y
18. compute meta gradients
19. update our randomly initialized model parameter theta with the meta gradients


In the last section we saw how MAML works. We saw how MAML obtains a better and robust model parameter  𝜃  that is generalisable across tasks.

Now we will better understand MAML by coding them from scratch. For better understanding, we consider a simple binary classification task. We randomly generate our input data and we train them with a simple single layer neural network and try to find the optimal parameter theta.

Now we will step by step how exactly we are doing this,

First we import all the necessary libraries,


Gradient Checkpointing Model-Agnostic Meta-Learning
We demonstrate how to use memory efficient MAML on CIFAR10. This notebook performs one forward and backward for MAML with a large number of iterations

Data: Random tensors (batch_size, 3, 224, 224)
Model: ResNet18
Optimizer: SGD with 0.01 learning rate
Batch size: 16
MAML steps: 100 (works with >500 on 11GB GPU)
GPU: whatever colab has to spare, probably K80



First, we set such max steps that fits memory for naive MAML to check the implementation

Clip meta-learning gradients by global norm to avoid explosion

Generate batch for demonstration. Note that we support using different batches for each MAML step (a-la SGD)


Now, we define a class called MAML where we implement the MAML algorithm. In the _init_ method we will initialise all the necessary variables. Then we define our sigmoid activation function. Followed by we define our train function.

You can check the comments written above each line of code for understanding.


So MAML works much better than transfer learning or random initialization for this problem. Yay!

However, it is a bit annoying that we have to use second order derivatives for this... it forces the code to be complicated and it also makes things a fair bit slower (around 33% according to the paper, which matches what we shall see here).

Is there an approximation of MAML that doesn't use the second order derivatives? Of course, we can simply pretend that the gradients that we used for the inner gradient descent just came out of nowhere, and thus just improve the initial parameters without taking into account these second order derivatives, which is what we did before by handling the first_order parameter.

So how good is this first order approximation? Almost as good as the original MAML, as it turns out!



