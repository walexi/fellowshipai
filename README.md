# Edge-labeling Graph Neural Network for one-shot learning (WIP)

This is a challenge submission for 
[fellowship.ai](https://fellowship.ai) cohort 15.

The challenge is to perform 1-shot learning on the [Omniglot 
dataset](https://github.com/brendenlake/omniglot).

## Omniglot Dataset 
- The Omniglot data set is designed for developing more human-like learning algorithms. 
- It contains 1623 different handwritten characters from 50 different alphabets. 
- Each of the 1623 characters was drawn online via Amazon's Mechanical Turk by 20 different people. 
- Each image is paired with stroke data, a sequences of [x,y,t] coordinates with time (t) in milliseconds.

## Challenge Goal/Objective/Approach
- Use background set of 30 alphabets for training and evaluate on set of 20 alphabets.
- Report one-shot classification (20-way) results using a meta learning approach like [MAML](https://arxiv.org/pdf/1703.03400.pdf).
- Some basic exploration, 1-NN experiment and demo run from dataset authors using a distance metric are in 

## One-shot Learning and Meta-learning
- One-shot learning is the task of learning information about object categories from a single training example.
- Each classification task T contains a support set S, a labeled set of input-label pairs, and a query set Q, an unlabeled set on which the learned classifier is evaluated.
- For a 20-way 1-shot cassification, the support set S contains K=1 labeled sample for each of N=20 unique classes
- Earlier work on one-shot learning is largely based on hand-engineered features
- With meta-learning (also known as learn to learn), we hope to learn a common feature representation for all tasks in an end-to-end fashion

## Existing solutions and current benchmarks
- [Lake et. al](http://science.sciencemag.org/content/350/6266/1332) use Bayesian Program learning to achieve an error rate < 5% on the 20-way one-shot classification task. This method makes use of the stroke information. For a Siamese ConvNet they report an error of < 10%.
- [Jake Snell et. al](https://arxiv.org/abs/1703.05175) use Prototypical networks (which map examples to a pdimensional vector space such that examples of a given output class are close together) to learn a metric space in which classification can be performed by computing distances to prototype representations of each class. They report an accuracy rate of 96.0% on the 20-way one-shot classification task on the Omniglot dataset
- [Vinyals et al](https://arxiv.org/pdf/1606.04080.pdf) use a differentiable nearest neighbours classifier (non-parametric approach). Accuracy of 93.8% and 93.2% were reported on the 20-way one-shot classification task on the Omniglot dataset respectively
- [Alex Nichol et al](https://arxiv.org/pdf/1803.02999v3.pdf) use First-Order Meta-Learning Algorithms (an approximation to MAML obtained by ignoring second-order derivatives). Accuracy of 89.4%, and 89.43% were reported on the 20-way one-shot classification task on the Omniglot dataset using First-order MAML (using Transduction) and a new algorithm Reptile (using Transduction) respectively

## Solution
- Edge-Labeling Graph Neural Network for Few-shot Learning
- It consists of a number of layers where each layer consists of a node-update block and an edge-update block
- The parameters of the network (for each layer) are learned by episodic training
- Updates edge labels as it utilizes both intra-cluster similarity and inter-cluster dissimilarity
- It can be easily extended to perform a transductive inference

## Possible improvements
- Use another loss such as the [cycle loss](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.371.1288&rep=rep1&type=pdf).

## Reference Papers
- [Edge-Labeling Graph Neural Network for Few-shot Learning](https://arxiv.org/pdf/1905.01436.pdf) Bryan Perozzi et al
<!--
## Challenge goals
1. Problem solving ability - did you understand the problem correctly, 
and did you take logical steps to solve it?  
2. Machine learning skills - what sort of models did you use? How 
rigorous was your exploratory analysis of the data, your choice and fine 
tuning of models, and your assessment of results.  
3. Communication skills - is your solution readable and well explained? 
Messiness and raw code with no explanation does not reflect well on your 
potential for working well with our business partners during the 
fellowship.

## Mistakes to avoid
- Skipping exploratory analysis and feature engineering  
Do not jump straight into fitting models without demonstrating to us, in 
your Jupyter notebook, that you have understood and thought about the 
dataset.

- Choosing models with no explanation  
Please use the notebook to explain your thought process. We care about 
this as much as we care about your results.

- Unreadable notebooks  
Make sure to run your notebook before sharing so that we can see the 
results. We won't be running your code on our machines. On the flip 
side, please do not print out the entire dataset or endless rounds of 
epochs.

- Overly simplistic final results  
Your final results should consist of more than a single number or 
percentage printout. Explain why you chose the success metrics you 
chose, and analyze what your output means.


## Questions to Consider
Ask yourself why would they have selected this problem for the 
challenge? What are some gotchas in this domain I should know about?  
What is the highest level of accuracy that others have achieved with 
this dataset or similar problems / datasets ?  
What types of visualizations will help me grasp the nature of the 
problem / data?  
What feature engineering might help improve the signal?  
Which modeling techniques are good at capturing the types of 
relationships I see in this data?  
Now that I have a model, how can I be sure that I didn't introduce a bug 
in the code? If results are too good to be true, they probably are!  
What are some of the weaknesses of the model and and how can the model 
be improved with additional work? -->

