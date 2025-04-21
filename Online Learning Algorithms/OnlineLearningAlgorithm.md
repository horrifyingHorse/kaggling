> [!TODO]
> Add $$Problem Setting$$ [file:///C:/Users/HP/Downloads/Online_Learning_Algo.pdf](pg.6)

# Examples:
Consider the task of weather forecasting. To predict the temperature or rainfall for the next hour, relying solely on historic data might not suffice. Weather is dynamic, and recent trends—like sudden pressure changes or recent storms—can drastically influence short-term predictions. An online learning model, which updates as new data arrives, can adapt to such shifts more effectively than a model trained once on old data.

Similarly, in road navigation systems, routing from point A to point B should ideally consider current traffic conditions, accidents, or ongoing road construction. A model trained offline may not be aware of real-time changes, whereas an online learning algorithm can incorporate the latest updates—improving decision-making and route efficiency in fast-changing environments.

# References:
* A. [Adam is FTRL in disguise](https://arxiv.org/abs/2402.01567)
* B. [Adam, by Kingma and Ba - 2014](https://arxiv.org/pdf/1412.6980) -> Cited by 211,252
* C. TextBook [Annual Review of Statistics and Its Application: Online Learning Algorithms](https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-040620-035329)
* D. [Introduction to Online Learning Algorithms](https://www.khoury.northeastern.edu/home/vip/teach/MLcourse/OCObook.pdf)
* E. [Lecture 16](https://people.cs.umass.edu/~akshay/courses/cs690m/files/lec16.pdf)
* F. [Online Learning Algorithms](file:///C:/Users/HP/Downloads/Online_Learning_Algo.pdf)

# Online Learning.Presentation
## Introduction
* In this paper - Online Learning Algorithms - specifically focusing on Online Convex Optimization (OCO) techniques
* To unserstand OLA - we need to understand how a ML model is trained.
* Traditional way - we have a dataset - we pick a learning model - train on entire dataset all at once - offline batch training
* Real world scenarios - nothing is static - data arrives continuously, ex: weather forecasting - reatraining model is inefficient if not impossible
* Online Learning - Done incrementally - data is taken one at a time - Sequentially train the model through mini batches (Sequential decision making, [cite:A.]) - Happens in production
* Model on Server trains dynamically - makes it a bit risky too. - The model on production (predicts + learns) unlike offline learning where it only predicts

**Real world Examples**
1. [Weather Forecasting](https://proceedings.mlr.press/v139/flaspohler21a/flaspohler21a.pdf): get latest data from satelites and keep current prediction up to date
2. [Navigation System](https://proceedings.mlr.press/v201/gollapudi23a/gollapudi23a.pdf): find best path from A to B given current traffic situation.
3. [Autonomous Driving Car](https://www.opensourceforu.com/2023/09/ai-and-ml-algorithms-that-power-self-driving-cars/): Steer accordingly with the current surrounding; [Also this](https://mindy-support.com/news-post/how-machine-learning-in-automotive-makes-self-driving-cars-a-reality/#:~:text=Wrap-up,build%20me%20a%20team%20button.)
4. [Stock Prediction](https://ieeexplore.ieee.org/abstract/document/8441038)
5. [Online Advertising]()

## Problem Setting
- Consider an online binary classification task 
- On each round
    - a learner receives a data instance
    - it then makes a prediction of the instance
    - After making the prediction, the learner receives the true answer, ie feedback. (hindsight)
    - Based on the feedback, the learner measures the loss suffered (thru *loss funcs*)
    - Finally, the learner updates its prediction model by some strategy.
- strategy - to improve predictive performance on future received instances.

### **Regret**: Measure of Quality in Online Learning
> it measures how much the algos(agent's) chosen actions deviate from the optimal actions in retrospect.

**Sublinear Regret**: means that the regret grows slower than linearly with the number of time steps. This indicates that the performance improves over time, and its decision-making becomes increasingly closer to the optimal strategy.
> if an online algorithm guarantees that its regret is sublinear as a function of T, i.e., RT = o(T), it implies that limT→∞ R(T)/T = 0 and thus on average the learner performs almost as well as the best fixed model in hindsight.

    Rt(a) = Sum(t=1, T)L(at, zt) - Sum(t=1, T)L(a, zt)

This value is called the (cumulative)regret of a learner wrt an action a (belongs to) A
* Sum(t=1, T)L(at, zt): Cumulative loss of the learner
* Sum(t=1, T)L(a, zt):  Cumulative loss of the competing action (loss suffered by the optimal model a that can only be known in hindsight after seeing all the instances and their class labels)

In general, from a theoretical perspective, online learning methodologies are founded based on theory and principles from three major theory communities: learning theory, optimization theory, and game theory.[cite:F.](pg.4:1.3:p1l3)

### Online Convex Optimization
* Many online learning problems can essentially be (re-)formulated as an Online Convex Optimization (OCO) task.[cite:F.](pg.8:2.3)
* An online convex optimization task consists of two major elements:
    1. convex set Cal(S)
    2. convex cost function lt(·)
* At each time step t, the online algorithm:
    - chooses a weight vector wt ∈ Cal(S)
    - it suffers a loss lt(wt) - computed based on a convex cost function lt(·) defined over S
The goal of the online algorithm is to choose a sequence of decisions w1, w2, . . . such that the regret in hindsight can be minimized.

---

# Online Learning.1
[Open-Meteo](https://open-meteo.com)

* OL is a Learning paradigm and not a specific type of model, like CNN or RNN.

**Examples** : Real world applications
1. [Weather Forecasting](https://proceedings.mlr.press/v139/flaspohler21a/flaspohler21a.pdf): get latest data from satelites and keep current prediction up to date
2. Sequential Investment: (done by banks)
3. [Navigation System](https://proceedings.mlr.press/v201/gollapudi23a/gollapudi23a.pdf): find best path from A to B given current traffic situation.
4. [Autonomous Driving Car](https://www.opensourceforu.com/2023/09/ai-and-ml-algorithms-that-power-self-driving-cars/): Steer accordingly with the current surrounding; [Also this](https://mindy-support.com/news-post/how-machine-learning-in-automotive-makes-self-driving-cars-a-reality/#:~:text=Wrap-up,build%20me%20a%20team%20button.)
5. [Stock Prediction](https://ieeexplore.ieee.org/abstract/document/8441038)

### Measure of Quality in Online Learning
**Regret**: 
*it measures how much the agent's chosen actions deviate from the optimal actions in retrospect.*

**Sublinear Regret**: means that the regret grows slower than linearly with the number of time steps. This indicates that the agent's performance improves over time, and its decision-making becomes increasingly closer to the optimal strategy.

    Rt(a) = Sum(t=1, T)L(at, zt) - Sum(t=1, T)L(a, zt)

This value is called the (cumulative)regret of a learner wrt an action a (belongs to) A
* Sum(t=1, T)L(at, zt): Cumulative loss of the learner
* Sum(t=1, T)L(a, zt):  Cumulative loss of the competing action

Objective of OLA is to minimize the regret

---

[cite:E.]
1. Hedge (Weighted Majority Algorithm)
2. Exponential Weights / Exponential Gradient

# Online Learning.2
Online learning is a framework for the design and analysis of algorithms that build predictive models by processing data one at the time. Besides being computationally efficient, online algorithms enjoy theoretical performance guarantees that do not rely on statistical assumptions on the data source [cite:C.]

1. Done incrementally
2. Sequentially train the model through mini batches (Sequential decision making, [cite:A.])
3. Happens in production
4. Model on Server trains dynamically
5. The model on production (predicts + learns) unlike offline learning where it only predicts

## Examples:
1. Chatbots
2. Youtube feed suggestions (recommendations)
3. Online advertising

### Online Convex Optimization
* The standard framework for the study of parametric online learning with convex losses is known as online convex optimization (OCO) [cite:C.]
* Many online learning problems can essentially be (re-)formulated as an Online Convex Optimization (OCO) task.[cite:F.](pg.8:2.3)

## When to use?
### Where there is concept drift.
1. the nature of problem is changing (its `volatile`)
Examples: 

    Stock Exchange
    E commerce platforms (Sales depending on festivals, seasonal changes in user buying preferences)
    Economics, finanace, health where new data patterns are constantly emrging

### Cost effective (https://arxiv.org/pdf/2310.04216 Pg 8)
### Faster Solution

Libs for online learning:
1. River
2. Vowpal Wabbit

## Learning Rate:

## Out of Core Learning:

## Disadvantage:
1. Risky: since it happens on production, if data set corrupts, then the model becoms biased

## Taxonomy:
Theoretically, Online Learning Algorithms are based on 3 major foundational ares:
1. Learning Theory
2. Optimization Theory
3. Game Theory

## Categories of Online Learning:
1. Online Supervised  Learning
2. Online Learning with Limited Feedback
3. Online Unsupervised Learning

# Perceptron
[Uni of Washington](https://courses.cs.washington.edu/courses/cse446/17wi/slides/online-perceptron-annotated.pdf)
[another uni](https://www.cs.cmu.edu/~avrim/ML14/lect0122.pdf)
for online binary classification, works only if data is linearly separable
(Single Layer Perceptron -> Step activation function)
1. **Input Layer** -> each representing feature of the data.
2. **Weights** -> Each i/p is assigned some weight that represents the importance in decision making process.
3. **Bias** -> constant term added to weighted sum of i/ps, helping the model to shift decision boundary.
4. **Activation Function** -> perceptron's decision (Threshold)

*Multilayer Perceptron Application*: 
1. [An Optimal Multilayer Perceptron with Dragonfly Algorithm for Intrusion Detection in Wireless Sensor Networks](https://ieeexplore.ieee.org/document/9418355)
2. [A Novel Method for Predicting the Flight Price using Multilayer Perceptron Algorithm Compared with Support Vector Machine Algorithm](https://ieeexplore.ieee.org/document/10568803)
3. [An Evolutionary MultiLayer Perceptron Algorithm for Real Time River Flood Prediction](https://ieeexplore.ieee.org/document/9161824)

# Passive Aggresive Algorithms
Pretty similar to Perceptron,
When a new data point arrives, the PA algo could behave in 2 ways:
* Passive: if the data point is correctly classified by the existing model, no updations (hinge loss is 0)
* Aggresive: if the data point is incorrectly classified, we make updations aggressively. (hinge loss is +ve)

> There also exists Passive Aggressive Regression, interesting, and the one we using is called PA Classifier.
> PA1 -> Hinge Loss
> PA2 -> Sqaured Hinge Loss

*Application*
1. [Authentic News Prediction in Machine Learning using Passive AggressiveAlgorithm](https://ieeexplore.ieee.org/document/9743010)
2. [Online Passive-Aggressive Multilabel Classification Algorithms](https://ieeexplore.ieee.org/document/9758938)
3. [Fake News Identification for Web Scrapped Data using Passive Aggressive Classifier](https://ieeexplore.ieee.org/document/10275818)
4. [Online Passive-Aggressive Active Learning for Trapezoidal Data Streams](https://ieeexplore.ieee.org/document/9791447) -> for binary and multiclass online classification tasks on trapezoidal data streams where the feature space may expand over time

# Follow the Leader
* Simple and popular algo in game theory
* players play the game, in each round, they choose w, suffer loss
* Their aim is to minimize the regret in comparison to best decision in hindsight
* good strategy: sublinear regret = o(T)
* Simplest approach is to choose at any time optimal decision in hindsight

*Application*
1. [Ad Click Prediction: a View from the Trenches](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf)

# Gradient Descend
-> backbone of Deep Learning! also used in different algos like TSNE, Linear Regression
*Optimization algorithm* to find the minimum of a function.
- our aim here is to minimise the cost function.
- Start with a random point on the function and then move in the direction opposite to that of gradient of the function to reach minimum.
- Assumption: Loss function is convex and differentiable
- it takes huge steps when it is far away from the minimum, and then takes mini steps when its closer! (this is because the slope decreases)

for Big Data, GD is slow

# Stochastic Gradient Descent
-> Relatively faster than GD (less epochs)
-> Hardware reqs won't fail, ie not entire tarining data requied in RAM unlike GD
It updates based on any randomly selected data point instead of entire data!
- Its stochastic, thus does not give a single solution every time we run (since random selection)
Use when:
1. Big Data
2. Non convex loss function

**Learning Schedules** -> help in varying the learning rate(with epochs) so that fluctuations in SGD become minimal

# AdaGrad (Adaptive Gradient)
* learning rate is not fixed (adaptive according to situation) (learning rate decays)
performs better when:
1. input features have large scale
2. features are sparse (most of the values are 0) (when you have elongated bowl problem: visualise _contour plot_)
* Mostly used in linear regression (works fine with convex)
*Disadvantage*:
1. Learning rate is decreasing, thus does not converge with minima! (for non convex)

# RMSProp (Root Mean Square Propogation)
Optimisation Technique
* Improvement over AdaGrad, solves its con, this coverges for a non convex optimisation
* Recent epochs have more effect on learning rate than those occured earlier.
* _No Disadvantages!_ -> Before the arrival of Adam, this was the most used optimisation technique used to optimize neural network!

# Adam (Adaptive Moment Estimation)
* Implements both momentum approach and learnin rate decay to maximize optimization!
