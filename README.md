# Recommender-Systems

Here in this project, I have compared two different recommender system methods viz. Matrix Factorization and Neural Network based Matrix Factorization (NeuMF).

I assume that readers have basic idea about collaborative filtering methods in recommender system. Neverthless, let me give you the overview of both the methods and then we will go towards actual results.

## 1. Matrix Factorization

Define a set of Users (U), items (D) such that the Matrix R is of size (|U| X |D|) and includes all the ratings given by users. The goal is to discover K latent features. Given with the input of two matrics matrices P (|U| X k) and Q (|D| X k), it would generate the product result R.

R = P X Transpose(Q)

Matrix P represents the association between a user and the features while matrix Q represents the association between an item and the features.

We can get the prediction of a rating of an item by the calculation of the dot product of the two vectors corresponding to u_i and d_j.

![image](https://user-images.githubusercontent.com/61937357/135051677-a75083b6-3f2e-454d-b1ad-e5a7dbb2533a.png)

To get two entities of both P and Q, we need to initialize the two matrices and calculate the difference of the product named as matrix M. Next, we minimize the difference through gradient descent, aiming at finding a local minimum of the difference.

![image](https://user-images.githubusercontent.com/61937357/135052074-7326262c-1cc6-45e8-8280-c68f30efefe7.png)

To minimize the error, the gradient is able to minimize the error, and therefore we differentiate the above equation with respect to these two variables separately.

![image](https://user-images.githubusercontent.com/61937357/135052192-091925f8-dc22-48a0-a000-b547c8400307.png)

From the gradient, the mathematic formula can be updated for both p_ik and q_kj. Alpha is the step to reach the minimum while the gradient is calculated, and Aplha is usually set with a small value.

![image](https://user-images.githubusercontent.com/61937357/135052513-2a142db0-c10d-4e19-b335-a5ed8fbc4608.png)

From the above equation, p_ik and q_kj can both be updated through iterations until the error converges to its minimum.

![image](https://user-images.githubusercontent.com/61937357/135052713-26e778dd-402b-4533-bbba-0aa133b37a73.png)


## 2. Neural Network based Matrix Factorization

This methods was proposed vide paper Neural Collaborative Filtering by Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua.

Following is the framework for the model:

![image](https://user-images.githubusercontent.com/61937357/135059723-f1dc0d07-9fb4-4a89-b5ab-a3c88620108f.png)

It mainly contains two units viz. Generalized Matrix Factorization (GMF) & Multi-Layer Perceptron (MLP).

say p_u = User embedded latent vector & q_u = Item embedded latent vector

**A. Generalized Matrix Factorization (GMF)**

MF can be interpreted as a framework to mimick colaborative filtering. It basically takes-in the embedded vectors of user and item, perform element-wise multiplication and pass the resultant vector into an activation function (linear or non-linear). Hence it helps in capturing the interactions between user and item latent features, just like in collaborative filtering method.

![image](https://user-images.githubusercontent.com/61937357/135062389-eefddc79-d04e-4dab-9507-77e7da4f8990.png)

**B. Multi-Layer Perceptron (MLP)**

MLP takes in the embedded vectors of user and item, concatenate them and pass the resultant vector into a feedforward neural network. In this sense, we can
endow the model a large level of flexibility and non-linearity to learn the interactions between user and item.

![image](https://user-images.githubusercontent.com/61937357/135062470-fe633d87-7509-40fb-97c6-ca3bb9380957.png)

The above two vectors from GMF and MLP are then concatenated and passed into a final Output layer for prediction. Output layer may be linear for regression task or with non-linear activation function (like sigmoid) for classification task.

Please refer [paper](https://arxiv.org/abs/1708.05031) for more details.


Now readers are encouraged to refer the attached Notebook for fascinating results.
