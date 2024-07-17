import numpy as np
import math

# -------------------------------------------------------------------------
'''
    Problem 1: softmax regression 
    In this problem, you will implement the softmax regression for multi-class classification problems.
    The main goal of this problem is to extend the logistic regression method to solving multi-class classification problems.
    We will get familiar with computing gradients of vectors/matrices.
    We will use multi-class cross entropy as the loss function and stochastic gradient descent to train the model parameters.
    You could test the correctness of your code by typing `pytest -v test1.py` in the terminal.

    Notations:
            ---------- input data ----------------------
            p: the number of input features, an integer scalar.
            c: the number of classes in the classification task, an integer scalar.
            x: the feature vector of a data instance, a float numpy array of shape (p, ). 
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).

            ---------- model parameters ----------------------
            W: the weight matrix of softmax regression, a float numpy array of shape (c, p). 
            b: the bias values of softmax regression, a float numpy array of shape (c, ).
            ---------- values ----------------------
            z: the linear logits, a float numpy array of shape (c, ).
            a: the softmax activations, a float numpy array of shape (c, ). 
            L: the multi-class cross entropy loss, a float scalar.

            ---------- partial gradients ----------------------
            dL_da: the partial gradients of the loss function L w.r.t. the activations a, a float numpy array of shape (c, ). 
                   The i-th element dL_da[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a[i]:  d_L / d_a[i].
            da_dz: the partial gradient of the activations a w.r.t. the logits z, a float numpy array of shape (c, c). 
                   The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] )
            dz_dW: the partial gradient of logits z w.r.t. the weight matrix W, a float numpy array of shape (c, p). 
                   The (i,j)-th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[i,j]:   d_z[i] / d_W[i,j]
            dz_db: the partial gradient of the logits z w.r.t. the biases b, a float matrix of shape (c, ). 
                   Each element dz_db[i] represents the partial gradient of the i-th logit z[i] w.r.t. the i-th bias b[i]:  d_z[i] / d_b[i]

            ---------- partial gradients of parameters ------------------
            dL_dW: the partial gradients of the loss function L w.r.t. the weight matrix W, a float numpy array of shape (c, p). 
                   The i,j-th element dL_dW[i,j] represents the partial gradient of the loss function L w.r.t. the i,j-th weight W[i,j]:  d_L / d_W[i,j]
            dL_db: the partial gradient of the loss function L w.r.t. the biases b, a float numpy array of shape (c, ).
                   The i-th element dL_db[i] represents the partial gradient of the loss function w.r.t. the i-th bias:  d_L / d_b[i]

            ---------- training ----------------------
            alpha: the step-size parameter of gradient descent, a float scalar.
            n_epoch: the number of passes to go through the training dataset in order to train the model, an integer scalar.
'''


# -----------------------------------------------------------------
# Forward Pass
# -----------------------------------------------------------------

# -----------------------------------------------------------------
def compute_z(x, W, b):
    '''
        Compute the linear logit values of a data instance. z =  W x + b
        Input:
            x: the feature vector of a data instance, a float numpy array of shape (p, ). Here p is the number of features/dimensions.
            W: the weight matrix of softmax regression, a float numpy array of shape (c, p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of shape (c, ).
        Output:
            z: the linear logits, a float numpy vector of shape (c, ).
        Hint: you could solve this problem using 1 line of code.
    '''
    return np.dot(W, x) + b


# -----------------------------------------------------------------
def compute_a(z):
    '''
        Compute the softmax activations.
        Input:
            z: the logit values of softmax regression, a float numpy vector of shape (c, ). Here c is the number of classes
        Output:
            a: the softmax activations, a float numpy vector of shape (c, ).
    '''
    z_shifted = z - np.max(z)
    exp_z = np.exp(np.clip(z_shifted, -500, 500))  # Clip to avoid overflow
    a = exp_z / np.sum(exp_z)
    return a


def compute_L(a, y):
    '''
        Compute multi-class cross entropy, which is the loss function of softmax regression.
        Input:
            a: the activations of a training instance, a float numpy vector of shape (c, ). Here c is the number of classes.
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
        Output:
            L: the loss value of softmax regression, a float scalar.
    '''
    epsilon = 1e-15  # Use a very small epsilon to prevent log(0)
    a_y = np.clip(a[y], epsilon, 1 - epsilon)  # Ensure a[y] is properly clipped

    if isinstance(a_y, np.ndarray):
        if a_y.size == 1:
            a_y = a_y.item()  # Ensure a_y is a scalar
        else:
            raise ValueError("a_y should be a scalar or a numpy array of size 1")

    L = float(-np.log(a_y))

    if a_y == epsilon:
        L = 1e6  # Return a very high value if a_y is extremely small
    return L







# -----------------------------------------------------------------
def forward(x, y, W, b):
    '''
       Forward pass: given an instance in the training data, compute the logits z, activations a and multi-class cross entropy L on the instance.
        Input:
            x: the feature vector of a training instance, a float numpy vector of shape (p, ). Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
            W: the weight matrix of softmax regression, a float numpy array of shape (c, p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of shape (c, ).
        Output:
            z: the logit values of softmax regression, a float numpy vector of shape (c, ). Here c is the number of classes
            a: the activations of a training instance, a float numpy vector of shape (c, ). Here c is the number of classes.
            L: the loss value of softmax regression, a float scalar.
    '''
    z = compute_z(x, W, b)
    a = compute_a(z)
    L = compute_L(a, y)
    return z, a, L


# -----------------------------------------------------------------
# Compute Local Gradients
# -----------------------------------------------------------------


# -----------------------------------------------------------------
def compute_dL_da(a, y):
    '''
        Compute local gradient of the multi-class cross-entropy loss function w.r.t. the activations.
        Input:
            a: the activations of a training instance, a float numpy vector of shape (c, ). Here c is the number of classes.
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
        Output:
            dL_da: the local gradients of the loss function w.r.t. the activations, a float numpy vector of shape (c, ).
                   The i-th element dL_da[i] represents the partial gradient of the loss function w.r.t. the i-th activation a[i]:  d_L / d_a[i].
    '''
    dL_da = np.zeros_like(a)
    dL_da[y] = -1 / (a[y] + 1e-15)  # Add small epsilon to avoid division by zero
    return dL_da


# -----------------------------------------------------------------
def compute_da_dz(a):
    '''
        Compute local gradient of the softmax activations a w.r.t. the logits z.
        Input:
            a: the activation values of softmax function, a numpy float vector of shape (c, ). Here c is the number of classes.
        Output:
            da_dz: the local gradient of the activations a w.r.t. the logits z, a float numpy array of shape (c, c).
                   The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] )
        Hint: you could solve this problem using 4 or 5 lines of code.
        (3 points)
    '''
    return np.diag(a) - np.outer(a, a)


# -----------------------------------------------------------------
def compute_dz_dW(x, c):
    '''
        Compute local gradient of the logits function z w.r.t. the weights W.
        Input:
            x: the feature vector of a data instance, a float numpy vector of shape (p, ). Here p is the number of features/dimensions.
            c: the number of classes, an integer.
        Output:
            dz_dW: the partial gradient of logits z w.r.t. the weight matrix, a float numpy array of shape (c, p).
                   The (i,j)-th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[i,j]:   d_z[i] / d_W[i,j]
        Hint: the partial gradients only depend on the input x and the number of classes
    '''
    return np.tile(x, (c, 1))


# -----------------------------------------------------------------
def compute_dz_db(c):
    '''
        Compute local gradient of the logits function z w.r.t. the biases b.
        Input:
            c: the number of classes, an integer.
        Output:
            dz_db: the partial gradient of the logits z w.r.t. the biases b, a float vector of shape (c, ).
                   Each element dz_db[i] represents the partial gradient of the i-th logit z[i] w.r.t. the i-th bias b[i]:  d_z[i] / d_b[i]
        Hint: you could solve this problem using 1 line of code.
    '''
    return np.ones(c)


# -----------------------------------------------------------------
# Back Propagation
# -----------------------------------------------------------------

# -----------------------------------------------------------------
def backward(x, y, a):
    '''
       Back Propagation: given an instance in the training data, compute the local gradients of the logits z, activations a, weights W and biases b on the instance.
        Input:
            x: the feature vector of a training instance, a float numpy vector of shape (p, ). Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
            a: the activations of a training instance, a float numpy vector of shape (c, ). Here c is the number of classes.
        Output:
            dL_da: the local gradients of the loss function w.r.t. the activations, a float numpy vector of shape (c, ).
                   The i-th element dL_da[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a[i]:  d_L / d_a[i].
            da_dz: the local gradient of the activation w.r.t. the logits z, a float numpy array of shape (c, c).
                   The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] )
            dz_dW: the partial gradient of logits z w.r.t. the weight matrix W, a float numpy array of shape (c, p).
                   The i,j -th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[i,j]:   d_z[i] / d_W[i,j]
            dz_db: the partial gradient of the logits z w.r.t. the biases b, a float vector of shape (c, ).
                   Each element dz_db[i] represents the partial gradient of the i-th logit z[i] w.r.t. the i-th bias:  d_z[i] / d_b[i]
    '''
    dL_da = compute_dL_da(a, y)
    da_dz = compute_da_dz(a)
    dz_dW = compute_dz_dW(x, len(a))
    dz_db = compute_dz_db(len(a))
    return dL_da, da_dz, dz_dW, dz_db


# -----------------------------------------------------------------
def compute_dL_dz(dL_da, da_dz):
    '''
       Given the local gradients, compute the gradient of the loss function L w.r.t. the logits z using chain rule.
        Input:
            dL_da: the local gradients of the loss function w.r.t. the activations, a float numpy vector of shape (c, ).
                   The i-th element dL_da[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a[i]:  d_L / d_a[i].
            da_dz: the local gradient of the activation w.r.t. the logits z, a float numpy array of shape (c, c).
                   The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] )
        Output:
            dL_dz: the gradient of the loss function L w.r.t. the logits z, a numpy float vector of shape (c, ).
                   The i-th element dL_dz[i] represents the partial gradient of the loss function L w.r.t. the i-th logit z[i]:  d_L / d_z[i].
    '''
    return np.dot(dL_da.T, da_dz)


# -----------------------------------------------------------------
def compute_dL_dW(dL_dz, dz_dW):
    '''
       Given the local gradients, compute the gradient of the loss function L w.r.t. the weights W using chain rule.
        Input:
            dL_dz: the gradient of the loss function L w.r.t. the logits z, a numpy float vector of shape (c, ).
                   The i-th element dL_dz[i] represents the partial gradient of the loss function L w.r.t. the i-th logit z[i]:  d_L / d_z[i].
            dz_dW: the partial gradient of logits z w.r.t. the weight matrix W, a float numpy array of shape (c, p).
                   The i,j -th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[i,j]:   d_z[i] / d_W[i,j]
        Output:
            dL_dW: the global gradient of the loss function w.r.t. the weight matrix, a float numpy array of shape (c, p).
                   Here c is the number of classes.
                   The i,j-th element dL_dW[i,j] represents the partial gradient of the loss function L w.r.t. the i,j-th weight W[i,j]:  d_L / d_W[i,j]
    '''
    return np.outer(dL_dz, dz_dW[0])


# -----------------------------------------------------------------
def compute_dL_db(dL_dz, dz_db):
    '''
       Given the local gradients, compute the gradient of the loss function L w.r.t. the biases b using chain rule.
        Input:
            dL_dz: the gradient of the loss function L w.r.t. the logits z, a numpy float vector of shape (c, ).
                   The i-th element dL_dz[i] represents the partial gradient of the loss function L w.r.t. the i-th logit z[i]:  d_L / d_z[i].
            dz_db: the local gradient of the logits z w.r.t. the biases b, a float numpy vector of shape (c, ).
                   The i-th element dz_db[i] represents the partial gradient ( d_z[i]  / d_b[i] )
        Output:
            dL_db: the global gradient of the loss function L w.r.t. the biases b, a float numpy vector of shape (c, ).
                   The i-th element dL_db[i] represents the partial gradient of the loss function w.r.t. the i-th bias:  d_L / d_b[i]
        Hint: you could solve this problem using 1 line of code in the block.
    '''
    return dL_dz


# -----------------------------------------------------------------
# gradient descent
# -----------------------------------------------------------------

# --------------------------
def update_W(W, dL_dW, alpha=0.001):
    '''
       Update the weights W using gradient descent.
        Input:
            W: the current weight matrix, a float numpy array of shape (c, p). Here c is the number of classes.
            alpha: the step-size parameter of gradient descent, a float scalar.
            dL_dW: the global gradient of the loss function w.r.t. the weight matrix, a float numpy array of shape (c, p).
                   The i,j-th element dL_dW[i,j] represents the partial gradient of the loss function L w.r.t. the i,j-th weight W[i,j]:  d_L / d_W[i,j]
        Output:
            W: the updated weight matrix, a float numpy array of shape (c, p).
        Hint: you could solve this problem using 1 line of code
    '''
    return W - alpha * dL_dW


# --------------------------
def update_b(b, dL_db, alpha=0.001):
    '''
       Update the biases b using gradient descent.
        Input:
            b: the current bias values, a float numpy vector of shape (c, ).
            dL_db: the global gradient of the loss function w.r.t. the biases b, a float numpy vector of shape (c, ).
                   The i-th element dL_db[i] represents the partial gradient of the loss function w.r.t. the i-th bias:  d_L / d_b[i]
            alpha: the step-size parameter of gradient descent, a float scalar.
        Output:
            b: the updated of bias vector, a float numpy vector of shape (c, ).
        Hint: you could solve this problem using 2 lines of code
    '''
    return b - alpha * dL_db


# --------------------------
# train
def train(X, Y, alpha=0.01, n_epoch=1000):
    '''
       Given a training dataset, train the softmax regression model by iteratively updating the weights W and biases b using the gradients computed over each data instance.
        Input:
            X: the feature matrix of training instances, a float numpy array of shape (n, p). Here n is the number of data instance in the training set, p is the number of features/dimensions.
            Y: the labels of training instance, a numpy integer numpy array of length n. The values can be 0 or 1.
            alpha: the step-size parameter of gradient ascent, a float scalar.
            n_epoch: the number of passes to go through the training set, an integer scalar.
        Output:
            W: the weight matrix trained on the training set, a float numpy array of shape (c, p).
            b: the bias, a float numpy vector of shape (c, ).
    '''
    # number of features
    N, p = X.shape
    # number of classes
    c = max(Y) + 1

    # randomly initialize W and b
    W = np.random.rand(c, p) * 0.01  # Initialize with small random values
    b = np.zeros(c)

    for _ in range(n_epoch):
        # go through each training instance
        for i in range(N):
            x, y = X[i], Y[i]

            # Forward pass
            z, a, _ = forward(x, y, W, b)

            # Backward pass
            dL_da, da_dz, dz_dW, dz_db = backward(x, y, a)

            # Compute gradients
            dL_dz = compute_dL_dz(dL_da, da_dz)
            dL_dW = compute_dL_dW(dL_dz, dz_dW)
            dL_db = compute_dL_db(dL_dz, dz_db)

            # Update parameters
            W -= alpha * dL_dW
            b -= alpha * dL_db
            #########################################
    return W, b


# --------------------------
def predict(Xtest, W, b):
    '''
       Predict the labels of the instances in a test dataset using softmax regression.
        Input:
            Xtest: the feature matrix of testing instances, a float numpy array of shape (n_test, p). Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
            W: the weight vector of the logistic model, a float numpy array of shape (c, p).
            b: the bias values of the softmax regression model, a float vector of shape (c, ).
        Output:
            Y: the predicted labels of test data, an integer numpy array of length ntest Each element can be 0, 1, ..., or (c-1)
            P: the predicted probabilities of test data to be in different classes, a float numpy array of shape (ntest,c). Each (i,j) element is between 0 and 1, indicating the probability of the i-th instance having the j-th class label.
        (2 points)
    '''
    n = Xtest.shape[0]
    c = W.shape[0]
    Y = np.zeros(n, dtype=int)  # Initialize Y as integer array
    P = np.zeros((n, c))  # Initialize P with correct shape
    for i, x in enumerate(Xtest):
        z = compute_z(x, W, b)
        a = compute_a(z)
        Y[i] = np.argmax(a)
        P[i] = a
    return Y, P


# -----------------------------------------------------------------
# gradient checking
# -----------------------------------------------------------------


# -----------------------------------------------------------------
def check_da_dz(z, delta=1e-7):
    '''
        Compute local gradient of the softmax function using gradient checking.
        Input:
            z: the logit values of softmax regression, a float numpy vector of shape (c, ). Here c is the number of classes
            delta: a small number for gradient check, a float scalar.
        Output:
            da_dz: the approximated local gradient of the activations w.r.t. the logits, a float numpy array of shape (c, c).
                   The (i,j)-th element represents the partial gradient ( d a[i]  / d z[j] )
    '''
    c = z.shape[0]  # number of classes
    da_dz = np.zeros((c, c))
    for i in range(c):
        for j in range(c):
            d = np.zeros(c)
            d[j] = delta
            da_dz[i, j] = (compute_a(z + d)[i] - compute_a(z)[i]) / delta
    return da_dz


# -----------------------------------------------------------------
def check_dL_da(a, y, delta=1e-7):
    '''
        Compute local gradient of the multi-class cross-entropy function w.r.t. the activations using gradient checking.
        Input:
            a: the activations of a training instance, a float numpy vector of shape (c, ). Here c is the number of classes.
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_da: the approximated local gradients of the loss function w.r.t. the activations, a float numpy vector of shape (c, ).
    '''
    c = a.shape[0]  # number of classes
    dL_da = np.zeros(c)  # initialize the vector as all zeros
    for i in range(c):
        d = np.zeros(c)
        d[i] = delta
        dL_da[i] = (compute_L(a + d, y) - compute_L(a, y)) / delta
    return dL_da


# --------------------------
def check_dz_dW(x, W, b, delta=1e-7):
    '''
        compute the local gradient of the logit function using gradient check.
        Input:
            x: the feature vector of a data instance, a float numpy vector of shape (p, ). Here p is the number of features/dimensions.
            W: the weight matrix of softmax regression, a float numpy array of shape (c, p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of shape (c, ).
            delta: a small number for gradient check, a float scalar.
        Output:
            dz_dW: the approximated local gradient of the logits w.r.t. the weight matrix computed by gradient checking, a float numpy array of shape (c, p).
                   The i,j -th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[i,j]:   d_z[i] / d_W[i,j]
    '''
    c, p = W.shape  # number of classes and features
    dz_dW = np.zeros((c, p))
    for i in range(c):
        for j in range(p):
            d = np.zeros((c, p))
            d[i, j] = delta
            dz_dW[i, j] = (compute_z(x, W + d, b)[i] - compute_z(x, W, b)[i]) / delta
    return dz_dW


# --------------------------
def check_dz_db(x, W, b, delta=1e-7):
    '''
        compute the local gradient of the logit function using gradient check.
        Input:
            x: the feature vector of a data instance, a float numpy vector of shape (p, ). Here p is the number of features/dimensions.
            W: the weight matrix of softmax regression, a float numpy array of shape (c, p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of shape (c, ).
            delta: a small number for gradient check, a float scalar.
        Output:
            dz_db: the approximated local gradient of the logits w.r.t. the biases using gradient check, a float vector of shape (c, ).
                   Each element dz_db[i] represents the partial gradient of the i-th logit z[i] w.r.t. the i-th bias b[i]:  d_z[i] / d_b[i]
    '''
    c, _ = W.shape  # number of classes and features
    dz_db = np.zeros(c)
    for i in range(c):
        d = np.zeros(c)
        d[i] = delta
        dz_db[i] = (compute_z(x, W, b + d)[i] - compute_z(x, W, b)[i]) / delta
    return dz_db


# -----------------------------------------------------------------
def check_dL_dW(x, y, W, b, delta=1e-7):
    '''
       Compute the gradient of the loss function w.r.t. the weights W using gradient checking.
        Input:
            x: the feature vector of a training instance, a float numpy vector of shape (p, ). Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
            W: the weight matrix of softmax regression, a float numpy array of shape (c, p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of shape (c, ).
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_dW: the approximated gradients of the loss function w.r.t. the weight matrix, a float numpy array of shape (c, p).
    '''
    c, p = W.shape
    dL_dW = np.zeros((c, p))
    for i in range(c):
        for j in range(p):
            d = np.zeros((c, p))
            d[i, j] = delta
            dL_dW[i, j] = (forward(x, y, W + d, b)[-1] - forward(x, y, W, b)[-1]) / delta
    return dL_dW


# -----------------------------------------------------------------
def check_dL_db(x, y, W, b, delta=1e-7):
    '''
    Compute the gradient of the loss function w.r.t. the bias b using gradient checking.
    Input:
        x: the feature vector of a training instance, a float numpy vector of shape (p, 1). Here p is the number of features/dimensions.
        y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
        W: the weight matrix of softmax regression, a float numpy array of shape (c, p). Here c is the number of classes.
        b: the bias values of softmax regression, a float numpy vector of shape (c, 1).
        delta: a small number for gradient check, a float scalar.
    Output:
        dL_db: the approximated gradients of the loss function w.r.t. the biases, a float vector of shape (c, 1).
    '''
    c = b.shape[0]
    dL_db = np.zeros(b.shape)
    for i in range(c):
        b_plus = b.copy()
        b_minus = b.copy()
        b_plus[i] += delta
        b_minus[i] -= delta
        z_plus, a_plus, L_plus = forward(x, y, W, b_plus)
        z_minus, a_minus, L_minus = forward(x, y, W, b_minus)
        dL_db[i] = (L_plus - L_minus) / (2 * delta)
    return dL_db





