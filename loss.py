#Loss functions
def gradient_penalty(samples, output, weight):
    gradients = K.gradients(output, samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))

    # (weight / 2) * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty) * weight

def hinge_d(y_true, y_pred):
    return K.mean(K.relu(1.0 + (y_true * y_pred)))

def w_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)