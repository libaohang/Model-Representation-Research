import numpy as np

class Adam:
    def __init__(self, network, lr=1e-3, warmupSteps = 100, betas=(0.9, 0.999), eps=1e-8):
        self.network = network
        self.targetlr = lr
        self.warmupSteps = warmupSteps
        self.beta1, self.beta2 = betas
        self.eps = eps

        self.t = 0
        self.params = self.network.parameters()
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
                
        if self.t <= self.warmupSteps:
            lr = self.targetlr * self.t / self.warmupSteps
        else:
            lr = self.targetlr
            
        parameters = self.params
        gradients = self.network.gradients()

        for i, (p, g) in enumerate(zip(parameters, gradients)):
            if g is None:
                continue

            # Update biased moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Parameter update (in-place)
            p -= lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p = p.astype(np.float32)

class AdamW:
    def __init__(self, network, lr=1e-3, warmupSteps=100, weightDecay=1e-2,
                 betas=(0.9, 0.999), eps=1e-8,
                 decay_tags=("weight",)):
        self.network = network
        self.targetlr = lr
        self.warmupSteps = warmupSteps
        self.weight_decay = weightDecay
        self.decay_tags = set(decay_tags)

        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        # params: list of (param, tag)
        self.params = self.network.parameters()

        # optimizer state per param array
        self.m = [np.zeros_like(p) for (p, _) in self.params]
        self.v = [np.zeros_like(p) for (p, _) in self.params]

    def step(self):
        self.t += 1

        # warmup
        if self.t <= self.warmupSteps:
            lr = self.targetlr * self.t / self.warmupSteps
        else:
            lr = self.targetlr

        grads = self.network.gradients()  # list of (grad_array, tag)

        # Checked parameter and gradient matching
        #assert len(self.params) == len(grads)
        #for (p, pt), (g, gt) in zip(self.params, grads):
        #    assert pt == gt
        #    if g is not None:
        #        print(p.shape, g.shape, pt)
        #        assert p.shape == g.shape

        for i, ((p, ptag), (g, gtag)) in enumerate(zip(self.params, grads)):

            if g is None:
                continue

            # AdamW decoupled weight decay
            if self.weight_decay != 0 and ptag in self.decay_tags:
                p -= lr * self.weight_decay * p

            # Adam moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # update
            p -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def lrDecay(self):
        self.targetlr /= 10