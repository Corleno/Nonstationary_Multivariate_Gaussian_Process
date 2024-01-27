"""
Multi-task GP (Bonilla)
"""

import torch
import gpytorch
import matplotlib.pyplot as plt
import pickle

import settings


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=3
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=3, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        print(mean_x.size(), covar_x.size())
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


if __name__ == "__main__":
    save_dir = "../res/"
    folder_name = "sim_c/"
    # Load data
    with open("../data/sim.pickle", "rb") as res:
        x, Y, _, _ = pickle.load(res)
    N, M = Y.shape
    # convert numpy to torch
    x = torch.from_numpy(x).type(settings.torchType)
    Y = torch.from_numpy(Y).type(settings.torchType)#.t().contiguous()
    print(x.size(), Y.size())

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
    model = MultitaskGPModel(x, Y, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    n_iter = 100
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(x)
        loss = -mll(output, Y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
        optimizer.step()

    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Initialize plots
    f, (y1_ax, y2_ax, y3_ax) = plt.subplots(1, 3, figsize=(8, 3))

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1, 51)
        predictions = likelihood(model(test_x))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

    # This contains predictions for both tasks, flattened out
    # The first half of the predictions is for the first task
    # The second half is for the second task

    # Plot training data as black stars
    y1_ax.plot(x.detach().numpy(), Y[:, 0].detach().numpy(), 'k*')
    # Predictive mean as blue line
    y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), 'b')
    # Shade in confidence
    y1_ax.fill_between(test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
    # y1_ax.set_ylim([-3, 3])
    y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y1_ax.set_title('Observed Values (Likelihood)')

    # Plot training data as black stars
    y2_ax.plot(x.detach().numpy(), Y[:, 1].detach().numpy(), 'k*')
    # Predictive mean as blue line
    y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), 'b')
    # Shade in confidence
    y2_ax.fill_between(test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
    # y2_ax.set_ylim([-3, 3])
    y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y2_ax.set_title('Observed Values (Likelihood)')

    # Plot training data as black stars
    y3_ax.plot(x.detach().numpy(), Y[:, 2].detach().numpy(), 'k*')
    # Predictive mean as blue line
    y3_ax.plot(test_x.numpy(), mean[:, 2].numpy(), 'b')
    # Shade in confidence
    y3_ax.fill_between(test_x.numpy(), lower[:, 2].numpy(), upper[:, 2].numpy(), alpha=0.5)
    # y3_ax.set_ylim([-3, 3])
    y3_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y3_ax.set_title('Observed Values (Likelihood)')

    plt.show()
