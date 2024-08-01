import torch
import numpy as np
import math
from time import process_time
import torchviz

epsilon = 1e-6


def l1Norm(tensor):
    """compute the approximation of l1-norm

    Parameters
    ----------
    tensor: Tensor
        Tensor on which we compute l1-norm

    """
    return torch.sqrt(torch.square(tensor) + epsilon)

def SV_loss(motion_field, weighting):
    """Sparse vatiation regularization term

    Parameters
    ----------
    motion_field: Tensor
        Tensor containing the estimated motion field
    weighting: float
        Sparse weighting parameter in [0, 1]. 0 sparse, and 1 not sparse

    """
    dw1_x2 = torch.zeros_like(motion_field[:,:,:,:,0])
    dw1_y2 = torch.zeros_like(motion_field[:,:,:,:,0])
    dw2_x2 = torch.zeros_like(motion_field[:,:,:,:,0])
    dw2_y2 = torch.zeros_like(motion_field[:,:,:,:,0])
    #calcul of (dw_1/dx)²
    dw1_x2[:,:,:-1,:] = motion_field[:, :, 1:, :, 0] - motion_field[:, :, :-1, :, 0]
    dw1_x2 = l1Norm(dw1_x2)
    dw1_y2[:,:,:,:-1] = motion_field[:, :, :, 1:, 0] - motion_field[:, :, :, :-1, 0]
    dw1_y2 = l1Norm(dw1_y2)
    dw2_x2[:,:,:-1,:] = motion_field[:, :, 1:, :, 1] - motion_field[:, :, :-1, :, 1]
    dw2_x2 = l1Norm(dw2_x2)
    dw2_y2[:,:,:,:-1] = motion_field[:, :, :, 1:, 1] - motion_field[:, :, :, :-1, 1]
    dw2_y2 = l1Norm(dw2_y2)

    sv = torch.sqrt((weighting * weighting * (dw1_x2 + dw1_y2 + dw2_x2 + dw2_y2)) \
        + ((1 - weighting) * (1 - weighting) * (l1Norm(motion_field[:, :, :, :, 0]) \
        + l1Norm(motion_field[:, :, :, :, 1]))) + epsilon)
    return sv


def data_term(motion_field, du_x, du_y, du_t):
    """Calculate the data term
    Parameters
    ----------
    motion_field: Tensor
        Tensor containing the estimated motion field
    du_x: Tensor
        Tensor containing the x partial derivative of the old image
    du_y: Tensor
        Tensor containing the y partial derivative of the old image
    du_t: Tensor
        Tensor containing the temporak partial derivative between the 2 images
    """

    dterm = torch.sqrt(torch.square(du_x * motion_field[:, :, :, :, 0] + du_y * motion_field[:, :, :, :, 1] + du_t) + epsilon)
    return dterm



class OF():
    """grayscale image optical flow

    Parameters
    ----------
    weight: float
        model weight between TV-L1 and sparsity. Value is in  [0, 1]. 0 = sparse; 1 = not sparse
    reg: float
        Regularization weight. Value is in [0, 1]. 0 = very regularized
    gradient_step: foat
        Gradient descent step
    precision: float
        Stop criterion. Stop gradient descent when the loss decrease less than precision

    """
    def __init__(self, weight, reg, gradient_step,
                 precision):
        super().__init__()
        self.weight = weight
        self.reg = reg
        self.precision = precision
        self.niter_ = 0
        self.max_iter_ = 2500
        self.gradient_step_ = gradient_step
        self.loss_ = None

    def __call__(self, old_image, new_image):
        return self.run(old_image, new_image)

    def run(self, old_image, new_image):
        """Implements optical flow with Sparse regularization (Spitfire method) for 2 consecutives 2D images

        Parameters
        ----------
        old_image: torch.Tensor
            first 2D image tensor
        new_image: torch.Tensor
            second 2D image tensor
        Returns
        -------
        motion Field: 2D torch.Tensor
            motion field representing optical flow between the 2 images

        """


        #normalizaton of images for quicker convergence of algorithm
        mini = torch.min(old_image)
        maxi = torch.max(old_image)
        old_image = (old_image-mini)/(maxi-mini)

        mini = torch.min(new_image)
        maxi = torch.max(new_image)
        new_image = (new_image-mini)/(maxi-mini)


        #creation of the motion field (shape = (x,y,dir))

        #motion_field = 0.1 * (torch.rand(np.shape(old_image)[0], np.shape(old_image)[1], 2) * 2 - torch.ones(np.shape(old_image)[0], np.shape(old_image)[1], 2))
        motion_field = torch.zeros((np.shape(old_image)[0], np.shape(old_image)[1], 2))

        #reshaping the motion field for ADAM algorithm (batches, channel, x, y, dir)
        motion_field = motion_field.view(1, 1, motion_field.shape[0], motion_field.shape[1], 2)


        #ready to compute gradient
        motion_field.requires_grad = True

        #initialization of optimizer
        optimizer = torch.optim.Adam([motion_field], lr=self.gradient_step_)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        previous_loss = 9e12
        count_eq = 0
        self.niter_ = 0

        #compute the partial derivatives
        du_y = torch.zeros_like(old_image)
        du_x = torch.zeros_like(old_image)

        du_y[1:-1,:] = (old_image[2:, :] - old_image[:-2, :])/2
        du_x[:,1:-1] = (old_image[:, 2:] - old_image[:, :-2])/2
        du_t = (new_image[:, :] - old_image[:, :])

        #optimization
        for i in range(self.max_iter_):
            self.niter_ += 1

            optimizer.zero_grad()

            loss = torch.mean((1-self.reg) * SV_loss(motion_field, self.weight) + \
                self.reg * data_term(motion_field, du_x, du_y, du_t))

            """
            #compute the autograd graph of pytorch
            if i == 0:
                dot = torchviz.make_dot(loss, params={"motion_field" : motion_field}, show_attrs=True, show_saved=True)
                dot.render(directory='/home/local_theotim/test/', view=True)
                break
            """

            print('iter:', self.niter_, ' loss:', loss.item())
            if abs(loss - previous_loss) < self.precision:
                count_eq += 1
            else:
                previous_loss = loss
                count_eq = 0
            if count_eq > 5:
                break

            loss.backward()
            optimizer.step()
            scheduler.step()


        self.loss_ = loss
        final_motion_field = torch.squeeze(motion_field.clone())

        """
        motion_norm = torch.sqrt(torch.square(final_motion_field[...,0]) + torch.square(final_motion_field[...,1]))
        quantile = torch.quantile(motion_norm,0.97)
        mask = torch.where(motion_norm > quantile, True, False)
        final_motion_field[mask == False] = torch.zeros(2)
        """

        return final_motion_field
