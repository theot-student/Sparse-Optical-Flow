import torch
import numpy as np
import math
from time import process_time
import torchviz


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
    dw1_x2 = torch.square(dw1_x2)
    dw1_y2[:,:,:,:-1] = motion_field[:, :, :, 1:, 0] - motion_field[:, :, :, :-1, 0]
    dw1_y2 = torch.square(dw1_y2)
    dw2_x2[:,:,:-1,:] = motion_field[:, :, 1:, :, 1] - motion_field[:, :, :-1, :, 1]
    dw2_x2 = torch.square(dw2_x2)
    dw2_y2[:,:,:,:-1] = motion_field[:, :, :, 1:, 1] - motion_field[:, :, :, :-1, 1]
    dw2_y2 = torch.square(dw2_y2)

    sv = torch.sqrt((weighting * weighting * (dw1_x2 + dw1_y2 + dw2_x2 + dw2_y2)) \
        + ((1 - weighting) * (1 - weighting) * (torch.square(motion_field[:, :, :, :, 0]) \
        + torch.square(motion_field[:, :, :, :, 1]))) + 1e-15)
    return sv


def data_term(motion_field, du_x, du_y, du_t):
    """Calculate the data term
    Parameters
    ----------
    motion_field: Tensor
        Tensor containing the estimated motion field
    old_image: Tensor
        Tensor containing the old image
    new_image: Tensor
        Tensor containing the new image
    """

    dterm = torch.square(du_x * motion_field[:, :, :, :, 0] + du_y * motion_field[:, :, :, :, 1] + du_t)
    return dterm



class OF():
    """Gray scaled image motion field

    Parameters
    ----------
    weight: float
        model weight between hessian and sparsity. Value is in  ]0, 1[. 0 = sparse; 1 = not sparse
    reg: float
        Regularization weight. Value is in [0, 1]
    gradient_step: foat
        Gradient descent step
    precision: float
        Stop criterion. Stop gradient descent when the loss decrease less than precision

    """
    def __init__(self, weight, reg, gradient_step,
                 precision, init):
        super().__init__()
        self.weight = weight
        self.reg = reg
        self.precision = precision
        self.niter_ = 0
        self.max_iter_ = 2500
        self.gradient_step_ = gradient_step
        self.loss_ = None
        self.init = init

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
        motion Field (3D torch.Tensor)

        """
        #self.progress(0)


        mini = torch.min(old_image)
        maxi = torch.max(old_image)
        old_image = (old_image-mini)/(maxi-mini)

        mini = torch.min(new_image)
        maxi = torch.max(new_image)
        new_image = (new_image-mini)/(maxi-mini)


        if self.init == 1:
            motion_field = (torch.rand(np.shape(old_image)[0], np.shape(old_image)[1], 2) * 2 - torch.ones(np.shape(old_image)[0], np.shape(old_image)[1], 2))
        elif self.init == 2:
            motion_field = 0.1 * (torch.rand(np.shape(old_image)[0], np.shape(old_image)[1], 2) * 2 - torch.ones(np.shape(old_image)[0], np.shape(old_image)[1], 2))
        elif self.init == 0:
            motion_field = torch.zeros((np.shape(old_image)[0], np.shape(old_image)[1], 2))
        else:
            motion_field = torch.zeros((np.shape(old_image)[0], np.shape(old_image)[1], 2))

        #reshaping the motion field for ADAM algorithm
        motion_field = motion_field.view(1, 1, motion_field.shape[0], motion_field.shape[1], 2)


        #ready to compute gradient
        motion_field.requires_grad = True

        optimizer = torch.optim.Adam([motion_field], lr=self.gradient_step_)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        previous_loss = 9e12
        count_eq = 0
        self.niter_ = 0

        du_y = torch.zeros_like(old_image)
        du_x = torch.zeros_like(old_image)

        du_y[1:-1,:] = (old_image[2:, :] - old_image[:-2, :])/2
        du_x[:,1:-1] = (old_image[:, 2:] - old_image[:, :-2])/2
        du_t = (new_image[:, :] - old_image[:, :])


        for i in range(self.max_iter_):
            #self.progress(int(100*i/self.max_iter_))
            self.niter_ += 1

            optimizer.zero_grad()

            loss = torch.mean((1-self.reg) * SV_loss(motion_field, self.weight) + \
                self.reg * data_term(motion_field, du_x, du_y, du_t))

            """
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
        final_motion_field = motion_field.view(motion_field.shape[2], motion_field.shape[3],2).clone()

        """
        motion_norm = torch.sqrt(torch.square(final_motion_field[...,0]) + torch.square(final_motion_field[...,1]))
        quantile = torch.quantile(motion_norm,0.97)
        mask = torch.where(motion_norm > quantile, True, False)
        final_motion_field[mask == False] = torch.zeros(2)
        """

        return final_motion_field
