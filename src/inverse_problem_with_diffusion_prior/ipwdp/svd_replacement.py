import torch
import numpy as np

class H_functions(torch.nn.Module):
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def __init__(self):
        super(H_functions, self).__init__()
    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec, for_H=True):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()
    
    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])
    
    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))
    
    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        return self.V(self.add_zeros(temp))

#a memory inefficient implementation for any general degradation H
class GeneralH(H_functions):
    def mat_by_vec(self, M, v):
        vshape = v.shape[1]
        if len(v.shape) > 2: vshape = vshape * v.shape[2]
        if len(v.shape) > 3: vshape = vshape * v.shape[3]
        return torch.matmul(M, v.view(v.shape[0], vshape,
                        1)).view(v.shape[0], M.shape[0])

    def __init__(self, H):
        self._U, self._singulars, self._V = torch.svd(H, some=False)
        self._Vt = self._V.transpose(0, 1)
        self._Ut = self._U.transpose(0, 1)

        ZERO = 1e-3
        self._singulars[self._singulars < ZERO] = 0
        print(len([x.item() for x in self._singulars if x == 0]))

    def V(self, vec):
        return self.mat_by_vec(self._V, vec.clone())

    def Vt(self, vec, for_H=True):
        return self.mat_by_vec(self._Vt, vec.clone())

    def U(self, vec):
        return self.mat_by_vec(self._U, vec.clone())

    def Ut(self, vec):
        return self.mat_by_vec(self._Ut, vec.clone())

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self._V.shape[0], device=vec.device)
        out[:, :self._U.shape[0]] = vec.clone().reshape(vec.shape[0], -1)
        return out

#Inpainting
class Inpainting(H_functions):
    def __init__(self, channels, img_dim, missing_indices, device):
        super(Inpainting, self).__init__()
        self.channels = channels
        self.img_dim = img_dim
        self._singulars = torch.nn.Parameter(torch.ones(channels * img_dim**2 - missing_indices.shape[0]).to(device), requires_grad=False)
        self.missing_indices = torch.nn.Parameter(missing_indices, requires_grad=False)
        self.kept_indices = torch.nn.Parameter(torch.Tensor([i for i in range(channels * img_dim**2) if i not in missing_indices]).to(device).long(), requires_grad=False)

    def V(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, self.kept_indices] = temp[:, :self.kept_indices.shape[0]]
        out[:, self.missing_indices] = temp[:, self.kept_indices.shape[0]:]
        return out.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

    def Vt(self, vec, for_H=True):
        temp = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, :self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0]:] = temp[:, self.missing_indices]
        return out

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim**2), device=vec.device)
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp

#Denoising
class Denoising(H_functions):
    def __init__(self, channels, img_dim, device):
        self._singulars = torch.ones(channels * img_dim**2, device=device)

    def V(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Vt(self, vec, for_H=True):
        return vec.clone().reshape(vec.shape[0], -1)

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

#Super Resolution
class SuperResolution(H_functions):
    def __init__(self, channels, img_dim, ratio, device): #ratio = 2 or 4
        super(SuperResolution, self).__init__()
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        H = torch.Tensor([[1 / ratio**2] * ratio**2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.U_small = torch.nn.Parameter(self.U_small, requires_grad=False)
        self.V_small = torch.nn.Parameter(self.V_small, requires_grad=False)
        self.singulars_small = torch.nn.Parameter(self.singulars_small, requires_grad=False)
        self.Vt_small = torch.nn.Parameter(self.V_small.transpose(0, 1), requires_grad=False)

    def V(self, vec):
        #reorder the vector back into patches (because singulars are ordered descendingly)
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(vec.shape[0], self.channels, self.y_dim**2, self.ratio**2, device=vec.device)
        patches[:, :, :, 0] = temp[:, :self.channels * self.y_dim**2].view(vec.shape[0], self.channels, -1)
        for idx in range(self.ratio**2-1):
            patches[:, :, :, idx+1] = temp[:, (self.channels*self.y_dim**2+idx)::self.ratio**2-1].view(vec.shape[0], self.channels, -1)
        #multiply each patch by the small V
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #repatch the patches into an image
        patches_orig = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        return recon

    def Vt(self, vec, for_H=True):
        #extract flattened patches
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        unfold_shape = patches.shape
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #multiply each by the small V transposed
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(vec.shape[0], self.channels * self.img_dim**2, device=vec.device)
        recon[:, :self.channels * self.y_dim**2] = patches[:, :, :, 0].view(vec.shape[0], self.channels * self.y_dim**2)
        for idx in range(self.ratio**2-1):
            recon[:, (self.channels*self.y_dim**2+idx)::self.ratio**2-1] = patches[:, :, :, idx+1].view(vec.shape[0], self.channels * self.y_dim**2)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec): #U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp

#Colorization
class Colorization(H_functions):
    def __init__(self, img_dim, device):
        super(Colorization, self).__init__()
        self.channels = 3
        self.img_dim = img_dim
        #Do the SVD for the per-pixel matrix
        H = torch.nn.Parameter(torch.Tensor([[0.3333, 0.3333, 0.3333]]), requires_grad=False).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)
        self.Vt_small = torch.nn.Parameter(self.Vt_small, requires_grad=False)
        self.V_small = torch.nn.Parameter(self.V_small, requires_grad=False)
        self.singulars_small = torch.nn.Parameter(self.singulars_small, requires_grad=False)
        self.U_small = torch.nn.Parameter(self.U_small, requires_grad=False)

    def V(self, vec):
        #get the needles
        needles = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1) #shape: B, WH, C'
        #multiply each needle by the small V
        needles = torch.matmul(self.V_small, needles.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1, self.channels) #shape: B, WH, C
        #permute back to vector representation
        recon = needles.permute(0, 2, 1) #shape: B, C, WH
        return recon.reshape(vec.shape[0], -1)

    def Vt(self, vec, for_H=True):
        #get the needles
        needles = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1) #shape: B, WH, C
        #multiply each needle by the small V transposed
        needles = torch.matmul(self.Vt_small, needles.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1, self.channels) #shape: B, WH, C'
        #reorder the vector so that the first entry of each needle is at the top
        recon = needles.permute(0, 2, 1).reshape(vec.shape[0], -1)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec): #U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.img_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim**2), device=vec.device)
        temp[:, :self.img_dim**2] = reshaped
        return temp

#Walsh-Hadamard Compressive Sensing
class WalshHadamardCS(H_functions):
    def fwht(self, vec): #the Fast Walsh Hadamard Transform is the same as its inverse
        a = vec.reshape(vec.shape[0], self.channels, self.img_dim**2)
        h = 1
        while h < self.img_dim**2:
            a = a.reshape(vec.shape[0], self.channels, -1, h * 2)
            b = a.clone()
            a[:, :, :, :h] = b[:, :, :, :h] + b[:, :, :, h:2*h]
            a[:, :, :, h:2*h] = b[:, :, :, :h] - b[:, :, :, h:2*h]
            h *= 2
        a = a.reshape(vec.shape[0], self.channels, self.img_dim**2) / self.img_dim
        return a

    def __init__(self, channels, img_dim, ratio, perm, device):
        self.channels = channels
        self.img_dim = img_dim
        self.ratio = ratio
        self.perm = perm
        self._singulars = torch.ones(channels * img_dim**2 // ratio, device=device)

    def V(self, vec):
        temp = torch.zeros(vec.shape[0], self.channels, self.img_dim**2, device=vec.device)
        temp[:, :, self.perm] = vec.clone().reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1)
        return self.fwht(temp).reshape(vec.shape[0], -1)

    def Vt(self, vec, for_H=True):
        return self.fwht(vec.clone())[:, :, self.perm].permute(0, 2, 1).reshape(vec.shape[0], -1)

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self.channels * self.img_dim**2, device=vec.device)
        out[:, :self.channels * self.img_dim**2 // self.ratio] = vec.clone().reshape(vec.shape[0], -1)
        return out

#Convolution-based super-resolution
class SRConv(H_functions):
    def mat_by_img(self, M, v, dim):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, dim,
                        dim)).reshape(v.shape[0], self.channels, M.shape[0], dim)

    def img_by_mat(self, v, M, dim):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, dim,
                        dim), M).reshape(v.shape[0], self.channels, dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, stride = 1):
        self.img_dim = img_dim
        self.channels = channels
        self.ratio = stride
        small_dim = img_dim // stride
        self.small_dim = small_dim
        #build 1D conv matrix
        H_small = torch.zeros(small_dim, img_dim, device=device)
        for i in range(stride//2, img_dim + stride//2, stride):
            for j in range(i - kernel.shape[0]//2, i + kernel.shape[0]//2):
                j_effective = j
                #reflective padding
                if j_effective < 0: j_effective = -j_effective-1
                if j_effective >= img_dim: j_effective = (img_dim - 1) - (j_effective - img_dim)
                #matrix building
                H_small[i // stride, j_effective] += kernel[j - i + kernel.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small.reshape(small_dim, 1), self.singulars_small.reshape(1, small_dim)).reshape(small_dim**2)
        #permutation for matching the singular values. See P_1 in Appendix D.5.
        self._perm = torch.Tensor([self.img_dim * i + j for i in range(self.small_dim) for j in range(self.small_dim)] + \
                                  [self.img_dim * i + j for i in range(self.small_dim) for j in range(self.small_dim, self.img_dim)]).to(device).long()

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)[:, :self._perm.shape[0], :]
        temp[:, self._perm.shape[0]:, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)[:, self._perm.shape[0]:, :]
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp, self.img_dim)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1), self.img_dim).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec, for_H=True):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone(), self.img_dim)
        temp = self.img_by_mat(temp, self.V_small, self.img_dim).reshape(vec.shape[0], self.channels, -1)
        #permute the entries
        temp[:, :, :self._perm.shape[0]] = temp[:, :, self._perm]
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.small_dim**2, self.channels, device=vec.device)
        temp[:, :self.small_dim**2, :] = vec.clone().reshape(vec.shape[0], self.small_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp, self.small_dim)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1), self.small_dim).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone(), self.small_dim)
        temp = self.img_by_mat(temp, self.U_small, self.small_dim).reshape(vec.shape[0], self.channels, -1)
        #permute the entries
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat_interleave(3).reshape(-1)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp

#Deblurring
class Deblurring(H_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, ZERO = 3e-2):
        self.img_dim = img_dim
        self.channels = channels
        #build 1D conv matrix
        H_small = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel.shape[0]//2, i + kernel.shape[0]//2):
                if j < 0 or j >= img_dim: continue
                H_small[i, j] = kernel[j - i + kernel.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        self.U_small = torch.nn.Parameter(self.U_small)
        self.singulars_small = torch.nn.Parameter(self.singulars_small)
        self.V_small = torch.nn.Parameter(self.V_small)
        #ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars = torch.nn.Parameter(torch.matmul(self.singulars_small.reshape(img_dim, 1), self.singulars_small.reshape(1, img_dim)).reshape(img_dim**2))
        #sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True) #, stable=True)
        self._singulars = torch.nn.Parameter(self._singulars)
        self._perm = torch.nn.Parameter(self._perm)

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec, for_H=True):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

#Anisotropic Deblurring
class Deblurring2D(H_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel1, kernel2, channels, img_dim, device):
        super(Deblurring2D, self).__init__()
        self.img_dim = img_dim
        self.channels = channels
        #build 1D conv matrix - kernel1
        H_small1 = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel1.shape[0]//2, i + kernel1.shape[0]//2):
                if j < 0 or j >= img_dim: continue
                H_small1[i, j] = kernel1[j - i + kernel1.shape[0]//2]
        #build 1D conv matrix - kernel2
        H_small2 = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel2.shape[0]//2, i + kernel2.shape[0]//2):
                if j < 0 or j >= img_dim: continue
                H_small2[i, j] = kernel2[j - i + kernel2.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small1, self.singulars_small1, self.V_small1 = torch.svd(H_small1, some=False)
        self.U_small2, self.singulars_small2, self.V_small2 = torch.svd(H_small2, some=False)
        ZERO = 3e-2
        self.singulars_small1[self.singulars_small1 < ZERO] = 0
        self.singulars_small2[self.singulars_small2 < ZERO] = 0

        self.U_small1, self.U_small2 = torch.nn.Parameter(self.U_small1, requires_grad=False), torch.nn.Parameter(self.U_small2, requires_grad=False)
        self.singulars_small1 = torch.nn.Parameter(self.singulars_small1, requires_grad=False)
        self.singulars_small2 = torch.nn.Parameter(self.singulars_small2, requires_grad=False)
        self.V_small1 = torch.nn.Parameter(self.V_small1, requires_grad=False)
        self.V_small2 = torch.nn.Parameter(self.V_small2, requires_grad=False)

        #calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small1.reshape(img_dim, 1), self.singulars_small2.reshape(1, img_dim)).reshape(img_dim**2)
        #sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True) #, stable=True)
        self._singulars = torch.nn.Parameter(self._singulars, requires_grad=False)
        self._perm = torch.nn.Parameter(self._perm, requires_grad=False)

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small1, temp)
        out = self.img_by_mat(out, self.V_small2.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec, for_H=True):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small1.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small2).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small1, temp)
        out = self.img_by_mat(out, self.U_small2.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small1.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small2).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, self.channels).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)


class DeblurringArbitral2D(H_functions):

    def __init__(self, kernel, channels, img_dim, device, conv_shape='same'):
        super(DeblurringArbitral2D, self).__init__()
        self.img_dim = img_dim
        self.channels = channels
        self.conv_shape = conv_shape
        _nextpow2 = lambda x: int(np.power(2, np.ceil(np.log2(x))))
        self.fft2_size = _nextpow2(img_dim + kernel.shape[0] - 1) # next pow 2
        self.kernel_size = (kernel.shape[-2], kernel.shape[-1])
        self.kernel = torch.nn.Parameter(kernel,  requires_grad=False)

        self.k_fft = torch.nn.Parameter(torch.fft.fft2(self.kernel, (self.fft2_size, self.fft2_size), norm="ortho"),
                                        requires_grad=False)

        #
        _eps_singulars = 0 * torch.randn_like(self.k_fft)
        self._singular_phases = ((self.k_fft + _eps_singulars) / torch.abs(self.k_fft + _eps_singulars)).reshape(-1)
        self._singulars = torch.abs(self.k_fft * self.fft2_size).reshape(-1)
        ZERO = 3e-5  # 0.05
        self._singulars[self._singulars < ZERO] = 0.0
        self._singulars, self._perm = self._singulars.sort(descending=True)
        self._singular_phases = self._singular_phases[self._perm]
        self._singulars = torch.nn.Parameter(self._singulars, requires_grad=False)
        self._singular_phases = torch.nn.Parameter(self._singular_phases, requires_grad=False)
        self._perm = torch.nn.Parameter(self._perm, requires_grad=False)
        self.device = device

        if conv_shape == 'same':
            self.out_img_dim = img_dim
        elif conv_shape == 'full':
            # TODO: rectangular kernel size
            self.out_img_dim = img_dim + (self.kernel_size[0] - 1)
        elif conv_shape == "same_interp":
            self.out_img_dim = img_dim

    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        ret = self.U(singulars * temp[:, :singulars.shape[0]])

        return ret

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[1]]))

    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        temp[:, :singulars.shape[1]] = temp[:, :singulars.shape[1]] / singulars
        return self.V(self.add_zeros(temp))

    def V(self, vec):

        vec = self.add_zeros(vec).reshape(vec.shape[0], self.channels, -1)
        vec = vec / self._singular_phases[None, None, :]
        vec = vec.reshape(vec.shape[0], self.channels, -1)

        vec = self._batch_inv_perm(vec, self._perm)

        vec_ifft = torch.fft.ifft2(vec.reshape(vec.shape[0], self.channels, self.fft2_size, self.fft2_size), \
                                   norm="ortho").real

        out = vec_ifft[:, :, :self.img_dim, :self.img_dim].reshape(vec.shape[0], -1)

        return out

    def Vt(self, vec, for_H=True):

        vec_fft = torch.fft.fft2(vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim),
                                 (self.fft2_size, self.fft2_size), norm="ortho")

        vec_fft = self._batch_perm(vec_fft.reshape(vec.shape[0], self.channels, -1), self._perm)
        vec_fft = vec_fft * self._singular_phases[None, None, :].to(vec_fft.device)
        
        if for_H:
            return vec_fft.reshape(vec.shape[0], -1)
        vec_fft_img = vec_fft.reshape(vec_fft.shape[0],
                               vec_fft.shape[1],
                               self.fft2_size,
                               self.fft2_size)
        return vec_fft_img[:, :, :self.img_dim, :self.img_dim].reshape(vec.shape[0], -1).real

    def U(self, vec):

        vec = vec.reshape(vec.shape[0], self.channels, -1)
        vec = self._batch_inv_perm(vec, self._perm)

        vec_ifft = torch.fft.ifft2(vec.reshape(vec.shape[0], self.channels, self.fft2_size, self.fft2_size), \
                                   norm="ortho").real

        if self.conv_shape == 'same':
            out = vec_ifft[:, :, self.kernel_size[0] // 2:int(self.kernel_size[0] // 2 + self.img_dim), \
                  self.kernel_size[1] // 2:int(self.kernel_size[1] // 2 + self.img_dim)]
        elif self.conv_shape == 'full':
            out = vec_ifft[:, :, :self.out_img_dim, :self.out_img_dim]
        else:  # elif self.conv_shape == "same_interp":
            out = vec_ifft[:, :, self.kernel_size[0] // 2:int(self.kernel_size[0] // 2 + self.img_dim), \
                  self.kernel_size[1] // 2:int(self.kernel_size[1] // 2 + self.img_dim)]

        return out


    def Ut(self, vec):

        _ks0 = self.kernel_size[0]
        _ks1 = self.kernel_size[1]
        _Nf = self.fft2_size

        if self.conv_shape == 'same':
            exec_zeropad = torch.nn.ZeroPad2d((_ks0 // 2, _Nf - _ks0 // 2 - self.img_dim, \
                                               _ks1 // 2, _Nf - _ks1 // 2 - self.img_dim))

            vec = exec_zeropad(vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim))
        elif self.conv_shape == 'full':
            vec = vec.reshape(vec.shape[0], self.channels, self.out_img_dim, self.out_img_dim)
            exec_zeropad = torch.nn.ZeroPad2d((0, _Nf - self.out_img_dim, 0, _Nf - self.out_img_dim))
            vec = exec_zeropad(vec)

        elif self.conv_shape == "same_interp":
            pass

        vec_fft = torch.fft.fft2(vec, (self.fft2_size, self.fft2_size), norm="ortho")

        vec_fft = self._batch_perm(vec_fft.reshape(vec.shape[0], self.channels, -1), self._perm)

        return vec_fft.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        tmp = torch.zeros(vec.shape[0], self.channels, self.fft2_size ** 2, device=vec.device, dtype=vec.dtype)
        reshaped = vec.clone().reshape(vec.shape[0], self.channels, -1)
        tmp[:, :, :reshaped.shape[2]] = reshaped

        return tmp.reshape(vec.shape[0], -1)

    def update_kernel(self, kernel):
        """
        Update the internal kernel and associated variables using the provided kernel tensor.

        Args:
            kernel (torch.Tensor): The kernel tensor for the update. It should have the same shape of self.kernel

        Returns:
            None
        """

    def _batch_perm(self, tensor, perm):

        bsz = tensor.shape[0]
        for i_bsz in range(bsz):
            if tensor.dim() == 2:
                tensor[i_bsz, :] = tensor[i_bsz, perm]
            elif tensor.dim() == 3:
                tensor[i_bsz, :, :] = tensor[i_bsz, :, perm]

        return tensor

    def _batch_inv_perm(self, tensor, perm):

        bsz = tensor.shape[0]
        for i_bsz in range(bsz):
            if tensor.dim() == 2:
                tensor[i_bsz, perm] = tensor[i_bsz, :].clone()
            elif tensor.dim() == 3:
                tensor[i_bsz, :, perm] = tensor[i_bsz, :, :].clone()

        return tensor

    @staticmethod
    def get_blur_kernel_batch(batch_size, kernel_type, device, kernel_size):
        """
        Generates a batch of blur kernels of the specified type.

        Args:
            batch_size (int): The number of blur kernels to generate.
            kernel_type (str): The type of blur kernel to generate. Can be one of "gauss", "motionblur", "from_png", or "uniform".
            device (torch.device): The device on which to generate the blur kernels.

        Returns:
            torch.Tensor: A batch of blur kernels of shape (batch_size, kernel_size, kernel_size).
        """

        if kernel_type == "gauss":
            sigma = 5
            pdf = lambda x : torch.exp(torch.Tensor([-0.5 * (x / sigma)]))
            kernel_size = 9 # must be odd
            kernel = torch.zeros((kernel_size, kernel_size)).to(device)
            for i in range(-(kernel_size//2), kernel_size//2+1):
                for j in range(-(kernel_size//2), kernel_size//2+1):
                    kernel[i+kernel_size//2, j+kernel_size//2] = pdf(torch.sqrt(torch.Tensor([i**2+j**2])))
                # zeropad_fun = torch.nn.ZeroPad2d((10, 10, 10, 10))
                # kernel = zeropad_fun(kernel)
            kernel = kernel / kernel.sum()
            kernel_batch = kernel.repeat(batch_size, 1, 1)

        elif kernel_type == "motionblur":
            #kernel_size = 5
            #kernel_size = 64
            kernel_batch = torch.zeros(batch_size, kernel_size, kernel_size, device=device)
            for i_batch in range(batch_size):
                kernel = (Kernel(size=(kernel_size, kernel_size), intensity=0.5).kernelMatrix)
                kernel = torch.from_numpy(kernel).clone().to(device)
                kernel = kernel / kernel.sum()
                kernel_batch[i_batch] = kernel
        else: # config.deblur.kernel_type == "uniform":
            kernel_size = 31
            kernel = torch.ones((kernel_size, kernel_size)).to(device)
            kernel = kernel / kernel.sum()
            kernel_batch = kernel.repeat(batch_size, 1, 1)
            if kernel_type != "uniform":
                print("please specify the kernel type from [gauss, mnist, uniform, motionblur]. uniform kernel is used.")

        return kernel_batch