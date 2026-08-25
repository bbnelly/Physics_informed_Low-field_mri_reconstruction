import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDualDomainCNN3D(nn.Module):
    def __init__(self, num_channels=32):
        super().__init__()

        def make_encoder():
            return nn.Sequential(
                nn.Conv3d(2, num_channels, 3, padding=1), nn.BatchNorm3d(num_channels), nn.ReLU(inplace=True),
                nn.Conv3d(num_channels, num_channels, 3, padding=1), nn.BatchNorm3d(num_channels), nn.ReLU(inplace=True),
                nn.Conv3d(num_channels, num_channels, 3, padding=1), nn.BatchNorm3d(num_channels), nn.ReLU(inplace=True),
            )

        self.enc_k = make_encoder()
        self.enc_i = make_encoder()
        self.output = nn.Conv3d(num_channels, 2, 3, padding=1)

    def forward(self, kspace_input, image_input):
        f = self.enc_k(kspace_input)
        i = self.enc_i(image_input)
        return self.output((f + i) / 2)


class DUNDD3D(nn.Module):
    """Dual-domain unrolled network for full 3D Cartesian k-space volumes."""

    def __init__(self, num_iterations=5, lambda_dc=0.5, num_channels=32):
        super().__init__()
        self.num_iterations = num_iterations
        self.lambda_dc = lambda_dc
        self.dual_cnn = SimpleDualDomainCNN3D(num_channels=num_channels)

    def forward(self, masked_kspace, mask):
        kspace = masked_kspace.clone()
        measured = torch.view_as_complex(masked_kspace.permute(0, 2, 3, 4, 1).contiguous())

        for _ in range(self.num_iterations):
            kc = torch.view_as_complex(kspace.permute(0, 2, 3, 4, 1).contiguous())
            img = torch.fft.fftshift(
                torch.fft.ifftn(torch.fft.ifftshift(kc, dim=(-3, -2, -1)),
                                dim=(-3, -2, -1), norm='ortho'),
                dim=(-3, -2, -1),
            )
            img_2ch = torch.view_as_real(img).permute(0, 4, 1, 2, 3).contiguous()

            update_2ch = self.dual_cnn(kspace, img_2ch)
            uc = torch.view_as_complex(update_2ch.permute(0, 2, 3, 4, 1).contiguous())
            update_kspace = torch.fft.fftshift(
                torch.fft.fftn(torch.fft.ifftshift(uc, dim=(-3, -2, -1)),
                               dim=(-3, -2, -1), norm='ortho'),
                dim=(-3, -2, -1),
            )

            kc_new = update_kspace + self.lambda_dc * (measured - update_kspace) * mask
            kc_new = measured * mask + kc_new * (1 - mask)
            kspace = torch.view_as_real(kc_new).permute(0, 4, 1, 2, 3).contiguous()

        return kspace


class CascadeNet3D(nn.Module):
    """Cascaded 3D CNN with hard k-space data consistency."""

    def __init__(self, num_cascades=5, features=24):
        super().__init__()
        self.cascades = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(2, features, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv3d(features, features, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv3d(features, features, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv3d(features, 2, 3, padding=1),
            ) for _ in range(num_cascades)
        ])

    def _hard_dc(self, pred, measured, mask):
        mask_exp = mask.unsqueeze(1).expand_as(pred)
        return pred * (1 - mask_exp) + measured * mask_exp

    def forward(self, masked_kspace, mask):
        x = masked_kspace.clone()
        for cnn in self.cascades:
            x = x + cnn(x)
            x = self._hard_dc(x, masked_kspace, mask)
        return x


class UNet3DBaseline(nn.Module):
    """Small 3D U-Net baseline operating on image-domain volumes."""

    def __init__(self, features=16):
        super().__init__()

        def block(ic, oc):
            return nn.Sequential(
                nn.Conv3d(ic, oc, 3, padding=1), nn.BatchNorm3d(oc), nn.ReLU(inplace=True),
                nn.Conv3d(oc, oc, 3, padding=1), nn.BatchNorm3d(oc), nn.ReLU(inplace=True),
            )

        self.enc1 = block(2, features)
        self.enc2 = block(features, features * 2)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = block(features * 2, features * 4)
        self.up2 = nn.ConvTranspose3d(features * 4, features * 2, 2, 2)
        self.dec2 = block(features * 4, features * 2)
        self.up1 = nn.ConvTranspose3d(features * 2, features, 2, 2)
        self.dec1 = block(features * 2, features)
        self.final = nn.Conv3d(features, 2, 1)

    def forward(self, masked_kspace, mask):
        kc = torch.view_as_complex(masked_kspace.permute(0, 2, 3, 4, 1).contiguous())
        img = torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(kc, dim=(-3, -2, -1)),
                            dim=(-3, -2, -1), norm='ortho'),
            dim=(-3, -2, -1),
        )
        x = torch.view_as_real(img).permute(0, 4, 1, 2, 3).contiguous()
        d, h, w = x.shape[-3:]
        pad_d, pad_h, pad_w = (4 - d % 4) % 4, (4 - h % 4) % 4, (4 - w % 4) % 4
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out_img = self.final(d1)[:, :, :d, :h, :w]

        out_c = torch.view_as_complex(out_img.permute(0, 2, 3, 4, 1).contiguous())
        out_k = torch.fft.fftshift(
            torch.fft.fftn(torch.fft.ifftshift(out_c, dim=(-3, -2, -1)),
                           dim=(-3, -2, -1), norm='ortho'),
            dim=(-3, -2, -1),
        )
        return torch.view_as_real(out_k).permute(0, 4, 1, 2, 3).contiguous()


class MoDL3D(nn.Module):
    """3D MoDL-style model with batch-aware CG in image-volume space."""

    def __init__(self, num_iterations=6, num_cg_steps=5, lambda_reg=0.05, features=32):
        super().__init__()
        self.K = num_iterations
        self.cg_steps = num_cg_steps
        inv_val = math.log(math.exp(lambda_reg) - 1 + 1e-8)
        self._lam_raw = nn.Parameter(torch.tensor(inv_val, dtype=torch.float32))
        self.denoiser = nn.Sequential(
            nn.Conv3d(2, features, 3, padding=1), nn.BatchNorm3d(features), nn.ReLU(inplace=True),
            nn.Conv3d(features, features, 3, padding=1), nn.BatchNorm3d(features), nn.ReLU(inplace=True),
            nn.Conv3d(features, features, 3, padding=1), nn.BatchNorm3d(features), nn.ReLU(inplace=True),
            nn.Conv3d(features, 2, 3, padding=1),
        )

    @property
    def lam(self):
        return F.softplus(self._lam_raw) + 1e-4

    def _r2c(self, x):
        return torch.view_as_complex(x.permute(0, 2, 3, 4, 1).contiguous())

    def _c2r(self, x):
        return torch.view_as_real(x).permute(0, 4, 1, 2, 3).contiguous()

    def _A(self, img_c, mask):
        k = torch.fft.fftshift(
            torch.fft.fftn(torch.fft.ifftshift(img_c, dim=(-3, -2, -1)),
                           dim=(-3, -2, -1), norm='ortho'),
            dim=(-3, -2, -1),
        )
        return k * mask

    def _AH(self, k_c, mask):
        return torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(k_c * mask, dim=(-3, -2, -1)),
                            dim=(-3, -2, -1), norm='ortho'),
            dim=(-3, -2, -1),
        )

    def _cg(self, rhs_c, mask, x0_c):
        def dot(a, b):
            return (a.conj() * b).real.sum(dim=(-3, -2, -1))

        def Ax_fn(v):
            return self._AH(self._A(v, mask), mask) + self.lam * v

        x = x0_c.clone()
        r = rhs_c - Ax_fn(x)
        p = r.clone()
        rs = dot(r, r)
        for _ in range(self.cg_steps):
            Ap = Ax_fn(p)
            alpha = (rs / (dot(p, Ap) + 1e-8)).view(-1, 1, 1, 1)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = dot(r, r)
            if rs_new.max().item() < 1e-10:
                break
            beta = (rs_new / (rs + 1e-8)).view(-1, 1, 1, 1)
            p = r + beta * p
            rs = rs_new
        return x

    def forward(self, masked_kspace, mask):
        y_c = self._r2c(masked_kspace)
        AHy = self._AH(y_c, mask)
        x = AHy.clone()
        for _ in range(self.K):
            Dw_x = self.denoiser(self._c2r(x))
            rhs = AHy + self.lam * self._r2c(Dw_x)
            x = self._cg(rhs, mask, x)
        out_k = torch.fft.fftshift(
            torch.fft.fftn(torch.fft.ifftshift(x, dim=(-3, -2, -1)),
                           dim=(-3, -2, -1), norm='ortho'),
            dim=(-3, -2, -1),
        )
        return self._c2r(out_k)


class VarNetBlock3D(nn.Module):
    def __init__(self, features=16):
        super().__init__()
        self.refiner = nn.Sequential(
            nn.Conv3d(2, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(features, 2, 3, padding=1),
        )
        self.eta = nn.Parameter(torch.tensor(1.0))

    def forward(self, kspace, masked_kspace, mask):
        mask_exp = mask.unsqueeze(1).expand_as(kspace)
        dc_grad = (kspace - masked_kspace) * mask_exp
        ref = self.refiner(kspace)
        return kspace - self.eta * dc_grad - ref


class E2EVarNet3D(nn.Module):
    """3D single-coil variational network over full k-space volumes."""

    def __init__(self, num_cascades=6, features=16):
        super().__init__()
        self.cascades = nn.ModuleList([VarNetBlock3D(features) for _ in range(num_cascades)])

    def forward(self, masked_kspace, mask):
        kspace = masked_kspace.clone()
        for cascade in self.cascades:
            kspace = cascade(kspace, masked_kspace, mask)
        return kspace
