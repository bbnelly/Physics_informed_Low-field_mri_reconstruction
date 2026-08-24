# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDualDomainCNN(nn.Module):
    def __init__(self, num_channels=64):
        super().__init__()
        def make_encoder():
            return nn.Sequential(
                nn.Conv2d(2, num_channels, 3, padding=1), nn.BatchNorm2d(num_channels), nn.ReLU(),
                nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.BatchNorm2d(num_channels), nn.ReLU(),
                nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.BatchNorm2d(num_channels), nn.ReLU(),
            )
        # Separate encoders: k-space and image domains have very different statistics.
        self.enc_k = make_encoder()
        self.enc_i = make_encoder()
        self.output = nn.Conv2d(num_channels, 2, 3, padding=1)

    def forward(self, kspace_input, image_input):
        f = self.enc_k(kspace_input)
        i = self.enc_i(image_input)
        return self.output((f + i) / 2)

class DUNDD(nn.Module):
    def __init__(self, num_iterations=5, lambda_dc=0.5, num_channels=64):
        super().__init__()
        self.num_iterations = num_iterations
        self.lambda_dc = lambda_dc
        self.dual_cnn = SimpleDualDomainCNN(num_channels=num_channels)

    def forward(self, masked_kspace, mask):
        B, _, H, W = masked_kspace.shape
        mask_bchw = mask.unsqueeze(1)
        kspace = masked_kspace.clone()
        for _ in range(self.num_iterations):
            kc = torch.view_as_complex(kspace.permute(0, 2, 3, 1).contiguous())
            kc = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kc, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
            img_2ch = torch.view_as_real(kc).permute(0, 3, 1, 2).contiguous()
            update_2ch = self.dual_cnn(kspace, img_2ch)
            uc = torch.view_as_complex(update_2ch.permute(0, 2, 3, 1).contiguous())
            update_kspace = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(uc, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
            measured = torch.view_as_complex(masked_kspace.permute(0, 2, 3, 1).contiguous())
            # Soft data consistency on sampled locations + hard replacement.
            # (measured - kc_new) is zero where mask=1 after replacement, so the
            # previous extra DC line was a no-op; apply soft DC before replacement.
            kc_new = update_kspace + self.lambda_dc * (measured - update_kspace) * mask_bchw[:, 0]
            kc_new = measured * mask_bchw[:, 0] + kc_new * (1 - mask_bchw[:, 0])
            kspace = torch.view_as_real(kc_new).permute(0, 3, 1, 2).contiguous()
        return kspace

class UNetBaseline(nn.Module):
    """Standard U-Net baseline — image domain only."""
    def __init__(self, features=32):
        super().__init__()
        def block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            )
        self.enc1 = block(2, features)
        self.enc2 = block(features, features * 2)
        self.enc3 = block(features * 2, features * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(features * 4, features * 8)
        self.up3 = nn.ConvTranspose2d(features * 8, features * 4, 2, 2)
        self.dec3 = block(features * 8, features * 4)
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, 2, 2)
        self.dec2 = block(features * 4, features * 2)
        self.up1 = nn.ConvTranspose2d(features * 2, features, 2, 2)
        self.dec1 = block(features * 2, features)
        self.final = nn.Conv2d(features, 2, 1)

    def forward(self, masked_kspace, mask):
        kc = masked_kspace[:, 0] + 1j * masked_kspace[:, 1]
        img = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kc, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
        x = torch.stack([img.real, img.imag], dim=1)
        h, w = x.shape[-2], x.shape[-1]
        pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
        x = F.pad(x, (0, pad_w, 0, pad_h))
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out_img = self.final(d1)[:, :, :h, :w]
        out_c = out_img[:, 0] + 1j * out_img[:, 1]
        out_k = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(out_c, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
        return torch.stack([out_k.real, out_k.imag], dim=1)

class CascadeNet(nn.Module):
    """Cascaded CNN with hard data consistency."""
    def __init__(self, num_cascades=5, features=32):
        super().__init__()
        self.cascades = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, features, 3, padding=1), nn.ReLU(),
                nn.Conv2d(features, features, 3, padding=1), nn.ReLU(),
                nn.Conv2d(features, features, 3, padding=1), nn.ReLU(),
                nn.Conv2d(features, 2, 3, padding=1),
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

# ── 3. MoDL ────────────────────────────────────────────────
class MoDL(nn.Module):
    """
    Model-Based Deep Learning with conjugate gradient data consistency.
    Aggarwal, Mani, Jacob. IEEE TMI 2019. arxiv.org/abs/1712.02862

    At each outer iteration, solves:
        (A^H A + λI) x = A^H y + λ D_w(x)
    via conjugate gradient, where D_w is a learned CNN denoiser.

    Fixes vs the broken version:
      1. Batch-aware CG  — inner products sum over H×W only, not over B.
         Without this, alpha is one shared scalar for the whole batch,
         which makes CG produce garbage.
      2. Correct lambda init — softplus inverse so lam ≈ lambda_reg at t=0.
      3. CG warm-starts from current x  (not from the RHS).
    """
    def __init__(self, num_iterations=8, num_cg_steps=6, lambda_reg=0.05):
        super().__init__()
        self.K        = num_iterations
        self.cg_steps = num_cg_steps

        # Softplus inverse: softplus(inv) ≈ lambda_reg at initialisation.
        # softplus(0.05) ≈ 0.744 — that was the original bug.
        # log(exp(x) - 1) is the inverse of softplus.
        import math
        inv_val = math.log(math.exp(lambda_reg) - 1 + 1e-8)
        self._lam_raw = nn.Parameter(torch.tensor(inv_val, dtype=torch.float32))

        self.denoiser = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 2,  3, padding=1),
        )

    @property
    def lam(self):
        return F.softplus(self._lam_raw) + 1e-4

    # ── helpers ─────────────────────────────────────────────
    def _r2c(self, x):
        """(B,2,H,W) float → (B,H,W) complex"""
        return torch.view_as_complex(x.permute(0,2,3,1).contiguous())

    def _c2r(self, x):
        """(B,H,W) complex → (B,2,H,W) float"""
        return torch.view_as_real(x).permute(0,3,1,2).contiguous()

    def _A(self, img_c, mask):
        """Forward operator: image → masked k-space."""
        k = torch.fft.fftshift(
                torch.fft.fft2(
                    torch.fft.ifftshift(img_c, dim=(-2,-1)),
                    norm='ortho'), dim=(-2,-1))
        return k * mask          # mask: (B,H,W) broadcasts over complex (B,H,W)

    def _AH(self, k_c, mask):
        """Adjoint operator: masked k-space → image."""
        return torch.fft.fftshift(
                   torch.fft.ifft2(
                       torch.fft.ifftshift(k_c * mask, dim=(-2,-1)),
                       norm='ortho'), dim=(-2,-1))

    # ── batch-aware conjugate gradient ──────────────────────
    def _cg(self, rhs_c, mask, x0_c):
        """
        Solve (A^H A + λI) x = rhs_c using conjugate gradient.

        Args:
            rhs_c : (B, H, W) complex  — right-hand side
            mask  : (B, H, W) float    — undersampling mask
            x0_c  : (B, H, W) complex  — warm-start (current x)

        The critical fix: all inner products sum over H×W only,
        keeping the batch dimension B intact so each sample in the
        batch gets its own alpha and beta scalars.
        """
        def dot(a, b):
            # (B,) — sum over spatial dims only, NOT over batch
            return (a.conj() * b).real.sum(dim=(-2, -1))

        def Ax_fn(v):
            return self._AH(self._A(v, mask), mask) + self.lam * v

        x  = x0_c.clone()
        r  = rhs_c - Ax_fn(x)
        p  = r.clone()
        rs = dot(r, r)                          # (B,)

        for _ in range(self.cg_steps):
            Ap    = Ax_fn(p)
            denom = dot(p, Ap)                  # (B,)
            alpha = (rs / (denom + 1e-8))       # (B,)
            alpha = alpha.view(-1, 1, 1)        # → (B,1,1) to broadcast over H×W

            x     = x + alpha * p
            r     = r - alpha * Ap
            rs_new = dot(r, r)                  # (B,)

            if rs_new.max().item() < 1e-10:
                break

            beta = (rs_new / (rs + 1e-8)).view(-1, 1, 1)
            p    = r + beta * p
            rs   = rs_new

        return x

    # ── forward pass ────────────────────────────────────────
    def forward(self, masked_kspace, mask):
        """
        masked_kspace : (B, 2, H, W) — zero-filled input
        mask          : (B, H, W)    — binary undersampling mask
        Returns       : (B, 2, H, W) — predicted k-space
        """
        y_c  = self._r2c(masked_kspace)      # (B, H, W) complex
        AHy  = self._AH(y_c, mask)           # (B, H, W) — initialise in image domain
        x    = AHy.clone()

        for _ in range(self.K):
            Dw_x   = self.denoiser(self._c2r(x))    # (B,2,H,W) denoiser output
            Dw_x_c = self._r2c(Dw_x)               # (B,H,W) complex

            # RHS of the linear system for this outer iteration
            rhs = AHy + self.lam * Dw_x_c

            # CG warm-starts from current x (not from rhs — that was bug 3)
            x = self._cg(rhs, mask, x)

        # Convert final image-domain x back to k-space to match pipeline output format
        out_k = torch.fft.fftshift(
                    torch.fft.fft2(
                        torch.fft.ifftshift(x, dim=(-2,-1)),
                        norm='ortho'), dim=(-2,-1))
        return self._c2r(out_k)             # (B, 2, H, W)


# ── 4. E2E-VarNet ──────────────────────────────────────────
class VarNetBlock(nn.Module):
    def __init__(self, features=32):
        super().__init__()
        def block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(),
                nn.Conv2d(oc, oc, 3, padding=1), nn.ReLU(),
            )
        self.enc1 = block(2, features)
        self.enc2 = block(features, features*2)
        self.pool = nn.MaxPool2d(2)
        self.bot  = block(features*2, features*2)
        # Use Upsample instead of ConvTranspose2d — no shape mismatch
        self.up1  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=False),
            nn.Conv2d(features*2, features, 3, padding=1)
        )
        self.dec1 = block(features*2, features)
        self.out  = nn.Conv2d(features, 2, 1)
        self.eta  = nn.Parameter(torch.tensor(1.0))

    def forward(self, kspace, masked_kspace, mask):
        mask_exp = mask.unsqueeze(1).expand_as(kspace)
        dc_grad  = (kspace - masked_kspace) * mask_exp
        e1 = self.enc1(kspace)
        e2 = self.enc2(self.pool(e1))
        b  = self.bot(self.pool(e2))
        u1 = self.up1(b)
        # Safe concat — pad if sizes differ by 1 pixel
        if u1.shape != e1.shape:
            u1 = F.pad(u1, [0, e1.shape[-1] - u1.shape[-1],
                            0, e1.shape[-2] - u1.shape[-2]])
        d1  = self.dec1(torch.cat([u1, e1], dim=1))
        ref = self.out(d1)
        return kspace - self.eta * dc_grad - ref

class E2EVarNet(nn.Module):
    """
    End-to-End Variational Network (single-coil adaptation).
    Sriram et al. MICCAI 2020. arxiv.org/abs/2004.06688
    
    Refines k-space through cascaded gradient descent + U-Net.
    Each cascade: gradient step on data fidelity + U-Net correction.
    
    Input:  (B, 2, H, W) masked k-space
    Output: (B, 2, H, W) reconstructed k-space
    """
    def __init__(self, num_cascades=8, features=32):
        super().__init__()
        self.cascades = nn.ModuleList([
            VarNetBlock(features) for _ in range(num_cascades)
        ])

    def forward(self, masked_kspace, mask):
        kspace = masked_kspace.clone()
        for cascade in self.cascades:
            kspace = cascade(kspace, masked_kspace, mask)
        return kspace