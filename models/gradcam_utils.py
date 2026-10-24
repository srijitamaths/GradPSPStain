import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM that can be used in two modes:
      - detach=True  : visualization (no gradients flow out of this module)
      - detach=False : differentiable CAM for using CAM inside a loss on input (e.g., to guide G)

    Important:
      - For detach=False, activations and gradients MUST NOT be detached.
      - Use register_full_backward_hook when available.
    """
    def __init__(self, model, target_layer: str, detach: bool = True, use_relu: bool = True):
        self.model = model
        self.target_layer = target_layer
        self.detach = detach
        self.use_relu = use_relu

        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            # out: [B, C, H, W] typically
            self.activations = out.detach() if self.detach else out

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] matches activation shape
            g = grad_output[0]
            self.gradients = g.detach() if self.detach else g

        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        if target_module is None:
            raise ValueError(f"GradCAM target_layer='{self.target_layer}' not found in model.named_modules().")

        self.hook_handles.append(target_module.register_forward_hook(forward_hook))

        # Prefer full backward hook if available
        if hasattr(target_module, "register_full_backward_hook"):
            self.hook_handles.append(target_module.register_full_backward_hook(backward_hook))
        else:
            self.hook_handles.append(target_module.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

    @torch.no_grad()
    def _normalize_cam_nograd(self, cam):
        # cam: [B,1,H,W]
        B = cam.size(0)
        cam_ = cam.view(B, -1)
        cam_min = cam_.min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        cam_max = cam_.max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        return (cam - cam_min) / (cam_max - cam_min + 1e-8)

    def _normalize_cam_grad(self, cam):
        # Differentiable per-sample normalization
        B = cam.size(0)
        cam_ = cam.view(B, -1)
        cam_min = cam_.min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        cam_max = cam_.max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        return (cam - cam_min) / (cam_max - cam_min + 1e-8)

    def __call__(self, x, class_idx=None):
        """
        x: [B,3,H,W]
        class_idx: LongTensor [B] or None. If None, uses argmax of logits.
        returns cam: [B,1,H,W]
        """
        self.gradients = None
        self.activations = None

        # Forward
        out = self.model(x)

        # If out is patch map [B,C,h,w], pool to [B,C]
        if out.dim() == 4:
            out = F.adaptive_avg_pool2d(out, (1, 1)).squeeze(-1).squeeze(-1)

        if out.dim() != 2:
            raise ValueError(f"GradCAM expected model output [B,C] or [B,C,h,w], got {tuple(out.shape)}")

        B, C = out.shape

        # Choose class indices
        if class_idx is None:
            class_idx = out.argmax(dim=1)
        else:
            class_idx = class_idx.view(-1).long()
            class_idx = torch.clamp(class_idx, 0, C - 1)

        # Build one-hot gradient signal
        one_hot = torch.zeros_like(out)
        one_hot.scatter_(1, class_idx.view(-1, 1), 1.0)

        # Backward to get gradients at target layer
        self.model.zero_grad(set_to_none=True)
        out.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks did not capture gradients/activations. Check target_layer name.")

        # Grad-CAM: channel weights by global-average-pooling grads
        w = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cam = (w * self.activations).sum(dim=1, keepdim=True)  # [B,1,h,w]

        if self.use_relu:
            cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Normalize
        if self.detach:
            cam = self._normalize_cam_nograd(cam)
        else:
            cam = self._normalize_cam_grad(cam)

        return cam
