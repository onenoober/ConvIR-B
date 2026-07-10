import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ConvIR import build_net


D7C_FIXED_THRESHOLD = 0.5773006677627563
GRAY_WEIGHTS = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)


class DensityNeedHead(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class MultiContextNeedHead(nn.Module):
    def __init__(self, in_channels=234, out_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


def load_checkpoint_state(path, map_location):
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and 'model' in state:
        return state['model']
    return state


def partial_load_model_state(model, state, allowed_missing):
    model_state = model.state_dict()
    loaded = {}
    unexpected = []
    shape_mismatch = []

    for key, value in state.items():
        if key not in model_state:
            unexpected.append(key)
        elif tuple(model_state[key].shape) != tuple(value.shape):
            shape_mismatch.append(
                {
                    'key': key,
                    'checkpoint_shape': list(value.shape),
                    'model_shape': list(model_state[key].shape),
                }
            )
        else:
            loaded[key] = value

    missing = [key for key in model_state if key not in loaded]
    bad_missing = [key for key in missing if key not in allowed_missing]
    if unexpected or shape_mismatch or bad_missing:
        raise RuntimeError(
            'partial-load failed: '
            f'unexpected={unexpected}, shape_mismatch={shape_mismatch}, '
            f'bad_missing={bad_missing}, missing={missing}'
        )

    model_state.update(loaded)
    model.load_state_dict(model_state, strict=True)
    return {
        'loaded_count': len(loaded),
        'missing_candidate_keys': missing,
        'unexpected_keys': unexpected,
        'shape_mismatch': shape_mismatch,
        'bad_missing': bad_missing,
    }


def load_density_head(path, device):
    head = DensityNeedHead(1).to(device)
    checkpoint = torch.load(path, map_location=device)
    head.load_state_dict(checkpoint['state_dict'])
    head.eval()
    for param in head.parameters():
        param.requires_grad_(False)
    return head


def load_d7c_head(path, device):
    head = MultiContextNeedHead().to(device)
    checkpoint = torch.load(path, map_location=device)
    head.load_state_dict(checkpoint['state_dict'])
    head.eval()
    for param in head.parameters():
        param.requires_grad_(False)
    return head


def density_pred_from_head(head, res1):
    return torch.sigmoid(head(res1))


def convir_a0_context(model, density_head, x):
    x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
    x_4 = F.interpolate(x_2, scale_factor=0.5, mode='bilinear', align_corners=False)
    z2 = model.SCM2(x_2)
    z4 = model.SCM1(x_4)
    x0 = model.feat_extract[0](x)
    res1 = model.Encoder[0](x0)
    z = model.feat_extract[1](res1)
    z = model.FAM2(z, z2)
    res2 = model.Encoder[1](z)
    z = model.feat_extract[2](res2)
    z = model.FAM1(z, z4)
    bottleneck = model.Encoder[2](z)

    z = model.Decoder[0](bottleneck)
    z = model.feat_extract[3](z)
    z = torch.cat([z, res2], dim=1)
    z = model.Convs[0](z)
    z = model.Decoder[1](z)
    z = model.feat_extract[4](z)
    z = torch.cat([z, res1], dim=1)
    z = model.Convs[1](z)
    z = model.Decoder[2](z)
    z = model.feat_extract[5](z)
    a0 = torch.clamp(z + x, 0, 1)

    res2_up = F.interpolate(res2, size=res1.shape[-2:], mode='bilinear', align_corners=False)
    bottleneck_up = F.interpolate(bottleneck, size=res1.shape[-2:], mode='bilinear', align_corners=False)
    density_pred = density_pred_from_head(density_head, res1)
    return torch.cat([res1, res2_up, bottleneck_up, x, a0, (x - a0).abs(), density_pred], dim=1)


def predict_d7c(head, context):
    logits = head(context)
    probs = [torch.sigmoid(logits[:, index : index + 1]) for index in range(4)]
    return torch.stack(probs, dim=0).mean(dim=0), logits


class D7CGateProducer(nn.Module):
    def __init__(self, base_model, density_head, d7c_head, threshold=D7C_FIXED_THRESHOLD):
        super().__init__()
        self.base_model = base_model
        self.density_head = density_head
        self.d7c_head = d7c_head
        self.threshold = float(threshold)
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        context = convir_a0_context(self.base_model, self.density_head, x)
        score, logits = predict_d7c(self.d7c_head, context)
        gate = (score >= self.threshold).to(dtype=x.dtype)
        return gate.detach(), score.detach(), logits.detach()


def build_d7c_gate_producer(args, device):
    if args.d7c_gate_mode == 'none':
        return None
    if args.d7c_gate_mode != 'd7c_fixed':
        raise ValueError(f'Unsupported d7c_gate_mode: {args.d7c_gate_mode}')
    if args.fam_mode != 'fam2_d7c_noop':
        raise ValueError('--d7c_gate_mode d7c_fixed requires --fam_mode fam2_d7c_noop')
    if not args.d7c_base_checkpoint:
        raise ValueError('--d7c_base_checkpoint is required for d7c_fixed gate mode')
    if not args.d7c_density_artifact:
        raise ValueError('--d7c_density_artifact is required for d7c_fixed gate mode')
    if not args.d7c_need_artifact:
        raise ValueError('--d7c_need_artifact is required for d7c_fixed gate mode')

    base_model = build_net(args.version, args.data, 'original').to(device).eval()
    base_state = load_checkpoint_state(args.d7c_base_checkpoint, device)
    base_model.load_state_dict(base_state, strict=True)
    for param in base_model.parameters():
        param.requires_grad_(False)

    density_head = load_density_head(args.d7c_density_artifact, device)
    d7c_head = load_d7c_head(args.d7c_need_artifact, device)
    return D7CGateProducer(base_model, density_head, d7c_head, args.d7c_threshold)


def get_d7c_gate(args, input_img):
    producer = getattr(args, 'd7c_gate_producer', None)
    if producer is None:
        return None
    gate, _, _ = producer(input_img)
    return gate


def forward_with_optional_d7c(model, args, input_img):
    gate = get_d7c_gate(args, input_img)
    if gate is None:
        return model(input_img)
    return model(input_img, d7c_gate=gate)


def collect_modulation_stats_with_optional_d7c(model, args, input_img):
    gate = get_d7c_gate(args, input_img)
    if gate is None:
        return model.collect_modulation_stats(input_img)
    return model.collect_modulation_stats(input_img, d7c_gate=gate)
