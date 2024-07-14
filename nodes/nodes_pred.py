import sys

import comfy
import torch
import latent_preview
from tqdm import tqdm

try:
    from comfy.samplers import calc_cond_batch
except ImportError:
    from comfy.samplers import calc_cond_uncond_batch

    def calc_cond_batch(model, conds, x_in, timestep, model_options):
        outputs = []

        index = 0
        while index < len(conds) - 1:
            outputs.extend(calc_cond_uncond_batch(model, conds[index], conds[index + 1], x_in, timestep, model_options))
            index += 2

        if index < len(conds):
            outputs.append(calc_cond_uncond_batch(model, conds[index], None, x_in, timestep, model_options)[0])

        return outputs

try:
    from comfy.sampler_helpers import convert_cond
except ImportError:
    from comfy.sample import convert_cond

try:
    from comfy.sampler_helpers import get_models_from_cond
except ImportError:
    from comfy.sample import get_models_from_cond

try:
    from comfy.sampler_helpers import prepare_mask
except ImportError:
    from comfy.sample import prepare_mask

class CustomNoisePredictor(torch.nn.Module):
    def __init__(
            self,
            model,
            pred,
            preds,
            conds,
            positive_pred,
            negative_pred,
            empty_pred):
        super().__init__()

        self.inner_model = model
        self.pred = pred
        self.preds = preds
        self.conds = conds
        self.positive_pred = positive_pred
        self.negative_pred = negative_pred
        self.empty_pred = empty_pred

    def apply_model(self, x, timestep, cond=None, uncond=None, cond_scale=None, model_options=None, seed=None):
        if model_options is None:
            model_options = {}

        for pred in self.preds:
            pred.begin_sample()

        try:
            result = self.pred.predict_noise(x, timestep, self.inner_model, self.conds, model_options, seed)

            # for improved compatibility with ComfyUI extensions that use it,
            # we call any sampler_post_cfg_functions
            # (required for CFG++ samplers, for example)
            if (cfg_hooks := model_options.get("sampler_post_cfg_function")) is not None:
                args = (x, timestep, self.inner_model, self.conds, model_options, seed)

                # TODO: would it be more performant to combine into a single
                #       call to calc_cond_batch? How?
                # TODO: it'd be nice to avoid computing cond/uncond/empty_cond
                if self.positive_pred is not None:
                    cond_pred = self.positive_pred.predict_noise(*args)
                    cond = x - cond_pred
                else:
                    cond = cond_pred = None
                
                if self.negative_pred is not None:
                    uncond_pred = self.negative_pred.predict_noise(*args)
                    uncond = x - uncond_pred
                else:
                    uncond = uncond_pred = None
                    
                if self.empty_pred is not None:
                    empty_pred = self.empty_pred.predict_noise(*args)
                    empty_cond = x - empty_pred
                else:
                    empty_pred = empty_cond = None
                
                for f in cfg_hooks:
                    args = {
                        "denoised": result,
                        "cond": cond,
                        "uncond": uncond,
                        "model": self.inner_model,
                        "uncond_denoised": uncond_pred,
                        "cond_denoised": cond_pred,
                        "sigma": timestep,
                        "model_options": model_options,
                        "input": x,
                        # not in the original call in samplers.py:cfg_function, but made available for future hooks
                        "empty_cond": empty_cond,
                        "empty_cond_denoised": empty_pred,
                    }
                    result = f(args)
        finally:
            for pred in self.preds:
                pred.end_sample()
        
        return result

    def forward(self, *args, **kwargs):
        return self.apply_model(*args, **kwargs)

class SamplerCustomPrediction:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": ("BOOLEAN", {"default": True}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "noise_prediction": ("PREDICTION",),
            }, "optional": {
                "positive_pred": ("PREDICTION",),
                "negative_pred": ("PREDICTION",),
                "empty_pred": ("PREDICTION",),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/prediction"

    def sample(
            self,
            model,
            add_noise,
            noise_seed,
            sampler,
            sigmas,
            latent_image,
            noise_prediction,
            positive_pred=None,
            negative_pred=None,
            empty_pred=None):
        latent_samples = latent_image["samples"]

        if not add_noise:
            torch.manual_seed(noise_seed)

            noise = torch.zeros(
                latent_samples.size(),
                dtype=latent_samples.dtype,
                layout=latent_samples.layout,
                device="cpu"
            )
        else:
            batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
            noise = comfy.sample.prepare_noise(latent_samples, noise_seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent_image:
            noise_mask = latent_image["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

        samples = sample_pred(
            model, noise, noise_prediction, sampler, sigmas, latent_samples,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=noise_seed,
            positive_pred=positive_pred,
            negative_pred=negative_pred,
            empty_pred=empty_pred,
        )

        out = latent_image.copy()
        out["samples"] = samples

        if "x0" in x0_output:
            out_denoised = latent_image.copy()
            out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out

        return (out, out_denoised)

def sample_pred(
    model,
    noise,
    predictor,
    sampler,
    sigmas,
    latent,
    noise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
    positive_pred=None,
    negative_pred=None,
    empty_pred=None,
):
    if noise_mask is not None:
        noise_mask = prepare_mask(noise_mask, noise.shape, model.load_device)

    dtype = model.model_dtype()
    device = model.load_device

    models = predictor.get_models()
    conds = predictor.get_conds()
    preds = predictor.get_preds()

    n_samples = 0
    for pred in preds:
        n_samples += pred.n_samples()

    inference_memory = model.memory_required([noise.shape[0] * n_samples] + list(noise.shape[1:]))

    for addtl in models:
        if "inference_memory_requirements" in addtl:
            inference_memory += addtl.inference_memory_requirements(dtype)

    comfy.model_management.load_models_gpu(models | set([model]), inference_memory)

    noise = noise.to(device)
    latent = latent.to(device)
    sigmas = sigmas.to(device)

    for name, cond in conds.items():
        conds[name] = cond[:]

    for cond in conds.values():
        comfy.samplers.resolve_areas_and_cond_masks(cond, noise.shape[2], noise.shape[3], device)

    for cond in conds.values():
        comfy.samplers.calculate_start_end_timesteps(model.model, cond)

    if latent is not None:
        latent = model.model.process_latent_in(latent)

    if hasattr(model.model, "extra_conds"):
        for name, cond in conds.items():
            conds[name] = comfy.samplers.encode_model_conds(
                model.model.extra_conds,
                cond, noise, device, name,
                latent_image=latent,
                denoise_mask=noise_mask,
                seed=seed
            )

    # ensure each cond area corresponds with all other areas
    for name1, cond1 in conds.items():
        for name2, cond2 in conds.items():
            if name2 == name1:
                continue

            for c1 in cond1:
                comfy.samplers.create_cond_with_same_area_if_none(cond2, c1)

    # TODO: support controlnet how?

    predictor_model = CustomNoisePredictor(model.model, predictor, preds, conds, positive_pred, negative_pred, empty_pred)
    extra_args = { "model_options": model.model_options, "seed": seed }

    for pred in preds:
        pred.begin_sampling()

    try:
        samples = sampler.sample(predictor_model, sigmas, extra_args, callback, noise, latent, noise_mask, disable_pbar)
    finally:
        for pred in preds:
            pred.end_sampling()

    samples = model.model.process_latent_out(samples.to(torch.float32))
    samples = samples.to(comfy.model_management.intermediate_device())

    comfy.sample.cleanup_additional_models(models)
    return samples

class NoisePredictor:
    OUTPUTS = { "prediction": "PREDICTION" }

    def get_models(self):
        """Returns all additional models transitively used by this predictor."""
        return set()

    def get_conds(self):
        """Returns all conditionings transitively defined by this predictor."""
        return {}

    def get_preds(self):
        """Returns all NoisePredcitors transitively used by this predictor, including itself."""
        return {self}

    def n_samples(self):
        """Returns the number of times a model will be sampled directly by this predictor."""
        return 0

    def predict_noise(self, x, timestep, model, conds, model_options, seed):
        raise NotImplementedError

    def begin_sampling(self):
        """Called when sampling begins for a batch."""
        pass

    def begin_sample(self):
        """Called when one sampling step begins."""
        pass

    def end_sample(self):
        """Called when one sampling step ends."""
        pass

    def end_sampling(self):
        """Called when sampling completes for a batch."""
        pass

    def get_models_from_conds(self):
        models = set()

        for cond in self.get_conds():
            for cnet in get_models_from_cond(cond, "control"):
                models |= cnet.get_models()

            for gligen in get_models_from_cond(cond, "gligen"):
                models |= [x[1] for x in gligen]

        return models

    @staticmethod
    def merge_models(*args):
        merged = set()

        for arg in args:
            if arg is None:
                continue

            if isinstance(arg, NoisePredictor):
                merged |= arg.get_models()
            elif isinstance(arg, set):
                merged |= arg
            else:
                merged.add(arg)

        return merged

    @staticmethod
    def merge_conds(*args):
        merged = {}

        for arg in args:
            if arg is None:
                continue

            if isinstance(arg, NoisePredictor):
                arg = arg.get_conds()

            for name, cond in arg.items():
                if name not in merged:
                    merged[name] = cond
                elif merged[name] != cond:
                    raise RuntimeError(f"Conditioning \"{name}\" is not unique.")

        return merged

    def merge_preds(self, *args):
        merged = {self}

        for arg in args:
            if arg is not None and arg not in merged:
                merged |= arg.get_preds()

        return merged

class CachingNoisePredictor(NoisePredictor):
    def __init__(self):
        self.cached_prediction = None

    def predict_noise(self, x, timestep, model, conds, model_options, seed):
        if self.cached_prediction is None:
            self.cached_prediction = self.predict_noise_uncached(x, timestep, model, conds, model_options, seed)

        return self.cached_prediction

    def predict_noise_uncached(self, x, timestep, model, conds, model_options, seed):
        raise NotImplementedError

    def begin_sample(self):
        self.cached_prediction = None

    def end_sample(self):
        self.cached_prediction = None

class ConditionedPredictor(CachingNoisePredictor):
    INPUTS = {
        "required": {
            "conditioning": ("CONDITIONING",),
            "name": ("STRING", {
                "multiline": False,
                "default": "positive"
            }),
        }
    }

    def __init__(self, conditioning, name):
        super().__init__()

        self.cond = convert_cond(conditioning)
        self.cond_name = name

    def get_conds(self):
        return { self.cond_name: self.cond }

    def get_models(self):
        return self.get_models_from_conds()

    def n_samples(self):
        return 1

    def predict_noise_uncached(self, x, timestep, model, conds, model_options, seed):
        return calc_cond_batch(model, [conds[self.cond_name]], x, timestep, model_options)[0]

class CombinePredictor(NoisePredictor):
    INPUTS = {
        "required": {
            "prediction_A": ("PREDICTION",),
            "prediction_B": ("PREDICTION",),
            "operation": ([
                "A + B",
                "A - B",
                "A * B",
                "A / B",
                "A proj B",
                "A oproj B",
                "min(A, B)",
                "max(A, B)",
            ],)
        }
    }

    def __init__(self, prediction_A, prediction_B, operation):
        self.lhs = prediction_A
        self.rhs = prediction_B

        match operation:
            case "A + B":
                self.op = torch.add
            case "A - B":
                self.op = torch.sub
            case "A * B":
                self.op = torch.mul
            case "A / B":
                self.op = torch.div
            case "A proj B":
                self.op = proj
            case "A oproj B":
                self.op = oproj
            case "min(A, B)":
                self.op = torch.minimum
            case "max(A, B)":
                self.op = torch.maximum
            case _:
                raise RuntimeError(f"unsupported operation: {self.op}")

    def get_conds(self):
        return self.merge_conds(self.lhs, self.rhs)

    def get_models(self):
        return self.merge_models(self.lhs, self.rhs)

    def get_preds(self):
        return self.merge_preds(self.lhs, self.rhs)

    def predict_noise(self, x, timestep, model, conds, model_options, seed):
        lhs = self.lhs.predict_noise(x, timestep, model, conds, model_options, seed)
        rhs = self.rhs.predict_noise(x, timestep, model, conds, model_options, seed)
        return self.op(lhs, rhs)

class InterpolatePredictor(NoisePredictor):
    INPUTS = {
        "required": {
            "prediction_A": ("PREDICTION",),
            "prediction_B": ("PREDICTION",),
            "scale_B": ("FLOAT", {"default": 0.5, "step": 0.01, "min": 0.0, "max": 1.0})
        }
    }

    def __init__(self, prediction_A, prediction_B, scale_B):
        self.lhs = prediction_A
        self.rhs = prediction_B
        self.lerp = scale_B

    def get_conds(self):
        return self.merge_conds(self.lhs, self.rhs)

    def get_models(self):
        return self.merge_models(self.lhs, self.rhs)

    def get_preds(self):
        return self.merge_preds(self.lhs, self.rhs)

    def predict_noise(self, x, timestep, model, conds, model_options, seed):
        match self.lerp:
            case 0.0:
                return self.lhs.predict_noise(x, timestep, model, conds, model_options, seed)

            case 1.0:
                return self.rhs.predict_noise(x, timestep, model, conds, model_options, seed)

            case _:
                return torch.lerp(
                    self.lhs.predict_noise(x, timestep, model, conds, model_options, seed),
                    self.rhs.predict_noise(x, timestep, model, conds, model_options, seed),
                    self.lerp
                )

class SwitchPredictor(NoisePredictor):
    """Switches predictions for specified sigmas"""
    INPUTS = {
        "required": {
            "prediction_A": ("PREDICTION",),
            "prediction_B": ("PREDICTION",),
            "sigmas_B": ("SIGMAS",),
        }
    }

    def __init__(self, prediction_A, prediction_B, sigmas_B):
        self.lhs = prediction_A
        self.rhs = prediction_B
        self.sigmas = sigmas_B

    def get_conds(self):
        return self.merge_conds(self.lhs, self.rhs)

    def get_models(self):
        return self.merge_models(self.lhs, self.rhs)

    def get_preds(self):
        return self.merge_preds(self.lhs, self.rhs)

    def predict_noise(self, x, sigma, model, conds, model_options, seed):
        rhs_mask = torch.isin(sigma.cpu(), self.sigmas)
        lhs_inds = torch.argwhere(~rhs_mask).squeeze(1)
        rhs_inds = torch.argwhere(rhs_mask).squeeze(1)

        if len(lhs_inds) == 0:
            return self.rhs.predict_noise(x, sigma, model, conds, model_options, seed)

        if len(rhs_inds) == 0:
            return self.lhs.predict_noise(x, sigma, model, conds, model_options, seed)

        preds = torch.empty_like(x)
        preds[lhs_inds] = self.lhs.predict_noise(
            x[lhs_inds],
            sigma[lhs_inds],
            model,
            conds,
            model_options,
            seed
        )
        preds[rhs_inds] = self.rhs.predict_noise(
            x[rhs_inds],
            sigma[rhs_inds],
            model,
            conds,
            model_options,
            seed
        )

        return preds

class EarlyMiddleLatePredictor(NoisePredictor):
    """Switches predictions based on an early-middle-late schedule."""
    INPUTS = {
        "required": {
            "sigmas": ("SIGMAS",),
            "early_prediction": ("PREDICTION",),
            "middle_prediction": ("PREDICTION",),
            "late_prediction": ("PREDICTION",),
            "early_steps": ("INT", { "min": 0, "max": 1000, "default": 1 }),
            "late_steps": ("INT", { "min": 0, "max": 1000, "default": 5 }),
        }
    }

    def __init__(self, early_prediction, middle_prediction, late_prediction, sigmas, early_steps, late_steps):
        self.early_pred = early_prediction
        self.middle_pred = middle_prediction
        self.late_pred = late_prediction

        late_step = -late_steps - 1

        self.early_sigmas = sigmas[:early_steps]
        self.middle_sigmas = sigmas[early_steps:late_step]
        self.late_sigmas = sigmas[late_step:-1]

        if torch.any(torch.isin(self.early_sigmas, self.middle_sigmas)) \
            or torch.any(torch.isin(self.early_sigmas, self.late_sigmas)) \
            or torch.any(torch.isin(self.middle_sigmas, self.late_sigmas)) \
        :
            raise ValueError("Sigma schedule is ambiguous.")

    def get_conds(self):
        return self.merge_conds(self.early_pred, self.middle_pred, self.late_pred)

    def get_models(self):
        return self.merge_models(self.early_pred, self.middle_pred, self.late_pred)

    def get_preds(self):
        return self.merge_preds(self.early_pred, self.middle_pred, self.late_pred)

    def predict_noise(self, x, sigma, model, conds, model_options, seed):
        cpu_sigmas = sigma.cpu()
        early_inds = torch.argwhere(torch.isin(cpu_sigmas, self.early_sigmas)).squeeze(1)
        middle_inds = torch.argwhere(torch.isin(cpu_sigmas, self.middle_sigmas)).squeeze(1)
        late_inds = torch.argwhere(torch.isin(cpu_sigmas, self.late_sigmas)).squeeze(1)

        preds = torch.empty_like(x)

        assert (len(early_inds) + len(middle_inds) + len(late_inds)) == len(x)

        if len(early_inds) > 0:
            preds[early_inds] = self.early_pred.predict_noise(
                x[early_inds],
                sigma[early_inds],
                model,
                conds,
                model_options,
                seed
            )

        if len(middle_inds) > 0:
            preds[middle_inds] = self.middle_pred.predict_noise(
                x[middle_inds],
                sigma[middle_inds],
                model,
                conds,
                model_options,
                seed
            )

        if len(late_inds) > 0:
            preds[late_inds] = self.late_pred.predict_noise(
                x[late_inds],
                sigma[late_inds],
                model,
                conds,
                model_options,
                seed
            )

        return preds

class ScaledGuidancePredictor(NoisePredictor):
    """Implements A * scale + B"""
    INPUTS = {
        "required": {
            "guidance": ("PREDICTION",),
            "baseline": ("PREDICTION",),
            "guidance_scale": ("FLOAT", {
                "default": 6.0,
                "step": 0.1,
                "min": 0.0,
                "max": 100.0,
            }),
            "stddev_rescale": ("FLOAT", {
                "default": 0.0,
                "step": 0.1,
                "min": 0.0,
                "max": 1.0,
            }),
        }
    }

    def __init__(self, guidance, baseline, guidance_scale, stddev_rescale):
        self.lhs = guidance
        self.rhs = baseline
        self.scale = guidance_scale
        self.rescale = stddev_rescale

    def get_conds(self):
        return self.merge_conds(self.lhs, self.rhs)

    def get_models(self):
        return self.merge_models(self.lhs, self.rhs)

    def get_preds(self):
        return self.merge_preds(self.lhs, self.rhs)

    def predict_noise(self, x, sigma, model, conds, model_options, seed):
        g = self.lhs.predict_noise(x, sigma, model, conds, model_options, seed)
        b = self.rhs.predict_noise(x, sigma, model, conds, model_options, seed)

        if self.rescale <= 0.0:
            return g * self.scale + b

        # CFG Rescale https://arxiv.org/pdf/2305.08891.pdf Sec. 3.4
        # Originally in eps -> v space, but now very clean in x0 thanks to Gaeros.

        sigma = sigma.view(sigma.shape[:1] + (1,) * (b.ndim - 1))
        x_norm = x * (1. / (sigma.square() + 1.0))

        x0 = x_norm - (g + b)
        x0_cfg = x_norm - (g * self.scale + b)

        std_x0 = torch.std(x0, dim=(1, 2, 3), keepdim=True)
        std_x0_cfg = torch.std(x0_cfg, dim=(1, 2, 3), keepdim=True)

        x0_cfg_norm = x0_cfg * (std_x0 / std_x0_cfg)
        x0_rescaled = torch.lerp(x0_cfg, x0_cfg_norm, self.rescale)

        return x_norm - x0_rescaled

class CharacteristicGuidancePredictor(CachingNoisePredictor):
    """Implements Characteristic Guidance with Anderson Acceleration

    https://arxiv.org/pdf/2312.07586.pdf"""

    INPUTS = {
        "required": {
            "cond": ("PREDICTION",),
            "uncond": ("PREDICTION",),
            "guidance_scale": ("FLOAT", { "default": 6.0, "step": 0.1, "min": 1.0, "max": 100.0 }),
            "history": ("INT", { "default": 2, "min": 1 }),
            "log_step_size": ("FLOAT", { "default": -3, "step": 0.1, "min": -6, "max": 0 }),
            "log_tolerance": ("FLOAT", { "default": -4.0, "step": 0.1, "min": -6.0, "max": -2.0 }),
            "keep_tolerance": ("FLOAT", { "default": 1.0, "step": 1.0, "min": 1.0, "max": 1000.0 }),
            "reuse_scale": ("FLOAT", { "default": 0.0, "step": 0.0001, "min": 0.0, "max": 1.0 }),
            "max_steps": ("INT", { "default": 20, "min": 5, "max": 1000 }),
            "precondition_gradients": ("BOOLEAN", { "default": True }),
        },
        "optional": {
            "fallback": ("PREDICTION",),
        }
    }

    def __init__(
        self,
        cond,
        uncond,
        guidance_scale,
        history,
        log_step_size,
        log_tolerance,
        keep_tolerance,
        max_steps,
        reuse_scale,
        precondition_gradients = True,
        fallback = None
    ):
        super().__init__()

        self.cond = cond
        self.uncond = uncond
        self.fallback = fallback
        self.scale = guidance_scale
        self.history = history
        self.step_size = 10 ** log_step_size
        self.tolerance = 10 ** log_tolerance
        self.keep_tolerance = keep_tolerance
        self.max_steps = max_steps
        self.reuse = reuse_scale
        self.precondition_gradients = precondition_gradients

        self.cond_deps = set()
        self.uncond_deps = set()
        self.restore_preds = []
        self.prev_dx = None
        self.sample = 0
        self.pbar = None

    def get_conds(self):
        return self.merge_conds(self.cond, self.uncond, self.fallback)

    def get_models(self):
        return self.merge_models(self.cond, self.uncond, self.fallback)

    def get_preds(self):
        return self.merge_preds(self.cond, self.uncond, self.fallback)

    def begin_sampling(self):
        super().begin_sampling()

        self.cond_deps = self.cond.get_preds()
        self.uncond_deps = self.uncond.get_preds()
        self.restore_preds.clear()
        self.prev_dx = None
        self.sample = 0

        if comfy.utils.PROGRESS_BAR_ENABLED:
            self.pbar = tqdm(bar_format="[{rate_fmt}] {desc} ")
            self.pbar.set_description_str("CHG")

    def end_sampling(self):
        super().end_sampling()

        self.cond_deps.clear()
        self.uncond_deps.clear()
        self.restore_preds.clear()
        self.prev_dx = None
        self.sample = 0

        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

    def predict_noise_cond(self, *args):
        return self.predict_noise_sample(self.cond, self.cond_deps, *args)

    def predict_noise_uncond(self, *args):
        return self.predict_noise_sample(self.uncond, self.uncond_deps, *args)

    def predict_noise_fallback(self, *args):
        return self.predict_noise_sample(self.fallback, self.fallback.get_preds(), *args)

    @staticmethod
    def predict_noise_sample(pred, deps, *args):
        for dep in deps:
            dep.begin_sample()

        try:
            return pred.predict_noise(*args)
        finally:
            for dep in deps:
                dep.end_sample()

    def status(self, msg, start=False, end=False):
        if self.pbar is not None:
            if start:
                self.pbar.unpause()

            self.pbar.set_description_str(f"CHG sample {self.sample:>2}: {msg}")

            if end:
                print()
                self.pbar.set_description_str("CHG")

    def progress(self):
        if self.pbar is not None:
            self.pbar.update()

    def restore(self):
        while len(self.restore_preds) > 0:
            pred, cached = self.restore_preds.pop()
            pred.cached_prediction = cached
            del cached

    def predict_noise_uncached(self, x, sigma, model, conds, model_options, seed):
        self.sample += 1
        self.status(f"starting ({len(x)}/{len(x)})", start=True)

        xb, xc, xh, xw = x.shape
        cvg = []
        uncvg = list(range(xb))

        dx_b = []
        g_b = []

        # initial prediction, for regularization, step 1, and fallback
        # p_r = (model(x) - model(x|c)) * sigma
        p_c = self.cond.predict_noise(x, sigma, model, conds, model_options, seed)
        p_u = self.uncond.predict_noise(x, sigma, model, conds, model_options, seed)
        p_r = (p_u - p_c) * sigma[:, None, None, None]

        if self.fallback is not None:
            del p_c, p_u

        # remember predictions for the original latents so we can restore them later
        # end the current sample step since we're about to sample with a different x
        for pred in (self.cond_deps | self.uncond_deps):
            if isinstance(pred, CachingNoisePredictor):
                self.restore_preds.append((pred, pred.cached_prediction))

            pred.end_sample()

        # when dx is 0, we can re-use the predictions made for regularization and save a step
        #
        # dx = 0
        # r = dx - p_r = -p_r
        # v = (model(x + (w+1)*dx) - model(x + w*dx|c)) * sigma = (model(x) - model(x|c)) * sigma = p_r
        #
        # P_r o v = proj(v, r) = proj(p_r, -p_r) = p_r
        # g = dx - P_r o v = -p_r
        #
        # dx' = dx - gamma * g = p_r * gamma

        if self.prev_dx is None:
            start_at = 1
            dx = p_r * self.step_size

            if self.reuse != 0.0:
                self.prev_dx = dx

            if self.history > 1:
                dx_b.append(torch.zeros_like(x))
                g_b.append(-p_r)
        else:
            start_at = 0
            dx = self.prev_dx
            dx *= self.reuse

        self.progress()

        for s in range(start_at, self.max_steps):
            comfy.model_management.throw_exception_if_processing_interrupted()
            self.status(f"step {s+1}/{self.max_steps} ({len(cvg)}/{len(x)})")

            # we only want to step the unconverged samples
            ub = len(uncvg)
            dxu = dx[uncvg]

            sigu = sigma[uncvg]
            cxu = x[uncvg] + dxu * (self.scale + 1)

            # p = (model(x + dx * (scale + 1)) - model(x + dx * scale|c)) * sigma
            p = (
                self.predict_noise_uncond(cxu, sigu, model, conds, model_options, seed) -
                self.predict_noise_cond(cxu - dxu, sigu, model, conds, model_options, seed)
            ) * sigu.view(ub, 1, 1, 1)
            del cxu, sigu

            # g = dx - P o (model(x + dx * (scale + 1)) - model(x + dx * scale|c)) * sigma
            g = dxu - proj(p, dxu - p_r[uncvg])
            del p

            # remember norm before AA to test for convergence
            g_norm = torch.linalg.vector_norm(g, dim=(1, 2, 3)).cpu()

            if self.history > 1:
                dx_b.append(dxu.clone())
                g_b.append(g.clone())

            if len(dx_b) >= 2:
                dx_b[-2] = dxu - dx_b[-2]
                g_b[-2] = g - g_b[-2]

                def as_mat(buf):
                    # (M[], B, C, H, W) -> (B, CHW, M[:-1])
                    return torch.stack([buf[m].view(ub, xc*xh*xw) for m in range(len(buf) - 1)], dim=2)

                # w = argmin_w ||A_g w - b_g||_l2
                a_g = as_mat(g_b)
                a_g_norm = None

                if self.precondition_gradients:
                    a_g_norm = torch.maximum(torch.linalg.vector_norm(a_g, dim=-2, keepdim=True), torch.tensor(1e-04))
                    a_g /= a_g_norm

                w = torch.linalg.lstsq(a_g, g.view(ub, xc*xh*xw, 1)).solution

                # g_AA = b_g - A_g w
                g -= (a_g @ w).view(ub, xc, xh, xw)
                del a_g

                # dx_AA = x_g - A_x w
                a_dx_b = as_mat(dx_b)

                if self.precondition_gradients:
                    a_dx_b /= a_g_norm

                dx[uncvg] -= (a_dx_b @ w).view(ub, xc, xh, xw)
                del a_dx_b, w, a_g_norm

                if len(dx_b) >= self.history:
                    del dx_b[0], g_b[0]

            # dx = dx_AA - gamma g_AA
            dx[uncvg] -= g * self.step_size
            del g

            # until ||g||_l2 <= tolerance * dim(g)
            uidx = []

            tolerance = self.tolerance * xc * xh * xw
            if s == self.max_steps - 1:
                tolerance *= self.keep_tolerance

            for i in reversed(range(len(uncvg))):
                if g_norm[i] <= tolerance:
                    cvg.append(uncvg.pop(i))
                else:
                    uidx.append(i)

            del g_norm

            if len(uncvg) < ub:
                cvg.sort()

            if len(uncvg) == 0:
                self.progress()
                break

            if len(uncvg) < ub:
                uidx.reverse()

                for m in range(len(dx_b)):
                    dx_b[m] = dx_b[m][uidx]
                    g_b[m] = g_b[m][uidx]

            self.progress()

        dx_b.clear()
        g_b.clear()
        del p_r

        result = torch.empty_like(x)

        if len(cvg) != 0:
            self.status(f"sampling ({len(cvg)}/{len(x)})")

            # predict only the converged samples
            dxc = dx[cvg]
            sigc = sigma[cvg]
            cxc = x[cvg] + dxc * self.scale

            # chg = model(x + dx * scale|c) * (scale + 1) - model(x + dx * (scale + 1)) * scale
            result[cvg] = (
                self.predict_noise_cond(cxc, sigc, model, conds, model_options, seed) * (self.scale + 1.0) -
                self.predict_noise_uncond(cxc + dxc, sigc, model, conds, model_options, seed) * self.scale
            )
            del dxc, sigc, cxc

            self.progress()

        if len(uncvg) != 0:
            if self.pbar is None:
                print(f"CHG sample {self.sample}: {len(uncvg)}/{len(x)} samples did not converge")

            if self.fallback is not None:
                self.status(f"fallback ({len(uncvg)}/{len(x)})")

                # in the very special case that nothing converged, we can restore now to hopefully speed-up the fallback
                if len(cvg) == 0:
                    self.restore()
                    result = self.fallback.predict_noise(x, sigma, model, conds, model_options, seed)
                else:
                    result[uncvg] = self.predict_noise_fallback(x[uncvg], sigma[uncvg], model, conds, model_options, seed)

                self.progress()
            else:
                # use vanilla CFG for unconverged samples if no fallback was specified
                result[uncvg] = p_c[uncvg] * (self.scale + 1.0) - p_u[uncvg] * self.scale

            # make sure the unconverged corrections are not reused
            if self.reuse != 0.0:
                for i in uncvg:
                    dx[i].zero_()

        if self.fallback is None:
            del p_c, p_u

        if self.pbar is not None:
            self.status(f"{s+1} steps, {len(uncvg)} unconverged", end=True)

        self.restore()
        return result

class AvoidErasePredictor(NoisePredictor):
    """Implements Avoid and Erase V2 guidance."""

    INPUTS = {
        "required": {
            "positive": ("PREDICTION",),
            "negative": ("PREDICTION",),
            "empty": ("PREDICTION",),
            "erase_scale": ("FLOAT", {
                "default": 0.2,
                "step": 0.01,
                "min": 0.0,
                "max": 1.0,
            }),
        }
    }

    OUTPUTS = { "guidance": "PREDICTION" }

    def __init__(self, positive, negative, empty, erase_scale):
        self.positive_pred = positive
        self.negative_pred = negative
        self.empty_pred = empty
        self.erase_scale = erase_scale

    def get_conds(self):
        return self.merge_conds(self.positive_pred, self.negative_pred, self.empty_pred)

    def get_models(self):
        return self.merge_models(self.positive_pred, self.negative_pred, self.empty_pred)

    def get_preds(self):
        return self.merge_preds(self.positive_pred, self.negative_pred, self.empty_pred)

    def predict_noise(self, x, timestep, model, conds, model_options, seed):
        pos = self.positive_pred.predict_noise(x, timestep, model, conds, model_options, seed)
        neg = self.negative_pred.predict_noise(x, timestep, model, conds, model_options, seed)
        empty = self.empty_pred.predict_noise(x, timestep, model, conds, model_options, seed)

        pos_ind = pos - empty
        neg_ind = neg - empty
        return oproj(pos_ind, neg_ind) - oproj(neg_ind, pos_ind) * self.erase_scale

def dot(a, b):
    return (a * b).sum(dim=(1, 2, 3), keepdims=True)

def proj(a, b):
    a_dot_b = dot(a, b)
    b_dot_b = dot(b, b)
    divisor = torch.where(
        b_dot_b != 0,
        b_dot_b,
        torch.ones_like(b_dot_b)
    )

    return b * (a_dot_b / divisor)

def oproj(a, b):
    return a - proj(a, b)

class ScalePredictor(NoisePredictor):
    INPUTS = {
        "required": {
            "prediction": ("PREDICTION",),
            "scale": ("FLOAT", {
                "default": 1.0,
                "step": 0.01,
                "min": -100.0,
                "max": 100.0,
            })
        }
    }

    def __init__(self, prediction, scale):
        self.inner = prediction
        self.scale = scale

    def get_conds(self):
        return self.inner.get_conds()

    def get_models(self):
        return self.inner.get_models()

    def get_preds(self):
        return self.merge_preds(self.inner)

    def predict_noise(self, x, timestep, model, conds, model_options, seed):
        return self.inner.predict_noise(x, timestep, model, conds, model_options, seed) * self.scale

class CFGPredictor(CachingNoisePredictor):
    INPUTS = {
        "required": {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "cfg_scale": ("FLOAT", {
                "default": 6.0,
                "min": 1.0,
                "max": 100.0,
                "step": 0.5,
            })
        }
    }

    def __init__(self, positive, negative, cfg_scale):
        super().__init__()

        self.positive = convert_cond(positive)
        self.negative = convert_cond(negative)
        self.cfg_scale = cfg_scale

    def get_conds(self):
        return {
            "positive": self.positive,
            "negative": self.negative,
        }

    def get_models(self):
        return self.get_models_from_conds()

    def n_samples(self):
        return 2

    def predict_noise_uncached(self, x, timestep, model, conds, model_options, seed):
        cond, uncond = calc_cond_batch(
            model,
            [conds["positive"], conds["negative"]],
            x,
            timestep,
            model_options
        )

        return uncond + (cond - uncond) * self.cfg_scale

class PerpNegPredictor(CachingNoisePredictor):
    INPUTS = {
        "required": {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "empty": ("CONDITIONING",),
            "cfg_scale": ("FLOAT", {
                "default": 6.0,
                "min": 1.0,
                "max": 100.0,
                "step": 0.5,
            }),
            "neg_scale": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 2.0,
                "step": 0.05,
            })
        }
    }

    def __init__(self, positive, negative, empty, cfg_scale, neg_scale):
        super().__init__()

        self.positive = convert_cond(positive)
        self.negative = convert_cond(negative)
        self.empty = convert_cond(empty)
        self.cfg_scale = cfg_scale
        self.neg_scale = neg_scale

    def get_conds(self):
        return {
            "positive": self.positive,
            "negative": self.negative,
            "empty": self.empty,
        }

    def get_models(self):
        return self.get_models_from_conds()

    def n_samples(self):
        return 3

    def predict_noise_uncached(self, x, timestep, model, conds, model_options, seed):
        cond, uncond, empty = comfy.samplers.calc_cond_batch(
            model,
            [conds["positive"], conds["negative"], conds["empty"]],
            x,
            timestep,
            model_options
        )

        positive = cond - empty
        negative = uncond - empty
        perp_neg = oproj(negative, positive) * self.neg_scale
        return empty + (positive - perp_neg) * self.cfg_scale

NODE_CLASS_MAPPINGS = {
    "SamplerCustomPrediction": SamplerCustomPrediction,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerCustomPrediction": "Sample Predictions",
}

def make_node(predictor, display_name, class_name=None, category="sampling/prediction"):
    if class_name is None:
        class_name = predictor.__name__[:-2] + "ion" # Predictor -> Prediction

    cls = type(class_name, (), {
        "FUNCTION": "get_predictor",
        "CATEGORY": category,
        "INPUT_TYPES": classmethod(lambda cls: predictor.INPUTS),
        "RETURN_TYPES": tuple(predictor.OUTPUTS.values()),
        "RETURN_NAMES": tuple(predictor.OUTPUTS.keys()),
        "get_predictor": lambda self, **kwargs: (predictor(**kwargs),),
    })

    setattr(sys.modules[__name__], class_name, cls)
    NODE_CLASS_MAPPINGS[class_name] = cls
    NODE_DISPLAY_NAME_MAPPINGS[class_name] = display_name

make_node(ConditionedPredictor, "Conditioned Prediction")
make_node(CombinePredictor, "Combine Predictions", class_name="CombinePredictions")
make_node(InterpolatePredictor, "Interpolate Predictions", class_name="InterpolatePredictions")
make_node(SwitchPredictor, "Switch Predictions", class_name="SwitchPredictions")
make_node(EarlyMiddleLatePredictor, "Switch Early/Middle/Late Predictions")
make_node(ScaledGuidancePredictor, "Scaled Guidance Prediction")
make_node(CharacteristicGuidancePredictor, "Characteristic Guidance Prediction")
make_node(AvoidErasePredictor, "Avoid and Erase Prediction")
make_node(ScalePredictor, "Scale Prediction")
make_node(CFGPredictor, "CFG Prediction")
make_node(PerpNegPredictor, "Perp-Neg Prediction")
