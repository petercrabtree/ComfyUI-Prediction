import comfy
import torch

class SelectSigmas:
    """Selects a subest of sigmas."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "select": ("STRING", { "multiline": False, "default": "mod 2" }),
                "chained": ("BOOLEAN", { "default": False })
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("selection",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/sigmas"

    def get_sigmas(self, sigmas, select, chained):
        count = len(sigmas)
        if not chained:
            count -= 1

        if select.startswith("mod "):
            arg = int(select[4:])
            mask = list(range(arg - 1, count, arg))
        else:
            mask = []
            for idx in select.split(","):
                idx = idx.strip()
                if ":" in idx:
                    start, end = idx.split(":")
                    start = int(start) if start != "" else 0
                    end = int(end) if end != "" else count

                    mask.extend(
                        idx for idx in range(start, end, 1 if start <= end else -1)
                        if -count <= idx < count
                    )
                elif idx != "":
                    idx = int(idx)
                    if -count <= idx < count:
                        mask.append(idx)

        return sigmas[mask],

class SplitAtSigma:
    """Splits a descending list of sigmas at the specfied sigma."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.0}),
            }
        }

    RETURN_TYPES = ("SIGMAS", "SIGMAS")
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/sigmas"

    def get_sigmas(self, sigmas, sigma):
        index = 0
        while sigmas[index].item() > sigma:
            index += 1

        return sigmas[:index+1], sigmas[index:]

class LogSigmas:
    """Logs a list of sigmas to the console."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "message": ("STRING", { "multiline": False, "default": "" }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "log_sigmas"
    CATEGORY = "sampling/custom_sampling/sigmas"
    OUTPUT_NODE = True

    def log_sigmas(self, sigmas, message):
        print(f"{message or 'SIGMAS'}: {sigmas.tolist()}")
        return ()

NODE_CLASS_MAPPINGS = {
    "SelectSigmas": SelectSigmas,
    "SplitAtSigma": SplitAtSigma,
    "LogSigmas": LogSigmas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectSigmas": "Select Sigmas",
    "SplitAtSigma": "Split at Sigma",
    "LogSigmas": "Log Sigmas",
}
