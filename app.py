import modal
from typing import List, Optional, Union, Dict, Any, Callable
import numpy as np
import torch
from PIL import Image

app = modal.App(name="hunyuan-lora-app")

CACHE_DIR = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install_from_requirements(
        "requirements_modal.txt"
    )
    .env(
        {
            "HF_HUB_CACHE": CACHE_DIR,
        }
    )
)

DEFAULT_PROMPT_TEMPLATE = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "4. background environment, light, style and atmosphere."
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}

with image.imports():
    from typing import List, Optional, Union, Dict, Any, Callable
    import cv2
    import numpy as np
    import torch
    import torchvision.transforms.v2 as transforms
    from diffusers import HunyuanVideoPipeline
    from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
    from diffusers.models import HunyuanVideoTransformer3DModel
    from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoPatchEmbed, HunyuanVideoTransformer3DModel
    from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import retrieve_timesteps
    from diffusers.pipelines.hunyuan_video.pipeline_output import HunyuanVideoPipelineOutput
    from diffusers.utils import export_to_video
    from PIL import Image
    from io import BytesIO

lora = "/lora"
videos = "/videos"

volume = modal.Volume.from_name("hunyuan-lora", create_if_missing=True)
store = modal.Volume.from_name("hunyuan-store", create_if_missing=True)

@app.cls(
    image=image,
    gpu="A100-80GB",
    volumes={lora: volume, videos: store},
    timeout=800 # in seconds
)
class Model:
    @modal.enter()
    def enter(self):
        import os
        from huggingface_hub import hf_hub_download 

        # Download checkpoint if not exists
        ckpt_path = lora + "/i2v.sft"

        if not os.path.exists(ckpt_path):
            print("Downloading checkpoint files.")
            hf_hub_download("dashtoon/hunyuan-video-keyframe-control-lora", "i2v.sft", local_dir=lora)

        return True
    
    def _resize_image_to_bucket(self, image: Union[Image.Image, np.ndarray], bucket_reso: tuple[int, int]) -> np.ndarray:
        """
        Resize the image to the bucket resolution.
        """

        is_pil_image = isinstance(image, Image.Image)
        if is_pil_image:
            image_width, image_height = image.size
        else:
            image_height, image_width = image.shape[:2]

        if bucket_reso == (image_width, image_height):
            return np.array(image) if is_pil_image else image

        bucket_width, bucket_height = bucket_reso

        scale_width = bucket_width / image_width
        scale_height = bucket_height / image_height
        scale = max(scale_width, scale_height)
        image_width = int(image_width * scale + 0.5)
        image_height = int(image_height * scale + 0.5)

        if scale > 1:
            image = Image.fromarray(image) if not is_pil_image else image
            image = image.resize((image_width, image_height), Image.LANCZOS)
            image = np.array(image)
        else:
            image = np.array(image) if is_pil_image else image
            image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)

        # crop the image to the bucket resolution
        crop_left = (image_width - bucket_width) // 2
        crop_top = (image_height - bucket_height) // 2
        image = image[crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width]

        return image
    
    def _call_pipe(
        self,
        pipe,
        prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        num_inference_steps: int = 50,
        sigmas: List[float] = None,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        # callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]]
        callback_on_step_end = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        max_sequence_length: int = 256,
        image_latents: Optional[torch.Tensor] = None,
    ):

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        pipe.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds,
            callback_on_step_end_tensor_inputs,
            prompt_template,
        )

        pipe._guidance_scale = guidance_scale
        pipe._attention_kwargs = attention_kwargs
        pipe._current_timestep = None
        pipe._interrupt = False

        device = pipe._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_template=prompt_template,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        transformer_dtype = pipe.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            pipe.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )

        # 5. Prepare latent variables
        num_channels_latents = pipe.transformer.config.in_channels
        num_latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
        latents = pipe.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_latent_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 6. Prepare guidance condition
        guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=device) * 1000.0

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
        pipe._num_timesteps = len(timesteps)

        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if pipe.interrupt:
                    continue

                pipe._current_timestep = t
                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = pipe.transformer(
                    hidden_states=torch.cat([latent_model_input, image_latents], dim=1),
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    pooled_projections=pooled_prompt_embeds,
                    guidance=guidance,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(pipe, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()
        pipe._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
            video = pipe.vae.decode(latents, return_dict=False)[0]
            video = pipe.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        pipe.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return HunyuanVideoPipelineOutput(frames=video)

    @modal.method()
    def run_inference(self, prompt, frame1_bytes, frame2_bytes, n_frames=77, height=1280, width=720, seed=0, file_name="output.mp4", num_inference_steps=50):

        video_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        model_id = "hunyuanvideo-community/HunyuanVideo"
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
        pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.bfloat16)

        pipe.to("cuda")
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        with torch.no_grad():  # enable image inputs
            initial_input_channels = pipe.transformer.config.in_channels
            new_img_in = HunyuanVideoPatchEmbed(
                patch_size=(pipe.transformer.config.patch_size_t, pipe.transformer.config.patch_size, pipe.transformer.config.patch_size),
                in_chans=pipe.transformer.config.in_channels * 2,
                embed_dim=pipe.transformer.config.num_attention_heads * pipe.transformer.config.attention_head_dim,
            )
            new_img_in = new_img_in.to(pipe.device, dtype=pipe.dtype)
            new_img_in.proj.weight.zero_()
            new_img_in.proj.weight[:, :initial_input_channels].copy_(pipe.transformer.x_embedder.proj.weight)

            if pipe.transformer.x_embedder.proj.bias is not None:
                new_img_in.proj.bias.copy_(pipe.transformer.x_embedder.proj.bias)

            pipe.transformer.x_embedder = new_img_in

            LORA_PATH = lora + "/i2v.sft"
            lora_state_dict = pipe.lora_state_dict(LORA_PATH)
            transformer_lora_state_dict = {f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.") and "lora" in k}
            pipe.load_lora_into_transformer(transformer_lora_state_dict, transformer=pipe.transformer, adapter_name="i2v", _pipeline=pipe)
            pipe.set_adapters(["i2v"], adapter_weights=[1.0])
            pipe.fuse_lora(components=["transformer"], lora_scale=1.0, adapter_names=["i2v"])
            pipe.unload_lora_weights()

            cond_frame1 = Image.open(BytesIO(frame1_bytes)).convert('RGB')
            cond_frame1 = self._resize_image_to_bucket(cond_frame1, bucket_reso=(width, height))

            cond_frame2 = Image.open(BytesIO(frame2_bytes)).convert('RGB')
            cond_frame2 = self._resize_image_to_bucket(cond_frame2, bucket_reso=(width, height))

            cond_video = np.zeros(shape=(n_frames, height, width, 3))
            cond_video[0], cond_video[-1] = np.array(cond_frame1), np.array(cond_frame2)

            cond_video = torch.from_numpy(cond_video.copy()).permute(0, 3, 1, 2)
            cond_video = torch.stack([video_transforms(x) for x in cond_video], dim=0).unsqueeze(0)

            image_or_video = cond_video.to(device="cuda", dtype=pipe.dtype)
            image_or_video = image_or_video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]
            cond_latents = pipe.vae.encode(image_or_video).latent_dist.sample()
            cond_latents = cond_latents * pipe.vae.config.scaling_factor
            cond_latents = cond_latents.to(dtype=pipe.dtype)

            video = self._call_pipe(
                pipe,
                prompt=prompt,
                num_frames=n_frames,
                num_inference_steps=num_inference_steps,
                image_latents=cond_latents,
                width=width,
                height=height,
                guidance_scale=6.0,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).frames[0]

            return export_to_video(video, fps=24, output_video_path=videos + "/" + file_name)


@app.local_entrypoint()
def main():
    from pathlib import Path
    
    a = Path("io/char01.png").read_bytes()
    b = Path("io/char02.png").read_bytes()

    """
    # Recommended Settings
    1. The model works best on human subjects. Single subject images work slightly better.
    2. It is recommended to use the following image generation resolutions 720x1280, 544x960, 1280x720, 960x544.
    3. It is recommended to set frames from 33 upto 97. Can go upto 121 frames as well (but not tested much).
    4. Prompting helps a lot but works even without. The prompt can be as simple as just the name of the object you want to generate or can be detailed.
    5. num_inference_steps is recommended to be 50, but for fast results you can use 30 as well. Anything less than 30 is not recommended.
    """

    result = Model().run_inference.remote(
        prompt="A girl changing her expression",
        frame1_bytes=a,
        frame2_bytes=b,
        n_frames=33,
        height=960,
        width=544,
        seed=1,
        file_name="output3.mp4",
        num_inference_steps=50,
    )

    print(f"Video saved at: {result}")

    return