from sgm.inference.api import (
    SamplingParams,
    SamplingPipeline,
    ModelArchitecture
)


def test_txt2img(pipeline: SamplingPipeline):
        output = pipeline.text_to_image(
            params=SamplingParams(steps=10),
            prompt="A professional photograph of an astronaut riding a pig",
            negative_prompt="",
            samples=1,
        )

        assert output is not None

arch = ModelArchitecture("stable-diffusion-xl-v1-base", None)
pipeline = SamplingPipeline(arch, "implementations/sgm_/models/sd_xl_base_1.0.safetensors", "implementations/sgm_/config.yaml")
test_txt2img(pipeline)