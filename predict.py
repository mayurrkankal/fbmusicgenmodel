# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torchaudio
import torch
import os


class Predictor(BasePredictor):
    def setup(self) :
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MusicGen.get_pretrained("medium")


    def predict(
        self,
        self.model: str = Input(
            description="Model to use for generation.",
            default="medium"),
        prompt: str = Input(
                        description="A description of the music you want to generate.", default=None
                        ),
        duration: int = Input(
                        description="Duration of the generated audio in seconds.", default=15, ge=5,le=30),
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        model = self.model

        if prompt is None and duration is None:
            raise ValueError("Must provide prompt and duration")

        set_generation_params = lambda duration: model.set_generation_params(
            duration=duration,
            )
        if prompt and duration:
            set_generation_params(duration)
            wav = model.generate([prompt], progress=True)

        audio_write(
            "out",
            wav[0].cpu(),
            model.sample_rate,
            strategy=normalization_strategy,
             )
        wav_path = "out.wav"
        path = wav_path
        return Path(path)
