import os
import uuid
import numpy as np
import soundfile as sf
import torch
from encodec import EncodecModel
from encodec.utils import convert_audio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load EnCodec model
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
model.eval()


class EncodedAudio(BaseModel):
    encodedAudio: list[float]
    shape: list[int]


@app.post("/upload")
async def upload_audio(audio: EncodedAudio):
    waveform = np.array(audio.encodedAudio).reshape(audio.shape)
    waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    waveform_22k = convert_audio(waveform_tensor, sr=24000, target_sr=22050, target_channels=1)
    waveform_22k = waveform_22k.squeeze().numpy()

    output_dir = "decoded_audio"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{uuid.uuid4().hex}.wav")

    sf.write(output_path, waveform_22k, 22050)

    return {"message": "Audio encoded and saved successfully"}
