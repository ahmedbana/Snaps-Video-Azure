from __future__ import annotations

import os
import time
import uuid
import subprocess
from fractions import Fraction

import requests
import numpy as np
import imageio_ffmpeg
import torch
import torchaudio
import io as std_io

from comfy_api.latest import io, Input, InputImpl, Types


class AzureVideoBlobUploader(io.ComfyNode):
    """
    Snaps Upload Video to Azure
    ===========================
    Accepts a VIDEO input (same type produced by all Snaps/VHS video nodes),
    encodes it in-memory to MP4 (no temp file), and uploads the bytes directly
    to Azure Blob Storage via SAS URL — identical to how the image node
    uploads PNG bytes.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AzureVideoBlobUploader",
            display_name="🎬 SNAPS Upload Video to Azure",
            category="cloud/azure",
            description=(
                "Uploads a VIDEO to Azure Blob Storage via SAS URL and fires a webhook."
            ),
            inputs=[
                io.Video.Input("video", tooltip="The video to upload."),
                io.String.Input(
                    "base_url",
                    default="https://snapsai.blob.core.windows.net/output/previews/",
                    multiline=False,
                    tooltip="Azure Blob container base URL (must end with /).",
                ),
                io.String.Input(
                    "sas_token",
                    default="",
                    multiline=False,
                    tooltip="Azure SAS token (including leading '?').",
                ),
                io.String.Input(
                    "file_name",
                    default="output.mp4",
                    multiline=False,
                    tooltip="Hint for the output file name.",
                ),
                io.String.Input(
                    "generation_id",
                    default="",
                    multiline=False,
                    tooltip="Generation ID embedded in the auto-generated filename and webhook.",
                ),
                io.String.Input(
                    "webhook_url",
                    default="https://api.bysnaps.ai/runpod-output",
                    multiline=False,
                    tooltip="Webhook endpoint to notify after upload.",
                ),
                io.String.Input(
                    "scene_order",
                    default="1",
                    multiline=False,
                    tooltip="Scene order value sent in the webhook payload.",
                ),
                io.String.Input(
                    "type",
                    default="Scene",
                    multiline=False,
                    tooltip="Type value sent in the webhook payload.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="uploaded_url"),
            ],
        )

    @classmethod
    def execute(
        cls,
        video: Input.Video,
        base_url: str,
        sas_token: str,
        file_name: str,
        generation_id: str,
        webhook_url: str,
        scene_order: str,
        type: str,
    ) -> io.NodeOutput:
        try:
            # ── Ensure base URL ends with slash ──────────────────────────────
            if not base_url.endswith("/"):
                base_url += "/"

            # ── Decode video frames (same pattern as all Snaps video nodes) ──
            comp = video.get_components()
            frames: torch.Tensor = comp.images       # [N, H, W, C] float 0-1
            fps: float = float(comp.frame_rate)

            N, H, W, C = frames.shape
            print(f"[AzureVideoBlobUploader] Encoding {N} frames @ {fps:.3f} fps  ({W}x{H})")

            # Convert to uint8 RGB numpy array — same as image node converts to PNG
            frames_np = (frames.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            # ── Handle audio if present ──────────────────────────────────────
            audio = comp.audio
            audio_url = None
            if audio is not None:
                waveform = audio["waveform"]
                sample_rate = audio["sample_rate"]
                if waveform.dim() == 3:
                    waveform = waveform.squeeze(0)
                elif waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                
                # Write standard WAV to memory
                wav_io = std_io.BytesIO()
                torchaudio.save(wav_io, waveform.cpu(), sample_rate, format="wav")
                audio_bytes = wav_io.getvalue()
                
                # Upload to Azure so ffmpeg can download it natively
                timestamp = int(time.time())
                random_id = str(uuid.uuid4())[:8]
                audio_filename = (
                    f"{timestamp}_{generation_id}_{random_id}_audio.wav"
                    if generation_id
                    else f"{timestamp}_{random_id}_audio.wav"
                )
                
                upload_audio_url = f"{base_url}{audio_filename}{sas_token}"
                print(f"[AzureVideoBlobUploader] Uploading temporary audio ({len(audio_bytes)} bytes) to Azure…")
                
                headers = {
                    "x-ms-blob-type": "BlockBlob",
                    "Content-Type": "audio/wav",
                }
                response = requests.put(upload_audio_url, headers=headers, data=audio_bytes)
                if response.status_code == 201:
                    audio_url = upload_audio_url
                else:
                    print(f"[AzureVideoBlobUploader] Warning: failed to upload temporary audio: {response.text}")

            # ── Encode to MP4 bytes in-memory via ffmpeg pipe ────────────────
            # Pipe raw RGB24 frames → ffmpeg stdin
            # ffmpeg writes fragmented MP4 to stdout (no temp file, no seek needed)
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{W}x{H}",
                "-pix_fmt", "rgb24",
                "-r", str(fps),
                "-i", "pipe:0",
            ]
            
            if audio_url:
                cmd.extend(["-i", audio_url])
                
            cmd.extend([
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "23",
            ])
            
            if audio_url:
                cmd.extend([
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest"
                ])
                
            cmd.extend([
                # fragmented MP4: streams directly to pipe without needing seek
                "-movflags", "frag_keyframe+empty_moov+default_base_moof",
                "-f", "mp4",
                "pipe:1",
            ])

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            video_bytes, stderr = proc.communicate(input=frames_np.tobytes())

            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace")
                return io.NodeOutput(f"❌ Encoding error: {err}")

            print(f"[AzureVideoBlobUploader] Encoded {len(video_bytes):,} bytes — uploading to Azure…")

            # ── Build filename ────────────────────────────────────────────────
            timestamp = int(time.time())
            random_id = str(uuid.uuid4())[:8]
            filename = (
                f"{timestamp}_{generation_id}_{random_id}.mp4"
                if generation_id
                else f"{timestamp}_{random_id}.mp4"
            )

            # ── Upload to Azure Blob Storage (identical to image node) ────────
            upload_url = f"{base_url}{filename}{sas_token}"
            sas_preview = sas_token[:30] + "..." if len(sas_token) > 30 else sas_token
            print(f"[AzureVideoBlobUploader] Target URL: {base_url}{filename}  SAS: {sas_preview}")

            headers = {
                "x-ms-blob-type": "BlockBlob",
                "Content-Type": "video/mp4",
            }

            response = requests.put(upload_url, headers=headers, data=video_bytes)

            if response.status_code != 201:
                return io.NodeOutput(
                    f"❌ Upload failed: {response.status_code} - {response.text}"
                )

            # ── Fire webhook (same payload shape as image node) ───────────────
            webhook_payload = {
                "status": "scene-completed",
                "generationId": generation_id if generation_id else "default",
                "sceneResult": {
                    "sceneOrder": scene_order,
                    "type": type,
                    "processedVideoUrl": upload_url,
                    "sceneBlurred": "false",
                    "status": "completed",
                },
            }

            try:
                webhook_response = requests.post(
                    webhook_url,
                    json=webhook_payload,
                    headers={"Content-Type": "application/json"},
                )
                if webhook_response.status_code in [200, 201, 202]:
                    result_url = upload_url
                else:
                    result_url = (
                        f"✅ Upload successful: {upload_url} | "
                        f"❌ Webhook failed: {webhook_response.status_code}"
                    )
            except Exception as webhook_error:
                result_url = (
                    f"✅ Upload successful: {upload_url} | "
                    f"❌ Webhook error: {str(webhook_error)}"
                )

            print(f"[AzureVideoBlobUploader] Done → {result_url}")
            return io.NodeOutput(result_url)

        except Exception as e:
            return io.NodeOutput(f"❌ Upload error: {str(e)}")


# ── Node registrations ────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "AzureVideoBlobUploader": AzureVideoBlobUploader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AzureVideoBlobUploader": "🎬 SNAPS Upload Video to Azure",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
