<div align="center">

# ComfyUI Omnichar

**One `.char` format for consistent, portable characters.**

Official `.char` integration for ComfyUI. Build a character once and keep the same face, body,
clothes and voice across every model you already use. Currently MiniMax H3, Krea 2, FLUX.2 dev and
klein 9B / 4B, with consistent voice on MiniMax H3.

[**Install**](#installation) ·
[**Omnichar Studio**](https://github.com/omnichar/OmniChar) ·
[**Try it in your browser**](https://cloud.omnichar.org) ·
[**omnichar-sdk**](packages/omnichar-sdk/README.md) ·
[**Example workflows**](workflows)

[![Website][website-shield]][website-url]
[![License: GPLv3][license-shield]][license-url]
[![Python 3.10+][python-shield]][python-url]
[![Latest release][release-shield]][release-url]
<br>
[![Discord][discord-shield]][discord-url]
[![Reddit][reddit-shield]][reddit-url]

<img width="1590" alt="Omnichar nodes in a ComfyUI graph" src="https://raw.githubusercontent.com/omnichar/ComfyUI-Omnichar/main/public/image.png" />

</div>

[website-shield]: https://img.shields.io/badge/Website-omnichar.org-blue?style=flat
[website-url]: https://omnichar.org
[license-shield]: https://img.shields.io/badge/License-GPLv3-blue?style=flat
[license-url]: LICENSE
[python-shield]: https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python&logoColor=white
[python-url]: https://www.python.org/downloads/
[release-shield]: https://img.shields.io/github/v/release/omnichar/ComfyUI-Omnichar?style=flat&label=Release&color=blue
[release-url]: ../../releases/latest
[discord-shield]: https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white&style=flat
[discord-url]: https://discord.gg/cSUS88VdY9
[reddit-shield]: https://img.shields.io/badge/Reddit-r%2Fomnichar-FF4500?logo=reddit&logoColor=white&style=flat
[reddit-url]: https://www.reddit.com/r/omnichar/

A `.char` holds a character's reference images, its locked description, and often a trained LoRA.
Build one here with Encode Character, or in [Omnichar Studio](https://omnichar.org) on your own GPU
or [Omnichar Cloud](https://cloud.omnichar.org). The same file then feeds FLUX.2, MiniMax H3 and
anything else that takes references.

## Features

- **Character Files**: Open a `.char` built in Omnichar Studio or Cloud
- **Reference Images**: One batch, or one per numbered slot, both from the same resolved set
- **Positions Kept**: Reference order is preserved, because a prompt addresses images by number
- **Conditioning**: Wire a CLIP to get conditioning straight out, or take the prompt as text
- **Trained LoRA**: Applied to MODEL and CLIP when the character carries one
- **Build Characters**: Encode face, body and wardrobe references into a new `.char`, three slots each
- **Voice**: Store a short voice clip in the character and send it to MiniMax H3 when the prompt has dialogue
- **Python Library**: Omnichar's standalone package for `.char` integration, no dependencies

## Requirements

- ComfyUI
- Python 3.10+
- `omnichar-sdk` (installed from `requirements.txt`)

## Installation

1. Go to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/omnichar/ComfyUI-Omnichar
   cd ComfyUI-Omnichar
   ```

3. Install the reader:
   ```bash
   pip install -r requirements.txt
   ```

4. Restart ComfyUI

## Where Characters Live

Put `.char` files in `ComfyUI/models/characters/`. The loader lists whatever is there.

To share one folder with Omnichar Studio, set `INLINE_CHARACTERS_DIR` to its characters directory
and both read the same files.

## Nodes

| Node | Inputs | Outputs |
| --- | --- | --- |
| Load Character | `char`, `char_path` | `char` |
| Decode Character | `char`, `style`, `clip`, `prompt`, `arch`, `max_references`, `size_from`, `fit`, `voice`, `audio_position` | `conditioning`, `references`, `refs`, `sheet`, `prompt`, `voice` |
| Character Reference | `refs`, `index` | `image`, `role`, `count` |
| Character References Split | `refs` | `image_0` to `image_4`, `count` |
| Character Reference Latent | `conditioning`, `refs`, `vae` | `conditioning` |
| Apply Character LoRA | `model`, `clip`, `char`, `strength`, `arch`, `min_key_coverage` | `model`, `clip` |
| Encode Character | `name`, `description`, `resolution`, `face`/`body`/`cloths` (3 slots each), `voice` | `char` |
| Save Character | `char`, `filename`, `overwrite` | `path` |

## Guide

A character is a few reference images plus a description. Encode Character sorts them by role,
so face comes first and the prompt numbers follow that order.

<table>
  <tr>
    <td align="center"><img src="workflows/images_sia/face.png" width="110"></td>
    <td align="center"><img src="workflows/images_sia/body.jpg" width="110"></td>
    <td align="center"><img src="workflows/images_sia/cloth1.jpg" width="110"></td>
    <td align="center"><img src="workflows/images_sia/cloth2.jpg" width="110"></td>
  </tr>
  <tr>
    <td align="center"><code>face</code></td>
    <td align="center"><code>body</code></td>
    <td align="center"><code>cloths</code></td>
    <td align="center"><code>cloths_2</code></td>
  </tr>
</table>

Those four go into Encode Character, which writes `sia.char`. Save Character puts it in
`ComfyUI/models/characters/`, and Load Character picks it up from there.

Decode Character turns a character into a prompt and a resolved reference list. Models
that take one batch read `references`. Models with numbered slots, like MiniMax H3, take `refs` into
a Character References Split node, or a Character Reference node per slot. Edit models that read references as latents, like FLUX.2, take
`refs` into a Character Reference Latent node on both the positive and the negative conditioning.

### Voice

A character can carry a voice for MiniMax H3 Reference to Video. Wire Load Audio into Encode
Character's `voice` input: about 30 seconds of clean speech, no music. On Decode Character, use the
`token` style and wire `voice` into the H3 node's `ref_audio_0`.

With `voice` on `auto`, the clip is sent only when the prompt has dialogue:

| Prompt | Voice |
| --- | --- |
| `sia faces the camera and says "I built this look one frame at a time."` | sent |
| `close-up of sia as she whispers to someone off camera` | sent |
| `sia walks along the lake at dusk` | not sent |

The match is on words, so "no speaking" still counts as dialogue; set `never` for a silent shot
like that, or `always` to force the voice in. H3 speaks new words in a voice like the clip, so
expect a likeness rather than a copy.

### Workflows

- [Build a `.char`](workflows/character_encode.json) from face, body and wardrobe references
- [FLUX.2 Klein 9B](workflows/flux_klein_9b_image_char.json), references as latents, to an image
- [MiniMax H3](workflows/minimax_h3_char_video.json), references in numbered slots, to a video
- [Build a `.char` with a voice](workflows/voice/encode_char_with_voice.json), a face reference plus a voice clip from Load Audio
- [MiniMax H3 with voice](workflows/voice/generate_with_voice_minimaxh3.json), the character's voice wired into `ref_audio_0`, to a talking video

## Python Library

Omnichar's standalone package for `.char` integration. Install it anywhere, not only in ComfyUI:

```bash
pip install omnichar-sdk
```

See [packages/omnichar-sdk/README.md](packages/omnichar-sdk/README.md).

## License

`packages/omnichar-sdk/` is Apache-2.0, so closed-source tools can read `.char` files.
Everything else is GPL-3.0-or-later, because ComfyUI is.

## Links

- [Omnichar Studio](https://omnichar.org)
- [Omnichar Cloud](https://cloud.omnichar.org)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Report an issue](https://github.com/omnichar/ComfyUI-Omnichar/issues)
