# omnichar-sdk

Read Omnichar Studio `.char` character files from Python.

A `.char` holds a character's reference images, its locked description, and often a trained LoRA.
This package opens one and hands back those parts. It reads; it never writes.

```python
from omnichar_sdk import Character

char = Character.open("Ada.char")

char.get_description()                       # the locked description
char.get_references(arch="flux2-klein")      # the compiled reference set, in prompt order
char.get_lora()                              # the trained adapter, with a portability verdict
char.get_prompt(style="ordinal")             # text that binds the character to its positions
char.get_voice()                             # the stored voice clip, or None
char.save_reference_sheet("ada.png")         # all the references as one numbered PNG
```

## Reference sheets

`reference_sheet()` lays the references out in a grid and numbers each cell. The numbers are the
point: a position is what a prompt refers to, so a sheet without them cannot be checked against
the prompt it belongs to. Pass the same `first_position` to both and they agree.

```python
char.reference_sheet(arch="flux2-klein", first_position=1)  # a PIL image
char.reference_sheet_png()                                  # PNG bytes
char.save_reference_sheet("out/ada.png")                    # writes it, adds .png if missing
```

It takes the same `arch`, `role`, `limit` and `origin` filters as `get_references`, plus
`columns`, `cell`, `fit`, `labels` and `background`. It needs the `images` extra.

## Install

```
pip install omnichar-sdk              # reads everything; needs only the standard library
pip install 'omnichar-sdk[images]'    # adds Pillow, for decoding and resizing references
```

The base install has no dependencies on purpose. It is meant to drop into ComfyUI, A1111, SwarmUI
or a batch job without competing with whatever that host has pinned, so it never converts images to
tensors. That belongs to the host.

## From a shell

```
omnichar-sdk inspect Ada.char --json
omnichar-sdk extract Ada.char -o out/
omnichar-sdk prompt Ada.char --style token
```

`inspect --json` prints the whole character record, so a tool in any language can read a `.char`
through this command without a Python binding.

## Notes

`style` picks the addressing a model was trained on. FLUX.2 reads ordinal prose, MiniMax H3 reads
`<Picture N>`, Seedance reads `@ImageN`. `description-only` drops positions, which is what a LoRA
needs. `first_position` sets the number the first reference gets, so a prompt and a sheet built at
the same offset agree about which image is image one.

`limit` divides the slots between face, body and outfit rather than cutting the end of the list,
so a character does not lose its wardrobe when a model takes fewer references than it has.

`common_size` picks the size a batch is built at and `fit` resizes onto it: `pad` letterboxes,
`cover` crops, `stretch` distorts.

A character's LoRA records the strength it was judged at. An overfitted adapter is only usable
turned down, so prefer the recorded value over 1.0.

## Building a character

`encode_character` writes a new `.char` from `(image, role)` pairs and compiles a reference set
for each architecture, so the result applies without a rebuild:

```python
from omnichar_sdk import encode_character, write

doc = encode_character("Ada", "A woman with short dark hair.", [(face, "face"), (body, "body")])
write("Ada.char", doc)
```

To give a character a voice, pass two PCM WAV clips: the sample as recorded, and the mono copy a
model hears, which needs at least 3 seconds of sound. `wanted` applies Omnichar's rule for when to
send the voice, which is when the prompt has dialogue:

```python
from omnichar_sdk import set_voice, wanted

set_voice(doc, sample_wav, mono_wav, source_name="ada.wav")
if wanted("auto", prompt):
    text = char.get_prompt(style="token", voice_position=1)
```

It does not compute identity vectors. Those need face encoders this package does not ship, so
Omnichar Studio's verify and continuity features stay unavailable until it re-encodes the file.
Rendering is unaffected.

## Licence

Apache-2.0. The rest of the repository, including the node pack, is GPL-3.0-or-later.
